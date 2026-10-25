"""Redacted run summaries and event-driven reflection, never session transcripts.

The host supplies agent_model and verifies outcome/contradicts before ingestion.
Run IDs are immutable: replaying an event cannot rewrite evidence or reopen work.
Explicit on_request reflection is independent of the automatic mechanism switches.
"""
from __future__ import annotations

import json
import time

from .learning import MAX_LIST, clean_ids, clean_run_id, clean_text, error_signature

LONG_STEPS = 8


def _run_id(value):
    return clean_run_id(value)


def _clean_event(event):
    """Whitelist and bound every persisted field; never stringify arbitrary data."""
    if not isinstance(event, dict):
        raise ValueError('A run must be an object')
    run_id = _run_id(event.get('run_id'))
    origin = event.get('origin', 'agent')
    outcome = event.get('outcome')
    steps = event.get('steps', 0)
    correction = event.get('user_correction', False)
    if origin not in ('agent', 'reviewer', 'eval'):
        raise ValueError('origin must be agent, reviewer or eval')
    if outcome not in ('success', 'failure'):
        raise ValueError('outcome must be success or failure')
    if type(steps) is not int or not 0 <= steps <= 2**63 - 1:
        raise ValueError('steps must be a non-negative SQLite integer')
    if type(correction) is not bool:
        raise ValueError('user_correction must be a boolean')
    model = event.get('agent_model', '')
    if not isinstance(model, str):
        raise ValueError('agent_model must be a string')
    errors = event.get('errors', [])
    errors = errors if isinstance(errors, list) else []
    return {
        'run_id': run_id, 'agent_model': clean_text(model, 200),
        'origin': origin, 'task': clean_text(event.get('task', '')),
        'outcome': outcome, 'steps': steps, 'user_correction': correction,
        'errors': [clean_text(error) for error in errors[:MAX_LIST]
                   if isinstance(error, str) and error.strip()],
        'skills_used': list(dict.fromkeys(clean_ids(event.get('skills_used')))),
        'contradicts': list(dict.fromkeys(clean_ids(event.get('contradicts')))),
        'summary': clean_text(event.get('summary', '')),
    }


def _trigger(learning, run_id, event, tier, now):
    learning.db.execute('''INSERT OR IGNORE INTO triggers
        (run_id, event, target_tier, status, detail, ts) VALUES (?,?,?,'pending','',?)''',
                        (run_id, event, tier, now))


def ingest_run(learning, event: dict, agent_model: str) -> dict:
    """Store one bounded summary and its signatures, atomically with its triggers.

    Earlier means ingestion order, including when clock timestamps tie or regress.
    Unknown keys (including tier, transcript and reasoning) are always discarded.
    Returns the stored tier and trigger names; dedup means this ID already existed.
    """
    # The host that ran the agent names the model; the event cannot choose its own tier.
    if not isinstance(agent_model, str) or not agent_model.strip():
        raise ValueError('ingest_run needs the host-attested agent_model')
    if not isinstance(event, dict):
        raise ValueError('A run must be an object')
    if event.get('agent_model', agent_model) != agent_model:
        raise ValueError('Event names a different agent_model than the host')
    data = _clean_event({**event, 'agent_model': agent_model})
    tier = learning.tier(agent_model)
    run_id = data['run_id']
    now = time.time()
    signatures = {error_signature(error) for error in data['errors']}
    with learning.db:
        inserted = learning.db.execute('''INSERT OR IGNORE INTO runs
            (run_id, agent_model, tier, origin, outcome, steps, data, ts)
            VALUES (?,?,?,?,?,?,?,?)''', (run_id, data['agent_model'], tier, data['origin'],
                                        data['outcome'], data['steps'],
                                        json.dumps(data, ensure_ascii=False), now)).rowcount
        if inserted:
            # Only earlier agent runs count: reviewer and eval failures are not experience.
            repeated = any(learning.db.execute(
                "SELECT 1 FROM signatures s JOIN runs r ON r.run_id=s.run_id "
                "WHERE s.signature=? AND r.origin='agent' LIMIT 1", (signature,)
            ).fetchone() for signature in signatures)
            learning.db.executemany('''INSERT INTO signatures
                (signature, run_id, outcome, ts) VALUES (?,?,?,?)''',
                                    [(s, run_id, data['outcome'], now) for s in sorted(signatures)])
            if data['origin'] == 'agent':
                success = data['outcome'] == 'success'
                if tier == 'weak' and learning.enabled('weak', 'reflection'):
                    events = {
                        'repeated_error': success and repeated,
                        'user_correction': data['user_correction'],
                        'long_procedure': success and data['steps'] >= LONG_STEPS,
                        'contradiction': bool(data['contradicts']),
                        'skill_failed': not success and bool(data['skills_used']),
                    }
                    for name, fired in events.items():
                        if fired:
                            _trigger(learning, run_id, name, 'weak', now)
                if (tier == 'frontier' and success and data['steps'] >= LONG_STEPS
                        and learning.enabled('weak', 'skills')):
                    _trigger(learning, run_id, 'distill', 'weak', now)
        else:
            tier = learning.db.execute('SELECT tier FROM runs WHERE run_id=?', (run_id,)).fetchone()['tier']
        triggers = [row['event'] for row in learning.db.execute(
            'SELECT event FROM triggers WHERE run_id=? ORDER BY rowid', (run_id,))]
    return {'run_id': run_id, 'tier': tier, 'triggers': triggers, 'dedup': not bool(inserted)}


def request_reflection(learning, run_id) -> dict:
    """Explicit reflection for either tier, even with automatic reflection off.

    Reviewer/eval-origin evidence cannot feed a reviewer loop, even on request.
    Repeated requests preserve the existing trigger's status and detail.
    """
    run_id = _run_id(run_id)
    with learning.db:
        run = learning.db.execute('SELECT tier, origin FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if run is None:
            raise ValueError('Unknown run_id')
        if run['origin'] != 'agent':
            raise ValueError('Reviewer and eval runs cannot trigger reflection')
        _trigger(learning, run_id, 'on_request', run['tier'], time.time())
        result = learning.db.execute(
            "SELECT * FROM triggers WHERE run_id=? AND event='on_request'", (run_id,)).fetchone()
    return dict(result)


def repeated_error_rate(learning, model=None):
    """Share of failed runs with at least one signature seen in an earlier run.

    A model filter selects the failed-run denominator; previous evidence can come
    from any model. Failed runs with no error text still count in the denominator.
    """
    row = learning.db.execute('''SELECT count(*) AS failed, coalesce(sum(EXISTS (
        SELECT 1 FROM signatures current
        JOIN signatures previous ON previous.signature=current.signature
        JOIN runs earlier ON earlier.run_id=previous.run_id
        WHERE current.run_id=r.run_id AND earlier.rowid < r.rowid AND earlier.origin='agent'
    )), 0) AS repeated FROM runs r
    WHERE r.outcome='failure' AND r.origin='agent' AND (? IS NULL OR r.agent_model=?)''',
                              (model, clean_text(model, 200) if model is not None else None)).fetchone()
    return row['repeated'] / row['failed'] if row['failed'] else 0.0
