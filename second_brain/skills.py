"""Reviewed skill packs and observational rollout metrics; no note mutations.

The host alone calls promote for human/eval rollout decisions. Promotion changes
delivery, never a note's review status. Metrics make no automatic rollout decision.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

from .learning import TIERS, clean_ids, clean_text

MAX_SCAN = 300  # search hits inspected per lookup
from .reflection import _run_id


def _skill_id(value):
    if not clean_ids([value]):
        raise ValueError('skill_id must be a canonical note ID')
    return value


def _render(parts):
    return '\n'.join(line for _, line, _ in parts) + '\n\n' + '\n\n'.join(body for _, _, body in parts)


def skill_pack(brain, learning, *, tier, task, run_id, k=3, budget_chars=4000, include_reflected=True) -> dict:
    """Return at most k reviewed, task-matching skills within a character budget.

    Canary membership is sha256(cleaned run_id + skill_id) modulo 100 < 20.
    include_reflected=False leaves out skills the reviewer wrote, so evaluation can
    tell the value of reflection (config C) from that of skills in general (B).
    Index lines and reference headings remain whole; only bodies may be cut. IDs
    are reported/served only when their index and some body fit in the pack.
    """
    empty = {'text': '', 'skill_ids': [], 'canary_ids': []}
    if tier not in TIERS:
        raise ValueError('Unknown tier')
    if not learning.enabled(tier, 'skills'):
        return empty
    if type(k) is not int or k < 0 or type(budget_chars) is not int or budget_chars < 0:
        raise ValueError('k and budget_chars must be non-negative integers')
    if not k or not budget_chars:
        return empty
    run_id = _run_id(run_id)
    query = ' '.join(re.findall(r'\w+', clean_text(task), re.UNICODE)[:12])
    if not query:
        return empty
    parts, canaries, offset = [], [], 0
    while len(parts) < k:
        page = brain.list(query, offset=offset, limit=100, any_terms=True)
        for item in page['items']:
            note = brain.get(item['id'])
            meta = note['meta']
            if (note['review'] != 'reviewed' or meta.get('memory_type') != 'skill'
                    or ('for_tier' in meta and meta['for_tier'] != tier)
                    or str(meta.get('source', '')).startswith('eval:')
                    or (not include_reflected and meta.get('agent_model') == 'reviewer')):
                continue
            ident = note['id']
            with learning.db:
                learning.db.execute('''INSERT OR IGNORE INTO rollout (skill_id, stage, ts)
                    VALUES (?,'canary',?)''', (ident, time.time()))
                stage = learning.db.execute('SELECT stage FROM rollout WHERE skill_id=?', (ident,)).fetchone()['stage']
            bucket = int(hashlib.sha256((run_id + ident).encode()).hexdigest(), 16) % 100
            if stage != 'full' and not (stage == 'canary' and bucket < 20):
                continue
            title = ' '.join(clean_text(note['title'], 200).split())
            applies_to = ' '.join(clean_text(meta.get('applies_to'), 500).split())
            line = f'- {title} (applies_to: {applies_to}) [{ident}]'
            heading = f'[{ident}]\n'
            available = budget_chars - len(_render(parts + [(ident, line, heading)]))
            body = clean_text(note['body']).strip()
            if available <= 0 or not body:
                continue
            parts.append((ident, line, heading + body[:available]))
            if stage == 'canary':
                canaries.append(ident)
            if len(parts) == k:
                break
        if page['next'] is None or page['next'] >= MAX_SCAN:
            break
        offset = page['next']
    if not parts:
        return empty
    ids = [ident for ident, _, _ in parts]
    run = learning.db.execute('SELECT agent_model FROM runs WHERE run_id=?', (run_id,)).fetchone()
    # Packs normally precede ingestion, so a model is not yet available there.
    learning.served(run_id, ids, 'skill_pack', clean_text(run['agent_model'], 200) if run else '')
    return {'text': _render(parts), 'skill_ids': ids, 'canary_ids': canaries}


def promote(learning, skill_id, stage, actor) -> dict:
    """Record a host human/eval decision to deliver a skill as canary/full/off.

    The existing rollout table has no actor column. Keep redacted decision
    provenance in state/rollout-decisions.jsonl without an unversioned migration.
    The file records requested decisions; the table is the committed current stage.
    An audit-write failure rolls back the stage change. Neither store changes notes.
    """
    skill_id = _skill_id(skill_id)
    if stage not in ('canary', 'full', 'off'):
        raise ValueError('stage must be canary, full or off')
    if not isinstance(actor, str) or not (actor in ('human', 'eval')
            or (actor.startswith(('human:', 'eval:')) and actor.split(':', 1)[1].strip())):
        raise ValueError('Rollout decisions require a human or eval actor supplied by the host')
    decision = {'skill_id': skill_id, 'stage': stage, 'actor': clean_text(actor, 200), 'ts': time.time()}
    with learning.db:
        learning.db.execute('''INSERT INTO rollout (skill_id, stage, ts) VALUES (?,?,?)
            ON CONFLICT(skill_id) DO UPDATE SET stage=excluded.stage, ts=excluded.ts''',
                            (skill_id, stage, decision['ts']))
        fd = os.open(learning.state / 'rollout-decisions.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(decision, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return decision


def canary_compare(learning, skill_id) -> dict:
    """Observed success with this skill vs without it on the exact same task text.

    All ingested origins are counted. This is not a randomized causal estimate or
    a promotion recommendation. No observations produce a null rate, not zero.
    """
    skill_id = _skill_id(skill_id)
    runs = []
    for row in learning.db.execute('SELECT outcome, data FROM runs'):
        try:
            data = json.loads(row['data'])
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or row['outcome'] not in ('success', 'failure'):
            continue
        task = clean_text(data.get('task'))
        if task.strip():
            runs.append((task, skill_id in clean_ids(data.get('skills_used')), row['outcome'] == 'success'))
    tasks = {task for task, used, _ in runs if used}

    def stats(used):
        outcomes = [success for task, has_skill, success in runs if task in tasks and has_skill == used]
        return {'count': len(outcomes), 'successes': sum(outcomes),
                'success_rate': sum(outcomes) / len(outcomes) if outcomes else None}

    return {'skill_id': skill_id, 'tasks': sorted(tasks),
            'with_skill': stats(True), 'without_skill': stats(False)}
