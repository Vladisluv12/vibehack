"""Durable learning state beside the vault: model tiers, retrieval usage, agent runs,
reflection triggers, mechanism switches and evaluation results.

Kept out of brain.sqlite on purpose: sync rebuilds that projection from Markdown,
while this history has no Markdown source. No provider calls here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from .redact import redact

SCHEMA_VERSION = 3
TIERS = ('weak', 'frontier')
DEFAULT_TIER = 'frontier'  # unknown models get no automatic self-improvement
MECHANISMS = ('skills', 'reflection')
MAX_TEXT = 4000
MAX_LIST = 50

MIGRATIONS = {
    1: [
        'CREATE TABLE served (run_id TEXT, note_id TEXT, tool TEXT, agent_model TEXT, ts REAL)',
        'CREATE INDEX served_note ON served(note_id)',
        'CREATE TABLE usage (run_id TEXT, note_id TEXT, helped INTEGER, outcome TEXT, ts REAL, PRIMARY KEY(run_id, note_id))',
        'CREATE TABLE runs (run_id TEXT PRIMARY KEY, agent_model TEXT, tier TEXT, origin TEXT, outcome TEXT, '
        'steps INTEGER, data TEXT, ts REAL)',
        'CREATE TABLE signatures (signature TEXT, run_id TEXT, outcome TEXT, ts REAL, PRIMARY KEY(signature, run_id))',
        'CREATE TABLE triggers (run_id TEXT, event TEXT, target_tier TEXT, status TEXT, detail TEXT, ts REAL, '
        'PRIMARY KEY(run_id, event))',
        'CREATE TABLE mechanisms (tier TEXT, name TEXT, enabled INTEGER, reason TEXT, actor TEXT, ts REAL, '
        'PRIMARY KEY(tier, name))',
        'CREATE TABLE rollout (skill_id TEXT PRIMARY KEY, stage TEXT, ts REAL)',
        'CREATE TABLE eval_results (batch TEXT, task_id TEXT, tier TEXT, config TEXT, agent_model TEXT, '
        'success INTEGER, tokens INTEGER, seconds REAL, signature TEXT, ts REAL)',
    ],
    # Batches are comparable only on identical task definitions and inputs.
    2: ['ALTER TABLE eval_results ADD COLUMN task_hash TEXT'],
    # Usage reports bind to the gateway process that served the notes, not just the model.
    3: ['ALTER TABLE served ADD COLUMN session TEXT'],
}
RUN_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,100}$')


def load_agents(state: Path) -> dict:
    """`state/agents.json`: {"models": {"<model id>": "weak"|"frontier"}, "default": "frontier"}.

    Written by the host who launches agents; nothing an agent calls can change it.
    """
    path = Path(state) / 'agents.json'
    if not path.exists():
        return {'models': {}, 'default': DEFAULT_TIER}
    data = json.loads(path.read_text(encoding='utf-8'))
    models = data.get('models', {}) if isinstance(data, dict) else None
    default = data.get('default', DEFAULT_TIER) if isinstance(data, dict) else None
    if not isinstance(models, dict) or any(not isinstance(k, str) or v not in TIERS for k, v in models.items()):
        raise ValueError('agents.json: models must map model ids to "weak" or "frontier"')
    if default != DEFAULT_TIER:
        raise ValueError('agents.json: unknown models are always "frontier"; list weak models explicitly')
    return {'models': dict(models), 'default': default}


def tier_of(agents: dict, model: str | None) -> str:
    return agents['models'].get(model or '', agents['default'])


def error_signature(text: str) -> str:
    """Same failure → same hash, regardless of paths, numbers, addresses and quoting."""
    text = text.strip().splitlines()
    text = '\n'.join(line.strip() for line in text if line.strip())[-2000:]
    text = re.sub(r'0x[0-9a-fA-F]+', '<addr>', text)
    text = re.sub(r'(?:[A-Za-z]:)?[\\/][^\s:\'"]+', '<path>', text)
    text = re.sub(r'\b[0-9a-f]{7,64}\b', '<hex>', text)
    text = re.sub(r'\d+', '<n>', text)
    text = re.sub(r'[\'"`]', '', text).lower()
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def clean_text(value, limit=MAX_TEXT) -> str:
    return redact(value)[:limit] if isinstance(value, str) else ''


def clean_run_id(value) -> str:
    """Run ids are opaque host/agent labels, never free text that could carry a secret."""
    if not isinstance(value, str) or not RUN_ID.fullmatch(value) or redact(value) != value:
        raise ValueError('run_id must be 1-100 characters of letters, digits and _.:-')
    return value


def clean_ids(value) -> list:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and re.fullmatch(r'[a-f0-9]{32}', v)][:MAX_LIST]


class Learning:
    def __init__(self, state: Path):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.state / 'learning.sqlite', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.migrate()
        self.agents = load_agents(self.state)

    def close(self):
        self.db.close()

    def migrate(self):
        """One serialized transaction per step: concurrent first starts and crashes
        mid-step cannot leave a half-built schema behind."""
        level, self.db.isolation_level = self.db.isolation_level, None
        try:
            while True:
                self.db.execute('BEGIN IMMEDIATE')
                try:
                    version = self.db.execute('PRAGMA user_version').fetchone()[0]
                    if version > SCHEMA_VERSION:
                        raise ValueError(f'learning.sqlite schema {version} is newer than this code ({SCHEMA_VERSION})')
                    if version == SCHEMA_VERSION:
                        self.db.execute('COMMIT')
                        return
                    for statement in MIGRATIONS[version + 1]:
                        self.db.execute(statement)
                    self.db.execute(f'PRAGMA user_version={version + 1}')
                    self.db.execute('COMMIT')
                except BaseException:
                    self.db.execute('ROLLBACK')
                    raise
        finally:
            self.db.isolation_level = level

    def tier(self, model):
        return tier_of(self.agents, model)

    # -- mechanisms ---------------------------------------------------------------
    def enabled(self, tier, name):
        """Mechanisms start off: only a measured gain (or a human) turns one on."""
        row = self.db.execute('SELECT enabled FROM mechanisms WHERE tier=? AND name=?', (tier, name)).fetchone()
        return bool(row and row['enabled'])

    def set_mechanism(self, tier, name, enabled, reason, actor):
        if tier not in TIERS or name not in MECHANISMS:
            raise ValueError('Unknown tier or mechanism')
        if name == 'reflection' and tier == 'frontier' and enabled:
            raise ValueError('Automatic reflection is reserved for weak models')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO mechanisms VALUES (?,?,?,?,?,?)',
                            (tier, name, int(bool(enabled)), clean_text(reason, 500), actor, time.time()))

    def mechanisms(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM mechanisms ORDER BY tier, name')]

    # -- retrieval usage ----------------------------------------------------------
    def served(self, run_id, note_ids, tool, model, session=''):
        if not run_id:
            return
        run_id, now = clean_run_id(run_id), time.time()
        with self.db:
            self.db.executemany('INSERT INTO served (run_id, note_id, tool, agent_model, ts, session) VALUES (?,?,?,?,?,?)',
                                [(run_id, n, tool, model or '', now, session) for n in clean_ids(list(note_ids))])

    def report_usage(self, run_id, used_ids, outcome, model='', session=''):
        """What the agent relied on, out of what this same gateway session served in this run.

        Notes it was not served are ignored, and the first report for a run stands:
        a later call, or another process using the same run id, cannot rewrite it.
        """
        run_id = clean_run_id(run_id)
        if outcome not in ('success', 'failure', 'unknown'):
            raise ValueError('outcome must be success, failure or unknown')
        used = set(clean_ids(used_ids))
        served = {r['note_id'] for r in self.db.execute(
            "SELECT DISTINCT note_id FROM served WHERE run_id=? AND agent_model=? AND coalesce(session,'')=?",
            (run_id, model or '', session))}
        if self.db.execute('SELECT 1 FROM usage WHERE run_id=? LIMIT 1', (run_id,)).fetchone():
            return {'run_id': run_id, 'recorded': False, 'reason': 'usage already reported for this run'}
        now = time.time()
        with self.db:
            for note in served:
                self.db.execute('INSERT OR IGNORE INTO usage VALUES (?,?,?,?,?)',
                                (run_id, note, int(note in used), outcome, now))
        return {'run_id': run_id, 'recorded': bool(served), 'served': len(served), 'used': len(used & served),
                'ignored_unserved': len(used - served)}

    def usefulness(self, note_id):
        served = self.db.execute('SELECT count(DISTINCT run_id) FROM served WHERE note_id=?', (note_id,)).fetchone()[0]
        helped = self.db.execute('SELECT count(*) FROM usage WHERE note_id=? AND helped=1', (note_id,)).fetchone()[0]
        return {'served_runs': served, 'helped_runs': helped}

    def stale_candidates(self, min_served=5):
        """Served often, never relied on: the measured version of 'stale'."""
        rows = self.db.execute('''SELECT s.note_id, count(DISTINCT s.run_id) AS served,
            (SELECT count(*) FROM usage u WHERE u.note_id=s.note_id AND u.helped=1) AS helped
            FROM served s GROUP BY s.note_id HAVING served >= ? AND helped = 0 ORDER BY served DESC''', (min_served,))
        return [dict(r) for r in rows]
