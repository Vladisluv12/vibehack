"""Shared log of agent access to memory, for the panel's live animation.

MCP gateways and the web server are separate processes with the same state
directory, so events go through `state/activity.sqlite`. Only note IDs, a
short redacted query and the actor name are stored; never note content.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import time

from .brain import ID
from .redact import redact

TYPES = {'memory.search', 'memory.read', 'memory.traverse', 'memory.draft', 'chat.answer'}
KEEP = 5000
RECENT = 20


def _open(state):
    db = sqlite3.connect(Path(state) / 'activity.sqlite', timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, type TEXT, ids TEXT, query TEXT)')
    return db


def record(state, actor, kind, ids=(), query=''):
    if kind not in TYPES:
        raise ValueError('Unknown activity type')
    actor = re.sub(r'[^\w.:@-]', '', str(actor))[:40] or 'agent'
    ids = [i for i in ids if isinstance(i, str) and ID.fullmatch(i)][:50]
    with closing(_open(state)) as db, db:
        db.execute('INSERT INTO events(ts,actor,type,ids,query) VALUES (?,?,?,?,?)',
                   (time.time(), actor, kind, json.dumps(ids), redact(str(query))[:200]))
        db.execute('DELETE FROM events WHERE seq <= (SELECT max(seq) FROM events) - ?', (KEEP,))


def safe_record(state, actor, kind, ids=(), query=''):
    """Logging must never break a tool call."""
    try:
        record(state, actor, kind, ids, query)
    except (OSError, sqlite3.Error, ValueError):
        pass


def since(state, after=None, limit=200):
    """New events after a cursor; without one, the latest few (not live) and the cursor."""
    with closing(_open(state)) as db:
        cursor = db.execute('SELECT coalesce(max(seq),0) FROM events').fetchone()[0]
        if after is None:
            rows = reversed(db.execute('SELECT * FROM events ORDER BY seq DESC LIMIT ?', (RECENT,)).fetchall())
        else:
            rows = db.execute('SELECT * FROM events WHERE seq > ? ORDER BY seq LIMIT ?', (int(after), min(500, int(limit)))).fetchall()
        items = [{'seq': r['seq'], 'ts': r['ts'], 'actor': r['actor'], 'type': r['type'], 'ids': json.loads(r['ids']),
                  'query': r['query'], 'live': after is not None} for r in rows]
    return {'cursor': cursor, 'items': items, 'initial': after is None}
