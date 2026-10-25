"""Append-only mutation ledger: the gate every note write passes.

An intent line (fsynced) precedes the file change, a commit or abort line follows.
If the intent cannot be written, the change does not happen. On start, intents
without an outcome are settled by comparing the file with the recorded hashes.
Before-images are kept as content-addressed blobs, so any batch can be rolled back.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

TERMINAL = {'commit', 'abort'}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else None


class Ledger:
    def __init__(self, state: Path):
        self.path = Path(state) / 'ledger.jsonl'
        self.blobs = Path(state) / 'ledger-blobs'
        self.blobs.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _append(self, entry):
        line = json.dumps(entry, ensure_ascii=False) + '\n'
        with open(self.path, 'a', encoding='utf-8') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def _blob(self, text):
        if text is None:
            return None
        digest = sha(text)
        target = self.blobs / f'{digest}.md'
        if not target.exists():
            tmp = target.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8', newline='') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        return digest

    def entries(self):
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                continue  # a torn last line from a crash carries no commitment
        return result

    @contextmanager
    def mutation(self, op, rel, actor, before, after, note_id='', reason='', target_op=''):
        """Callers hold the vault lock. Yields the op id; the body performs the write."""
        op_id = uuid.uuid4().hex
        self._blob(before)
        self._append({'op_id': op_id, 'phase': 'intent', 'ts': time.time(), 'op': op, 'path': rel,
                      'note_id': note_id, 'actor': actor, 'reason': reason[:300],
                      'sha256_before': sha(before), 'sha256_after': sha(after), 'target_op': target_op})
        try:
            yield op_id
        except BaseException:
            self._append({'op_id': op_id, 'phase': 'abort', 'ts': time.time()})
            raise
        self._append({'op_id': op_id, 'phase': 'commit', 'ts': time.time()})

    def pending(self):
        open_ops = {}
        for entry in self.entries():
            if entry.get('phase') == 'intent':
                open_ops[entry['op_id']] = entry
            elif entry.get('phase') in TERMINAL:
                open_ops.pop(entry.get('op_id'), None)
        return list(open_ops.values())

    def recover(self, read_current):
        """Settle intents left by a crash: the file tells whether the write landed."""
        settled = []
        for entry in self.pending():
            current = read_current(entry['path'])
            landed = sha(current) == entry['sha256_after']
            self._append({'op_id': entry['op_id'], 'phase': 'commit' if landed else 'abort',
                          'ts': time.time(), 'recovered': True})
            settled.append((entry['op_id'], landed))
        return settled

    def committed(self):
        """Committed changes not yet undone, oldest first."""
        intents, done = {}, set()
        for entry in self.entries():
            if entry.get('phase') == 'intent':
                intents[entry['op_id']] = entry
            elif entry.get('phase') == 'commit':
                done.add(entry['op_id'])
        undone = {e['target_op'] for k, e in intents.items() if k in done and e.get('op') == 'rollback'}
        return [e for k, e in intents.items() if k in done and k not in undone and e.get('op') != 'rollback']

    def before_text(self, entry):
        digest = entry.get('sha256_before')
        return None if digest is None else (self.blobs / f'{digest}.md').read_text(encoding='utf-8')
