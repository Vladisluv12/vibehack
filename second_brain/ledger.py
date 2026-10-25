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
        with open(self.path, 'a+b') as stream:
            # A crash can leave a torn intent without its newline. Terminate it first,
            # or the next record would fuse with it and be lost to the parser.
            size = os.fstat(stream.fileno()).st_size
            if size:
                stream.seek(size - 1)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')
            stream.write(line.encode('utf-8'))
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
                # Torn lines are intents cut before their fsync completed; the file
                # write only starts after a complete intent, so they changed nothing.
                continue
        return result

    @contextmanager
    def mutation(self, op, rel, actor, before, after, read_current, note_id='', reason='', target_op='', meta=None):
        """Callers hold the vault lock. Yields the op id; the body performs the write.

        If the body fails, the file decides the outcome: it may already hold the new
        text (commit) or still the old one (abort); anything else stays pending for
        recovery instead of being recorded as something it was not.
        """
        op_id = uuid.uuid4().hex
        self._blob(before)
        self._append({'op_id': op_id, 'phase': 'intent', 'ts': time.time(), 'op': op, 'path': rel,
                      'note_id': note_id, 'actor': actor, 'reason': reason[:300], **(meta or {}),
                      'sha256_before': sha(before), 'sha256_after': sha(after), 'target_op': target_op})
        try:
            yield op_id
        except BaseException:
            current = sha(read_current(rel))
            if current == sha(after):
                self._append({'op_id': op_id, 'phase': 'commit', 'ts': time.time(), 'after_error': True})
            elif current == sha(before):
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
        """Settle intents left by a crash: the file tells whether the write landed.

        Run under the vault lock before every mutation: a writer holds the lock for
        its whole write, so an open intent seen here belongs to a dead process.
        """
        settled = []
        for entry in self.pending():
            current = sha(read_current(entry['path']))
            if current == entry['sha256_after']:
                phase = 'commit'
            elif current == entry['sha256_before']:
                phase = 'abort'
            else:
                phase, entry = 'abort', {**entry, 'unknown': True}  # changed by someone else since
            self._append({'op_id': entry['op_id'], 'phase': phase, 'ts': time.time(), 'recovered': True,
                          'unknown': entry.get('unknown', False)})
            settled.append((entry['op_id'], phase))
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
