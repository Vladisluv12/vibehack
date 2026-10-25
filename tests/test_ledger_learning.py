import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from second_brain.brain import Brain
from second_brain.learning import Learning, error_signature, load_agents


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.brain = Brain(self.root/'vault', self.root/'state', actor='agent:weak-1')

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_every_write_is_journaled_with_hashes(self):
        ident = self.brain.create('Заметка', 'текст')
        self.brain.sync()
        note = self.brain.get(ident)
        self.brain.edit(ident, note['content'] + '\nправка', note['hash'])
        entries = self.brain.ledger.entries()
        intents = [e for e in entries if e['phase'] == 'intent']
        self.assertEqual([e['op'] for e in intents], ['create', 'edit'])
        self.assertEqual(sum(e['phase'] == 'commit' for e in entries), 2)
        self.assertIsNone(intents[0]['sha256_before'])
        self.assertEqual(intents[1]['sha256_before'], note['hash'])
        self.assertEqual(intents[1]['actor'], 'agent:weak-1')

    def test_no_ledger_no_write(self):
        with mock.patch.object(self.brain.ledger, '_append', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.brain.create('Заметка', 'текст')
        self.assertEqual(list((self.root/'vault').rglob('Notes/*.md')), [])

    def test_crash_between_intent_and_commit_is_settled_on_open(self):
        real = self.brain.ledger._append
        calls = []

        def crash_on_commit(entry):
            calls.append(entry['phase'])
            if entry['phase'] == 'commit':
                raise KeyboardInterrupt  # process dies after the file landed
            real(entry)
        with mock.patch.object(self.brain.ledger, '_append', side_effect=crash_on_commit):
            with self.assertRaises(KeyboardInterrupt):
                self.brain.create('Заметка', 'текст')
        self.assertEqual(len(self.brain.ledger.pending()), 1)
        self.brain.close()
        self.brain = Brain(self.root/'vault', self.root/'state')
        self.assertEqual(self.brain.ledger.pending(), [])
        recovered = [e for e in self.brain.ledger.entries() if e.get('recovered')]
        self.assertEqual(recovered[0]['phase'], 'commit')

    def test_rollback_by_actor_restores_and_parks_created_notes(self):
        human = Brain(self.root/'vault', self.root/'state', actor='human:local')
        keep = human.create('Человек', 'исходный текст'); human.sync()
        original = human.get(keep)
        agent_note = self.brain.create('Агент', 'черновик'); self.brain.sync()
        self.brain.edit(keep, original['content'] + '\nдописал агент', original['hash'])
        result = human.rollback(actor='agent:weak-1')
        self.assertEqual(len(result['undone']), 2)
        self.assertEqual(result['conflicts'], [])
        human.sync()
        self.assertEqual(human.get(keep)['hash'], original['hash'])
        with self.assertRaises(ValueError):
            human.get(agent_note)
        self.assertEqual(len(list((self.root/'vault'/'Versions'/agent_note).glob('rolled-back-*.md'))), 1)
        self.assertEqual(human.rollback(actor='agent:weak-1')['undone'], [])
        human.close()

    def test_rollback_reports_conflict_instead_of_overwriting(self):
        ident = self.brain.create('Агент', 'черновик'); self.brain.sync()
        path = self.root/'vault'/self.brain.get(ident)['path']
        path.write_text(path.read_text(encoding='utf-8') + '\nправка человека в Obsidian', encoding='utf-8')
        result = self.brain.rollback(actor='agent:weak-1')
        self.assertEqual(result['undone'], [])
        self.assertEqual(result['conflicts'][0]['error'], 'changed since')
        self.assertTrue(path.exists())


class TierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_host_assigns_tiers_and_unknown_is_frontier(self):
        (self.state/'agents.json').write_text(json.dumps({'models': {'qwen3.8-flash': 'weak'}}))
        learning = Learning(self.state)
        self.assertEqual(learning.tier('qwen3.8-flash'), 'weak')
        self.assertEqual(learning.tier('claude-opus-5-5'), 'frontier')
        self.assertEqual(learning.tier(None), 'frontier')
        learning.close()

    def test_bad_agents_file_is_rejected(self):
        (self.state/'agents.json').write_text(json.dumps({'models': {'x': 'genius'}}))
        with self.assertRaises(ValueError):
            load_agents(self.state)

    def test_mechanisms_start_off_and_frontier_cannot_auto_reflect(self):
        learning = Learning(self.state)
        self.assertFalse(learning.enabled('weak', 'skills'))
        learning.set_mechanism('weak', 'reflection', True, 'test', 'human:local')
        self.assertTrue(learning.enabled('weak', 'reflection'))
        with self.assertRaises(ValueError):
            learning.set_mechanism('frontier', 'reflection', True, 'test', 'human:local')
        learning.close()

    def test_schema_is_versioned(self):
        learning = Learning(self.state)
        self.assertEqual(learning.db.execute('PRAGMA user_version').fetchone()[0], 1)
        learning.close()
        Learning(self.state).close()  # reopening does not re-run migrations

    def test_error_signature_ignores_paths_numbers_and_hashes(self):
        a = error_signature('File "/home/a/x.py", line 12\nKeyError: \'user_id\' at 0x7f00ab')
        b = error_signature('File "/tmp/b/y.py", line 99\nKeyError: "user_id" at 0x55ff01')
        c = error_signature('File "/tmp/b/y.py", line 99\nTypeError: bad operand')
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == '__main__':
    unittest.main()
