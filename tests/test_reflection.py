import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from second_brain.learning import Learning, MAX_LIST, MAX_TEXT, error_signature
from second_brain.reflection import LONG_STEPS, ingest_run, repeated_error_rate, request_reflection


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        state = Path(self.temp.name)
        (state / 'agents.json').write_text(json.dumps({'models': {'small': 'weak', 'large': 'frontier'}}))
        self.learning = Learning(state)
        self.addCleanup(self.learning.close)

    def enable(self, mechanism='reflection'):
        self.learning.set_mechanism('weak', mechanism, True, 'fixture', 'human:test')

    def run_event(self, run_id='r1', **extra):
        return ingest_run(self.learning, dict(run_id=run_id, agent_model='small',
                                             task='build project', outcome='success') | extra)

    def test_individual_triggers_and_threshold(self):
        self.enable()
        ident = 'a' * 32
        cases = [({'user_correction': True}, ['user_correction']),
                 ({'steps': LONG_STEPS}, ['long_procedure']),
                 ({'steps': LONG_STEPS - 1}, []),
                 ({'steps': LONG_STEPS, 'outcome': 'failure'}, []),
                 ({'contradicts': [ident]}, ['contradiction']),
                 ({'skills_used': [ident], 'outcome': 'failure'}, ['skill_failed']),
                 ({'skills_used': [ident]}, [])]
        for i, (event, expected) in enumerate(cases):
            with self.subTest(event=event):
                self.assertEqual(self.run_event(str(i), **event)['triggers'], expected)
        self.assertEqual({r['status'] for r in self.learning.db.execute('SELECT * FROM triggers')}, {'pending'})

    def test_repeated_error_requires_a_distinct_earlier_run_and_success(self):
        self.enable()
        error = 'Failure at /tmp/build/a.py line 27 address 0xabba'
        changed = 'Failure at /home/test/b.py line 81 address 0x1234'
        self.assertEqual(error_signature(error), error_signature(changed))
        # Equal timestamps must not turn errors from one run into earlier evidence.
        with mock.patch('second_brain.reflection.time.time', return_value=123):
            self.assertEqual(self.run_event('first', errors=[error, changed])['triggers'], [])
            self.assertEqual(self.run_event('failed', errors=[changed], outcome='failure')['triggers'], [])
            self.assertEqual(self.run_event('fixed', errors=[changed])['triggers'], ['repeated_error'])
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM signatures').fetchone()[0], 3)

    def test_off_and_frontier_distillation_are_separate(self):
        event = dict(steps=LONG_STEPS, user_correction=True, contradicts=['a' * 32])
        self.assertEqual(self.run_event('off', **event)['triggers'], [])
        self.enable()
        for model in ('large', 'unknown'):
            result = self.run_event(model, agent_model=model, tier='weak', **event)
            self.assertEqual((result['tier'], result['triggers']), ('frontier', []))
        self.enable('skills')
        result = self.run_event('teacher', agent_model='large', **event)
        self.assertEqual(result['triggers'], ['distill'])
        trigger = self.learning.db.execute("SELECT * FROM triggers WHERE event='distill'").fetchone()
        self.assertEqual(trigger['target_tier'], 'weak')
        self.assertEqual(self.run_event('bad-teacher', agent_model='large', outcome='failure',
                                       steps=LONG_STEPS)['triggers'], [])

    def test_reviewer_and_eval_runs_never_loop(self):
        self.enable()
        self.enable('skills')
        for origin in ('reviewer', 'eval'):
            for model in ('small', 'large'):
                run_id = origin + model
                result = self.run_event(run_id, origin=origin, agent_model=model, steps=LONG_STEPS,
                                        user_correction=True, contradicts=['b' * 32])
                self.assertEqual(result['triggers'], [])
                with self.assertRaises(ValueError):
                    request_reflection(self.learning, run_id)

    def test_replays_do_not_rewrite_evidence_or_reopen_triggers(self):
        self.enable()
        self.run_event(user_correction=True)
        with self.learning.db:
            self.learning.db.execute("UPDATE triggers SET status='done', detail='kept'")
        again = self.run_event(agent_model='large', errors=['different'], outcome='failure')
        self.assertEqual(again, {'run_id': 'r1', 'tier': 'weak', 'triggers': ['user_correction'], 'dedup': True})
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM runs').fetchone()[0], 1)
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM signatures').fetchone()[0], 0)
        row = self.learning.db.execute('SELECT * FROM triggers').fetchone()
        self.assertEqual((row['status'], row['detail']), ('done', 'kept'))

    def test_explicit_request_allows_either_tier_with_switches_off(self):
        for model in ('small', 'large'):
            self.run_event(model, agent_model=model)
            first = request_reflection(self.learning, model)
            self.assertEqual(first['target_tier'], self.learning.tier(model))
            self.assertEqual(first['event'], 'on_request')
            self.assertEqual(request_reflection(self.learning, model), first)
        with self.assertRaises(ValueError):
            request_reflection(self.learning, 'missing')

    def test_whitelist_redaction_and_bounds(self):
        secret = 'ghp_abcdefghijklmnopqrstuvwxyz'
        self.run_event(task=secret + 'x' * 5000, summary='password=private-value',
                       errors=[secret] * (MAX_LIST + 1), skills_used=['a' * 32, 'bad', 'a' * 32],
                       contradicts=['b' * 32, 123], transcript='raw transcript', reasoning='private thoughts',
                       tier='frontier', extra={'review': 'reviewed'})
        row = self.learning.db.execute('SELECT * FROM runs').fetchone()
        data = json.loads(row['data'])
        self.assertEqual(set(data), {'run_id', 'agent_model', 'origin', 'task', 'outcome', 'steps',
                                     'user_correction', 'errors', 'skills_used', 'contradicts', 'summary'})
        self.assertNotIn(secret, row['data'])
        self.assertNotIn('private-value', row['data'])
        self.assertLessEqual(len(data['task']), MAX_TEXT)
        self.assertEqual(len(data['errors']), MAX_LIST)
        self.assertEqual(data['skills_used'], ['a' * 32])
        self.assertEqual(data['contradicts'], ['b' * 32])
        self.assertEqual(row['tier'], 'weak')

    def test_malformed_control_fields_fail_without_partial_rows(self):
        for patch in ({'run_id': ''}, {'run_id': 'x' * 101}, {'steps': -1}, {'steps': True},
                      {'steps': 2**80}, {'outcome': 'unknown'}, {'origin': 'tool'},
                      {'user_correction': 'false'}, {'agent_model': []}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.run_event(**patch)
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM runs').fetchone()[0], 0)

    def test_repeated_error_rate_counts_runs_not_error_occurrences(self):
        self.assertEqual(repeated_error_rate(self.learning), 0.0)
        with mock.patch('second_brain.reflection.time.time', return_value=42):
            self.run_event('first', outcome='failure', errors=['bad 1', 'bad 2'])
            self.run_event('second', outcome='failure', errors=['bad 3'])
            self.run_event('empty', outcome='failure')
            self.run_event('foreign', outcome='failure', agent_model='large', errors=['bad 4'])
            self.run_event('fixed', errors=['bad 5'])
        self.assertEqual(repeated_error_rate(self.learning), 0.5)
        self.assertEqual(repeated_error_rate(self.learning, 'small'), 1 / 3)
        self.assertEqual(repeated_error_rate(self.learning, 'large'), 1.0)
        self.assertEqual(repeated_error_rate(self.learning, 'missing'), 0.0)


if __name__ == '__main__':
    unittest.main()
