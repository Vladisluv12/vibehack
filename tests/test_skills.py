import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from second_brain.brain import Brain
from second_brain.learning import Learning
from second_brain.reflection import ingest_run
from second_brain.skills import canary_compare, promote, skill_pack


class SkillsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.brain = Brain(root / 'vault', root / 'state')
        self.addCleanup(self.brain.close)
        self.learning = Learning(root / 'state')
        self.addCleanup(self.learning.close)

    def enable(self, tier='weak'):
        self.learning.set_mechanism(tier, 'skills', True, 'fixture', 'human:test')

    def skill(self, title='Build skill', body='build safely', review='reviewed', source='', **extra):
        ident = self.brain.create(title, body, review=review, source=source,
                                  extra={'memory_type': 'skill', 'applies_to': 'build projects', **extra})
        self.brain.sync()
        return ident

    def pack(self, **extra):
        return skill_pack(self.brain, self.learning,
                          **(dict(tier='weak', task='build', run_id='run') | extra))

    def test_off_produces_no_pack_rollout_or_served_rows(self):
        self.skill()
        self.assertEqual(self.pack(), {'text': '', 'skill_ids': [], 'canary_ids': []})
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM rollout').fetchone()[0], 0)
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM served').fetchone()[0], 0)

    def test_review_kind_tier_scope_eval_exclusion_and_live_review(self):
        self.enable()
        allowed = self.skill('Build allowed', for_tier='weak')
        excluded = [self.skill('Build proposed', review='proposed'),
                    self.skill('Build unreviewed', review='unreviewed'),
                    self.skill('Build frontier', for_tier='frontier'),
                    self.skill('Build ordinary', memory_type='experience'),
                    self.skill('Build eval', source='eval:fixture'),
                    self.skill('Gardening', body='water flowers', applies_to='gardens')]
        for ident in [allowed, *excluded]:
            promote(self.learning, ident, 'full', 'human:test')
        self.assertEqual(self.pack(k=20)['skill_ids'], [allowed])
        # A human's Markdown change after the last sync must still be respected.
        note = self.brain.get(allowed)
        path = self.brain.vault / note['path']
        path.write_text(note['content'].replace('review: "reviewed"', 'review: "proposed"'), encoding='utf-8')
        self.assertEqual(self.pack()['skill_ids'], [])

    def test_k_and_budget_include_complete_index_and_reference(self):
        self.enable()
        ids = [self.skill(f'Build {i}', body='build details ' * 50) for i in range(5)]
        for ident in ids:
            promote(self.learning, ident, 'full', 'eval:test')
        pack = self.pack(k=2)
        self.assertEqual(len(pack['skill_ids']), 2)
        self.assertLessEqual(len(pack['text']), 4000)
        self.assertEqual(pack['canary_ids'], [])
        self.assertEqual(sum(line.startswith('- ') for line in pack['text'].splitlines()), 2)
        self.assertIn('(applies_to: build projects)', pack['text'])
        for ident in pack['skill_ids']:
            self.assertEqual(pack['text'].count(ident), 2)
        clipped = self.pack(k=3, budget_chars=180)
        self.assertEqual(len(clipped['text']), 180)
        self.assertEqual(len(clipped['skill_ids']), 1)
        self.assertIn(f"[{clipped['skill_ids'][0]}]\nbuild", clipped['text'])
        for options in ({'k': 0}, {'budget_chars': 0}, {'budget_chars': 10}, {'task': ''}):
            self.assertEqual(self.pack(**options), {'text': '', 'skill_ids': [], 'canary_ids': []})

    def test_canary_exact_hash_rule_deterministic_and_default(self):
        self.enable()
        ident = self.skill()
        seen, absent = [], []
        for i in range(100):
            run_id = f'run-{i}'
            expected = int(hashlib.sha256((run_id + ident).encode()).hexdigest(), 16) % 100 < 20
            result = self.pack(run_id=run_id)
            self.assertEqual(result['skill_ids'], [ident] if expected else [])
            self.assertEqual(result['canary_ids'], result['skill_ids'])
            (seen if expected else absent).append(run_id)
            self.assertEqual(self.pack(run_id=run_id), result)
        self.assertTrue(seen)
        self.assertTrue(absent)
        self.assertEqual(self.learning.db.execute('SELECT stage FROM rollout').fetchone()['stage'], 'canary')
        promote(self.learning, ident, 'full', 'human:test')
        self.assertEqual(self.pack(run_id=absent[0])['skill_ids'], [ident])
        promote(self.learning, ident, 'off', 'eval:test')
        self.assertEqual(self.pack(run_id=seen[0])['skill_ids'], [])

    def test_served_only_contains_rendered_notes_and_known_model(self):
        self.enable()
        ident = self.skill()
        promote(self.learning, ident, 'full', 'human:test')
        ingest_run(self.learning, {'run_id': 'run', 'agent_model': 'small', 'task': 'build', 'outcome': 'success'}, 'small')
        self.pack(budget_chars=10)
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM served').fetchone()[0], 0)
        self.pack()
        served = [dict(r) for r in self.learning.db.execute('SELECT * FROM served')]
        self.assertEqual(len(served), 1)
        self.assertEqual((served[0]['run_id'], served[0]['note_id'], served[0]['tool'], served[0]['agent_model']),
                         ('run', ident, 'skill_pack', 'small'))

    def test_pagination_does_not_hide_skill_after_non_skills(self):
        self.enable()
        for i in range(101):
            self.brain.create(f'Build {i:03}', 'build', review='reviewed')
        ident = self.skill('Build zzz', body='build')
        promote(self.learning, ident, 'full', 'human:test')
        self.assertEqual(self.pack()['skill_ids'], [ident])

    def test_redaction_and_frontier_flag(self):
        self.enable('frontier')
        secret = 'ghp_abcdefghijklmnopqrstuvwxyz'
        ident = self.skill(body='build password=private-value ' + secret)
        promote(self.learning, ident, 'full', 'human:test')
        with self.assertRaises(ValueError):  # a run id is a label, never text that may carry a secret
            self.pack(tier='frontier', run_id='password=private-value')
        pack = self.pack(tier='frontier', run_id='run-1')
        self.assertEqual(pack['skill_ids'], [ident])
        self.assertNotIn(secret, pack['text'])
        self.assertNotIn('private-value', pack['text'])
        self.assertEqual(self.learning.db.execute('SELECT run_id FROM served').fetchone()[0], 'run-1')

    def test_promotion_is_audited_and_never_changes_a_note_or_mechanism(self):
        ident = self.skill(review='proposed')
        before = self.brain.get(ident)['hash']
        for stage, actor in [('canary', 'human:token=private-value'), ('full', 'eval:batch'), ('off', 'human')]:
            result = promote(self.learning, ident, stage, actor)
            self.assertEqual(result['stage'], stage)
        records = [json.loads(line) for line in
                   (self.learning.state / 'rollout-decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([r['stage'] for r in records], ['canary', 'full', 'off'])
        self.assertNotIn('private-value', records[0]['actor'])
        self.assertEqual(self.brain.get(ident)['hash'], before)
        self.assertFalse(self.learning.enabled('weak', 'skills'))
        for stage, actor in [('reviewed', 'human'), ('full', 'agent:reviewer'), ('full', 'human:'), ('full', '')]:
            with self.subTest(stage=stage, actor=actor), self.assertRaises(ValueError):
                promote(self.learning, ident, stage, actor)
        self.assertEqual(self.learning.db.execute('SELECT stage FROM rollout').fetchone()[0], 'off')

    def test_failed_audit_write_rolls_back_promotion(self):
        ident = self.skill()
        promote(self.learning, ident, 'canary', 'human')
        with mock.patch('second_brain.skills.os.open', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                promote(self.learning, ident, 'full', 'human')
        self.assertEqual(self.learning.db.execute('SELECT stage FROM rollout').fetchone()[0], 'canary')

    def test_comparison_matches_tasks_counts_failures_and_never_promotes(self):
        ident = self.skill()
        rows = [('build', True, 'success'), ('build', True, 'success'), ('build', True, 'failure'),
                ('build', False, 'success'), ('build', False, 'failure'),
                ('garden', False, 'failure'), ('garden', False, 'failure')]
        for i, (task, used, outcome) in enumerate(rows):
            ingest_run(self.learning, {'run_id': str(i), 'task': task, 'outcome': outcome,
                                      'skills_used': [ident] if used else []}, 'small')
        result = canary_compare(self.learning, ident)
        self.assertEqual(result['tasks'], ['build'])
        self.assertEqual(result['with_skill'], {'count': 3, 'successes': 2, 'success_rate': 2 / 3})
        self.assertEqual(result['without_skill'], {'count': 2, 'successes': 1, 'success_rate': 0.5})
        self.assertEqual(self.learning.db.execute('SELECT count(*) FROM rollout').fetchone()[0], 0)
        missing = canary_compare(self.learning, 'f' * 32)
        self.assertEqual(missing['with_skill'], {'count': 0, 'successes': 0, 'success_rate': None})
        self.assertEqual(missing['without_skill'], missing['with_skill'])


if __name__ == '__main__':
    unittest.main()
