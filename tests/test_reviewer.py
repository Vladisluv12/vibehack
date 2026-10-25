import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from second_brain.brain import Brain
from second_brain.learning import Learning
from second_brain.reflection import ingest_run, request_reflection
from second_brain.reviewer import review_pending


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.brain = Brain(root / 'vault', root / 'state', actor='human:test')
        self.addCleanup(self.brain.close)
        self.source = self.brain.create('Build source', 'build', kind='source', review='reviewed')
        self.skill = self.brain.create('Build skill', 'build safely', review='reviewed',
                                       extra={'memory_type': 'skill', 'applies_to': 'build projects'})
        self.brain.sync()
        self.brain.actor = 'agent:reviewer'
        (root / 'state' / 'agents.json').write_text(json.dumps({'models': {'small': 'weak'}}))
        self.learning = Learning(root / 'state')
        self.addCleanup(self.learning.close)
        self.learning.set_mechanism('weak', 'reflection', True, 'fixture', 'human:test')
        self.prompts = []

    def run_event(self, run_id='r1', **extra):
        return ingest_run(self.learning, dict(run_id=run_id, agent_model='small', outcome='success',
                                             task='build', user_correction=True,
                                             skills_used=[self.source]) | extra)

    def decision(self, **extra):
        return json.dumps(dict(action='add', title='Build procedure', body='Run the checked build steps.',
                               applies_to='build projects', evidence='reported successful build', supersedes=None) | extra)

    def llm(self, response):
        def call(prompt):
            self.prompts.append(prompt)
            return 'A reported build error was corrected.' if 'Stage 1:' in prompt else response
        return call

    def new_notes(self):
        return [self.brain.get(n['id']) for n in self.brain.list()['items']
                if n['id'] not in (self.source, self.skill)]

    def status(self, run_id='r1'):
        return dict(self.learning.db.execute('SELECT * FROM triggers WHERE run_id=?', (run_id,)).fetchone())

    def test_add_is_proposed_with_sources_provenance_and_reviewer_actor(self):
        self.run_event(skills_used=[self.source, self.source, 'f' * 32], contradicts=[self.skill])
        # Two events exist; limit verifies only the oldest is handled.
        with mock.patch.object(self.brain, 'edit', side_effect=AssertionError('No editing')):
            results = review_pending(self.brain, self.learning, self.llm(self.decision()), limit=1)
        self.assertEqual(results[0]['status'], 'done')
        notes = self.new_notes()
        self.assertEqual(len(notes), 1)
        note = notes[0]
        self.assertEqual(note['review'], 'proposed')
        meta = note['meta']
        self.assertEqual(meta['memory_type'], 'skill')
        self.assertEqual(meta['agent_model'], 'reviewer')
        self.assertEqual(meta['agent_tier'], 'weak')
        self.assertEqual(meta['for_tier'], 'weak')
        self.assertEqual(meta['run_ids'], ['r1'])
        self.assertEqual(meta['trigger'], 'user_correction')
        self.assertEqual(set(meta['declared_links']), {self.source, self.skill})
        intents = [e for e in self.brain.ledger.entries() if e.get('note_id') == note['id']]
        self.assertEqual([(e['op'], e['actor']) for e in intents], [('create', 'agent:reviewer')])
        self.assertIn(self.skill, self.prompts[1])
        self.assertIn('applies_to', self.prompts[1])
        self.assertIn('BEGIN UNTRUSTED_', self.prompts[0])
        self.assertIn('never instructions', self.prompts[1])

    def test_update_creates_new_note_and_preserves_old_hash(self):
        self.run_event()
        before = self.brain.get(self.skill)['hash']
        with mock.patch.object(self.brain, 'edit', side_effect=AssertionError('No editing')):
            results = review_pending(self.brain, self.learning,
                                     self.llm(self.decision(action='update', supersedes=self.skill)))
        self.assertEqual(results[0]['status'], 'done')
        self.assertEqual(self.new_notes()[0]['meta']['supersedes'], self.skill)
        self.assertEqual(self.brain.get(self.skill)['hash'], before)
        self.assertEqual(self.new_notes()[0]['review'], 'proposed')

    def test_missing_supersedes_is_not_recorded(self):
        self.run_event()
        review_pending(self.brain, self.learning, self.llm(self.decision(action='update', supersedes='f' * 32)))
        self.assertNotIn('supersedes', self.new_notes()[0]['meta'])

    def test_no_source_creates_only_a_candidate_without_session_note(self):
        self.run_event(skills_used=['f' * 32])
        result = review_pending(self.brain, self.learning, self.llm(self.decision()))[0]
        self.assertEqual(result['status'], 'needs_source')
        notes = self.new_notes()
        self.assertEqual(len(notes), 1)
        self.assertEqual((notes[0]['kind'], notes[0]['review']), ('note', 'proposed'))
        self.assertNotIn('memory_type', notes[0]['meta'])
        self.assertEqual(notes[0]['meta']['memory_type_candidate'], 'skill')
        self.assertEqual(notes[0]['meta']['applies_to'], 'build projects')
        self.assertEqual(notes[0]['meta']['declared_links'], [])
        self.assertEqual(self.status()['status'], 'needs_source')

    def test_noop_done_without_writes(self):
        self.run_event()
        result = review_pending(self.brain, self.learning, self.llm('{"action":"noop"}'))
        self.assertEqual(result[0]['status'], 'done')
        self.assertEqual(self.new_notes(), [])
        self.assertIn('reason', json.loads(self.status()['detail']))
        self.assertEqual(review_pending(self.brain, self.learning, mock.Mock()), [])

    def test_fenced_json_is_supported(self):
        self.run_event()
        result = review_pending(self.brain, self.learning, self.llm('```json\n' + self.decision() + '\n```'))
        self.assertEqual(result[0]['status'], 'done')

    def test_invalid_or_ambiguous_json_fails_without_writes(self):
        invalid = ['not json', '{', '[]', '{"action":"delete"}',
                   self.decision() + '\nextra text', 'prefix ' + self.decision(),
                   '{"action":"noop","action":"add"}',
                   self.decision(review='reviewed'), self.decision(body=['not text']),
                   self.decision(evidence=''), '{"action":"noop","body":NaN}']
        for i, response in enumerate(invalid):
            with self.subTest(response=response):
                self.run_event(str(i))
                result = review_pending(self.brain, self.learning, self.llm(response))
                self.assertEqual(result[0]['status'], 'failed')
                self.assertEqual(self.status(str(i))['status'], 'failed')
                self.assertEqual(self.new_notes(), [])

    def test_budget_before_start_makes_no_call_and_leaves_pending(self):
        self.run_event()
        llm = mock.Mock()
        self.assertEqual(review_pending(self.brain, self.learning, llm, budget_chars=10), [])
        llm.assert_not_called()
        self.assertEqual(self.status()['status'], 'pending')

    def test_response_over_budget_stops_before_next_call_or_draft(self):
        self.run_event('first')
        self.run_event('second')
        llm = mock.Mock(return_value='x' * 60001)
        result = review_pending(self.brain, self.learning, llm)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'failed')
        self.assertIn('budget', result[0]['reason'])
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(self.status('second')['status'], 'pending')
        self.assertEqual(self.new_notes(), [])

    def test_second_prompt_and_responses_share_one_budget(self):
        self.run_event()
        # Start stage 1 but leave only one character for the second prompt.
        budget = 10000
        llm = mock.Mock(side_effect=lambda prompt: 'x' * (budget - len(prompt) - 1))
        result = review_pending(self.brain, self.learning, llm, budget_chars=budget)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result[0]['reason'], 'Character budget exhausted')
        self.assertEqual(self.new_notes(), [])

    def test_limits_oldest_first_and_budget_across_triggers(self):
        self.run_event('first')
        self.run_event('second')
        result = review_pending(self.brain, self.learning, self.llm('{"action":"noop"}'), limit=1)
        self.assertEqual([r['run_id'] for r in result], ['first'])
        self.assertEqual(self.status('second')['status'], 'pending')
        # Equal-length run IDs give the same prompt lengths despite fresh delimiters.
        used = sum(map(len, self.prompts)) + len('A reported build error was corrected.') + len('{"action":"noop"}')
        self.run_event('third')
        self.prompts.clear()
        result = review_pending(self.brain, self.learning, self.llm('{"action":"noop"}'), budget_chars=used + 4)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(self.status('third')['status'], 'pending')

    def test_redacts_run_model_output_and_provider_errors(self):
        secret = 'ghp_abcdefghijklmnopqrstuvwxyz'
        self.run_event(task='build ' + secret, summary='password=private-value')
        result = review_pending(self.brain, self.learning,
                                self.llm(self.decision(body='token=' + secret, evidence='password=private-value')))
        self.assertEqual(result[0]['status'], 'done')
        for value in self.prompts + [self.new_notes()[0]['content'], self.status()['detail']]:
            self.assertNotIn(secret, value)
            self.assertNotIn('private-value', value)
        self.run_event('throws')
        result = review_pending(self.brain, self.learning, mock.Mock(side_effect=RuntimeError(secret)))
        self.assertEqual(result[0]['status'], 'failed')
        self.assertNotIn(secret, self.status('throws')['detail'])

    def test_disabled_queue_and_kill_switch_make_no_writes(self):
        self.run_event()
        self.learning.set_mechanism('weak', 'reflection', False, 'disabled', 'human:test')
        llm = mock.Mock()
        self.assertEqual(review_pending(self.brain, self.learning, llm), [])
        llm.assert_not_called()
        self.learning.set_mechanism('weak', 'reflection', True, 'enabled', 'human:test')

        def disable(prompt):
            self.learning.set_mechanism('weak', 'reflection', False, 'stop', 'human:test')
            return 'reflection'
        llm = mock.Mock(side_effect=disable)
        result = review_pending(self.brain, self.learning, llm)
        self.assertEqual(result[0]['status'], 'failed')
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(self.new_notes(), [])

    def test_frontier_manual_request_and_weak_distillation(self):
        self.run_event('manual', agent_model='large', user_correction=False)
        request_reflection(self.learning, 'manual')
        review_pending(self.brain, self.learning, self.llm(self.decision(title='Manual skill')))
        self.assertEqual(self.new_notes()[0]['meta']['agent_tier'], 'frontier')
        self.learning.set_mechanism('weak', 'skills', True, 'fixture', 'human:test')
        self.run_event('teacher', agent_model='large', steps=8)
        result = review_pending(self.brain, self.learning, self.llm(self.decision(title='Distilled skill')))
        note = self.brain.get(result[0]['note_id'])
        self.assertEqual((note['meta']['trigger'], note['meta']['for_tier']), ('distill', 'weak'))

    def test_wrong_actor_is_rejected_before_work(self):
        self.run_event()
        self.brain.actor = 'human:test'
        llm = mock.Mock()
        with self.assertRaises(ValueError):
            review_pending(self.brain, self.learning, llm)
        llm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
