import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from second_brain.evals import (decide, excluded_sources, load_agent_argv, load_tasks, lock_tasks,
                                metrics, run_batch)
from second_brain.learning import Learning, error_signature


FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'evals'
AGENT = [sys.executable, str(FIXTURES / 'fake_agent.py'), '--prompt', '{prompt}', '--workspace', '{workspace}']


class EvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(FIXTURES / 'tasks', self.root / 'evals' / 'tasks')
        shutil.copytree(FIXTURES / 'workspace', self.root / 'evals' / 'workspace')
        self.task_dir = self.root / 'evals' / 'tasks'
        self.task_file = self.task_dir / 'skill.json'
        task = json.loads(self.task_file.read_text(encoding='utf-8'))
        task['check']['argv'][0] = sys.executable
        self.task_file.write_text(json.dumps(task), encoding='utf-8')
        self.learning = Learning(self.root / 'state')
        self.addCleanup(self.learning.close)
        lock_tasks(self.task_dir)
        self.tasks = load_tasks(self.task_dir)

    def batch(self, config='A', **kwargs):
        return run_batch(self.tasks, config=config, tier='weak', agent_model='fake', agent_argv=AGENT,
                         learning=self.learning, **kwargs)

    def stored_batch(self, name, successes, *, tokens=10, seconds=2.0, ts=1, ids=None, signature='failure', tier='weak'):
        with self.learning.db:
            for index, success in enumerate(successes):
                self.learning.db.execute('INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?)',
                                         (name, ids[index] if ids else f'task-{index:03}', tier, 'A', 'fake',
                                          int(success), tokens, seconds, '' if success else signature, ts))
        return name

    def decision(self, baseline='base', candidate='candidate', **kwargs):
        return decide(self.learning, baseline, candidate, tier='weak', mechanism='skills', **kwargs)

    def test_lock_uses_exact_task_file_bytes_and_relative_workspaces(self):
        lock = json.loads((self.task_dir.parent / 'tasks.lock').read_text(encoding='utf-8'))
        self.assertEqual(lock, {'skill-task': hashlib.sha256(self.task_file.read_bytes()).hexdigest()})
        self.assertEqual(Path(self.tasks[0]['workspace']), self.task_dir.parent / 'workspace')
        self.assertEqual(excluded_sources(self.tasks), {'eval:skill-task'})
        self.task_file.write_text(self.task_file.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
            load_tasks(self.task_dir)
        lock_tasks(self.task_dir)  # re-locking is explicit
        self.assertEqual(load_tasks(self.task_dir)[0]['id'], 'skill-task')

    def test_added_removed_missing_lock_and_duplicate_id_are_refused(self):
        other = self.task_dir / 'other.json'
        task = json.loads(self.task_file.read_text(encoding='utf-8'))
        task['id'] = 'other'
        other.write_text(json.dumps(task), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
            load_tasks(self.task_dir)
        lock_tasks(self.task_dir)
        other.unlink()
        with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
            load_tasks(self.task_dir)
        (self.task_dir.parent / 'tasks.lock').unlink()
        with self.assertRaisesRegex(ValueError, 'Cannot read tasks.lock'):
            load_tasks(self.task_dir)
        shutil.copyfile(self.task_file, other)
        with self.assertRaisesRegex(ValueError, 'Duplicate eval task id'):
            lock_tasks(self.task_dir)

    def test_A_vs_B_use_fake_agent_and_persist_costs_and_signatures(self):
        pack = mock.Mock(return_value='EVAL_SKILL_MARKER: copy the expected input.')
        hook = mock.Mock()
        baseline = self.batch('A', skill_pack=pack, after_run=hook)
        pack.assert_not_called()
        candidate = self.batch('B', skill_pack=pack, after_run=hook)
        pack.assert_called_once()
        hook.assert_not_called()
        a, b = metrics(self.learning, baseline), metrics(self.learning, candidate)
        self.assertEqual(a['success_rate'], 0)
        self.assertEqual(a['cost_per_success'], {'tokens': None, 'seconds': None})
        self.assertEqual(b['success_rate'], 1)
        self.assertEqual(b['success_first_try'], 1)
        self.assertEqual((a['tokens'], b['tokens']), (10, 10))
        self.assertEqual(b['cost_per_success']['tokens'], 10)
        self.assertGreater(b['cost_per_success']['seconds'], 0)
        row = dict(self.learning.db.execute('SELECT * FROM eval_results WHERE batch=?', (baseline,)).fetchone())
        self.assertEqual(row['signature'], error_signature('answer does not match expected input\n'))
        self.assertEqual((row['config'], row['tier'], row['agent_model']), ('A', 'weak', 'fake'))
        self.assertEqual(self.decision(baseline, candidate)['recommend'], 'enable')
        self.assertFalse(self.learning.enabled('weak', 'skills'))

    def test_C_hook_observes_committed_result_and_fresh_workspace_then_cleanup(self):
        self.tasks.append({**self.tasks[0], 'id': 'second-task'})
        seen, prompts = [], []
        original = self.tasks[0]['prompt'] + '\nquotes " and $() and literal {workspace} and {prompt}'
        for task in self.tasks:
            task['prompt'] = original

        def hook(task, result):
            workspace = Path(result['workspace'])
            seen.append(workspace)
            prompts.append((workspace / 'prompt.txt').read_text(encoding='utf-8'))
            self.assertEqual((workspace / 'answer.txt').read_bytes(), (workspace / 'expected.txt').read_bytes())
            self.assertTrue(result['success'])
            self.assertEqual(result['agent']['returncode'], 0)
            self.assertEqual(result['check']['returncode'], 0)
            stored = self.learning.db.execute('SELECT count(*) FROM eval_results WHERE batch=?', (result['batch'],)).fetchone()[0]
            self.assertEqual(stored, len(seen))

        batch = self.batch('C', skill_pack=lambda task: 'EVAL_SKILL_MARKER', after_run=hook)
        self.assertEqual(metrics(self.learning, batch)['successes'], 2)
        self.assertEqual(len(set(seen)), 2)
        self.assertTrue(all(not path.exists() for path in seen))
        self.assertEqual(prompts, ['EVAL_SKILL_MARKER\n\n' + original] * 2)
        self.assertEqual(self.tasks[0]['prompt'], original)
        source = self.task_dir.parent / 'workspace'
        self.assertFalse((source / 'answer.txt').exists())
        self.assertFalse((source / 'ran.txt').exists())

    def test_defaults_for_B_and_C_do_not_invent_a_skill(self):
        for config in ('B', 'C'):
            self.assertEqual(metrics(self.learning, self.batch(config))['success_rate'], 0)

    def test_repeated_error_rate_counts_only_earlier_batch_same_task_signature(self):
        first = self.batch()
        second = self.batch()
        self.assertEqual(metrics(self.learning, first)['repeated_error_rate'], 0)
        self.assertEqual(metrics(self.learning, second)['repeated_error_rate'], 1)
        fixed = self.batch('B', skill_pack=lambda task: 'EVAL_SKILL_MARKER')
        self.assertEqual(metrics(self.learning, fixed)['repeated_error_rate'], 0)
        self.stored_batch('earlier', [False, False], ids=['x', 'y'], ts=1, signature='same')
        self.stored_batch('later', [False, False], ids=['x', 'z'], ts=2, signature='same')
        self.assertEqual(metrics(self.learning, 'later')['repeated_error_rate'], 0.5)
        self.stored_batch('different', [False], ids=['x'], ts=3, signature='other')
        self.assertEqual(metrics(self.learning, 'different')['repeated_error_rate'], 0)

    def test_repeated_errors_tie_break_batch_start_without_counting_current_batch(self):
        self.stored_batch('first', [False, False], ids=['x', 'y'], ts=1)
        self.stored_batch('second', [False, False], ids=['x', 'y'], ts=1)
        self.assertEqual(metrics(self.learning, 'first')['repeated_error_rate'], 0)
        self.assertEqual(metrics(self.learning, 'second')['repeated_error_rate'], 1)

    def test_no_gain_disables_mechanism_with_eval_actor_and_reason(self):
        self.stored_batch('base', [True, False] * 10)
        self.stored_batch('candidate', [True, False] * 10, ts=2)
        self.learning.set_mechanism('weak', 'skills', True, 'enabled by human', 'human')
        result = self.decision()
        self.assertEqual((result['gain'], result['ci_low'], result['ci_high']), (0, 0, 0))
        self.assertEqual(result['decision'], 'disable')
        self.assertFalse(self.learning.enabled('weak', 'skills'))
        switch = self.learning.mechanisms()[0]
        self.assertEqual(switch['actor'], 'eval')
        self.assertEqual(switch['reason'], result['reason'])

    def test_measured_gain_recommends_enable_without_changing_switch(self):
        self.stored_batch('base', [False] * 10 + [True] * 10)
        self.stored_batch('candidate', [True] * 20, ts=2)
        with mock.patch.object(self.learning, 'set_mechanism', wraps=self.learning.set_mechanism) as switch:
            result = self.decision(seed=123)
            switch.assert_not_called()
        self.assertEqual(result['gain'], 0.5)
        self.assertGreater(result['ci_low'], 0)
        self.assertLess(result['ci_low'], result['gain'])
        self.assertGreater(result['ci_high'], result['gain'])
        self.assertEqual(result['costs']['baseline']['tokens'], 20)
        self.assertEqual(result['costs']['candidate']['tokens'], 10)
        self.assertEqual(result['recommend'], 'enable')
        self.assertFalse(self.learning.enabled('weak', 'skills'))
        self.assertEqual(result, self.decision(seed=123))

    def test_cost_gate_and_exact_twenty_percent_boundary(self):
        self.stored_batch('base', [False] * 10 + [True] * 10)
        self.stored_batch('candidate', [True] * 20, tokens=24, ts=2)
        self.assertEqual(self.decision()['recommend'], 'enable')
        self.stored_batch('expensive', [True] * 20, tokens=25, ts=3)
        result = self.decision(candidate='expensive')
        self.assertGreater(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.assertIn('cost', result['reason'])

    def test_gain_requires_strict_lower_bound_not_just_positive_mean(self):
        self.stored_batch('base', [False, False])
        self.stored_batch('candidate', [True, False], ts=2)
        result = self.decision()
        self.assertEqual(result['gain'], 0.5)
        self.assertEqual(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.stored_batch('perfect', [True, True], ts=3)
        self.assertEqual(self.decision(candidate='perfect', min_gain=1)['decision'], 'disable')

    def test_decision_pairs_tasks_and_costs_and_refuses_tier_mismatch(self):
        self.stored_batch('base', [False], ids=['shared'])
        self.stored_batch('candidate', [True], ids=['shared'], ts=2)
        self.stored_batch('candidate', [False], ids=['unpaired'], tokens=100000, ts=2)
        result = self.decision()
        self.assertEqual(result['paired_tasks'], 1)
        self.assertEqual(result['costs']['candidate']['tokens'], 10)
        self.assertEqual(result['recommend'], 'enable')
        self.stored_batch('other-tier', [True], ids=['shared'], tier='frontier', ts=3)
        with self.assertRaisesRegex(ValueError, 'requested tier'):
            self.decision(candidate='other-tier')
        self.stored_batch('no-overlap', [True], ids=['different'], ts=4)
        self.assertEqual(self.decision(candidate='no-overlap')['decision'], 'disable')

    def test_zero_baseline_cost_and_no_successes_costs(self):
        self.stored_batch('base', [False, True], tokens=0)
        self.stored_batch('candidate', [True, True], tokens=1, ts=2)
        self.assertEqual(self.decision()['decision'], 'disable')
        self.stored_batch('failures', [False, False], ts=3)
        report = metrics(self.learning, 'failures')
        self.assertEqual(report['cost_per_success'], {'tokens': None, 'seconds': None})

    def test_agent_timeout_still_runs_check_and_persists_failure(self):
        captured = []
        batch = run_batch(self.tasks, config='C', tier='weak', agent_model='fake',
                          agent_argv=AGENT + ['--mode', 'slow'], learning=self.learning,
                          timeout=0.3, after_run=lambda task, result: captured.append(result))
        self.assertEqual(metrics(self.learning, batch)['success_rate'], 0)
        self.assertTrue(captured[0]['agent']['timed_out'])
        self.assertEqual(captured[0]['check']['returncode'], 1)
        self.assertTrue(captured[0]['signature'])

    def test_check_timeout_or_missing_executable_is_not_an_expected_failure_exit(self):
        captured = []
        for argv in ([sys.executable, '-c', 'import time; time.sleep(10)'], ['missing-eval-command-12345']):
            task = {**self.tasks[0], 'check': {'argv': argv, 'expect_exit': 1}}
            batch = run_batch([task], config='C', tier='weak', agent_model='fake', agent_argv=AGENT,
                              learning=self.learning, timeout=0.2,
                              after_run=lambda task, result: captured.append(result))
            self.assertEqual(metrics(self.learning, batch)['success_rate'], 0)
            self.assertIsNone(captured[-1]['check']['returncode'])
        self.assertTrue(captured[0]['check']['timed_out'])
        self.assertFalse(captured[1]['check']['timed_out'])

    def test_oracle_decides_success_even_if_agent_exit_is_nonzero(self):
        batch = run_batch(self.tasks, config='B', tier='weak', agent_model='fake',
                          agent_argv=AGENT + ['--mode', 'nonzero', '--no-usage'], learning=self.learning,
                          skill_pack=lambda task: 'EVAL_SKILL_MARKER')
        report = metrics(self.learning, batch)
        self.assertEqual((report['success_rate'], report['tokens']), (1, 0))
        self.tasks[0]['check']['expect_exit'] = 1
        self.assertEqual(metrics(self.learning, self.batch())['success_rate'], 1)

    def test_agents_json_argv_and_validation_before_execution(self):
        agents = self.root / 'agents.json'
        agents.write_text(json.dumps(AGENT), encoding='utf-8')
        self.assertEqual(load_agent_argv(agents), AGENT)
        agents.write_text(json.dumps('python3 agent.py'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'argv'):
            load_agent_argv(agents)
        with mock.patch('second_brain.evals.subprocess.run', side_effect=AssertionError('must not execute')):
            for changes in ({'config': 'D'}, {'tier': 'other'}, {'timeout': 0}, {'agent_argv': 'shell command'},
                            {'tasks': []}, {'tasks': [self.tasks[0]] * 2}):
                args = dict(tasks=self.tasks, config='A', tier='weak', agent_model='fake',
                            agent_argv=AGENT, learning=self.learning)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    run_batch(**(args | changes))
        with self.assertRaisesRegex(ValueError, 'Unknown or empty'):
            metrics(self.learning, 'missing')
        with self.assertRaisesRegex(ValueError, 'resamples'):
            self.decision(resamples=0)


if __name__ == '__main__':
    unittest.main()
