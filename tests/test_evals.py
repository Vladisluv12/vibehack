import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
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
        args = dict(tasks=self.tasks, config=config, tier='weak', agent_model='fake', agent_argv=AGENT,
                    learning=self.learning)
        return run_batch(**(args | kwargs))

    def relock(self, tasks):
        for path in self.task_dir.glob('*.json'):
            path.unlink()
        for index, task in enumerate(tasks):
            definition = {key: value for key, value in task.items() if key != 'task_hash' and not key.startswith('_')}
            path = self.task_file if index == 0 else self.task_dir / f'task-{index}.json'
            path.write_text(json.dumps(definition), encoding='utf-8')
        lock_tasks(self.task_dir)
        self.tasks = load_tasks(self.task_dir)

    def stored_batch(self, name, successes, *, tokens=10, seconds=2.0, ts=1, ids=None, signature='failure',
                     tier='weak', config='C', agent_model='fake', task_hash=None):
        with self.learning.db:
            for index, success in enumerate(successes):
                ident = ids[index] if ids else f'task-{index:03}'
                digest = task_hash if task_hash is not None else hashlib.sha256(ident.encode('utf-8')).hexdigest()
                self.learning.db.execute('''INSERT INTO eval_results
                    (batch, task_id, tier, config, agent_model, success, tokens, seconds, signature, ts, task_hash)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (name, ident, tier, config, agent_model, int(success), tokens, seconds,
                     '' if success else signature, ts, digest))
        return name

    def decision(self, baseline='base', candidate='candidate', **kwargs):
        return decide(self.learning, baseline, candidate, **({'tier': 'weak', 'mechanism': 'skills'} | kwargs))

    def test_lock_uses_exact_json_and_sorted_input_bytes_and_relative_paths(self):
        lock = json.loads((self.task_dir.parent / 'tasks.lock').read_text(encoding='utf-8'))
        files = {}
        for directory in (self.task_dir.parent / 'workspace', self.task_dir / 'oracle'):
            for path in directory.rglob('*'):
                if path.is_file():
                    files[path.relative_to(self.task_dir.parent).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.dumps(sorted(files.items()), separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        expected = hashlib.sha256(self.task_file.read_bytes() + b'\0' + manifest).hexdigest()
        self.assertEqual(lock, {'skill-task': expected})
        self.assertEqual(self.tasks[0]['task_hash'], expected)
        self.assertEqual(Path(self.tasks[0]['workspace']), self.task_dir.parent / 'workspace')
        self.assertEqual(Path(self.tasks[0]['oracle']), self.task_dir / 'oracle')
        self.assertEqual(excluded_sources(self.tasks), {'eval:skill-task'})
        self.task_file.write_text(self.task_file.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
            load_tasks(self.task_dir)
        lock_tasks(self.task_dir)  # re-locking is explicit
        self.assertEqual(load_tasks(self.task_dir)[0]['id'], 'skill-task')

    def test_missing_oracle_is_refused_even_with_an_absolute_check_executable(self):
        task = json.loads(self.task_file.read_text(encoding='utf-8'))
        del task['oracle']
        self.task_file.write_text(json.dumps(task), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'oracle'):
            lock_tasks(self.task_dir)
        with self.assertRaisesRegex(ValueError, 'oracle'):
            load_tasks(self.task_dir)

    def test_every_workspace_and_oracle_file_is_locked_including_additions_and_removals(self):
        for field in ('workspace', 'oracle'):
            directory = Path(self.tasks[0][field])
            original = directory / 'expected.txt'
            raw = original.read_bytes()
            with self.subTest(field=field, change='edit'):
                original.write_bytes(raw + b'changed')
                with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
                    load_tasks(self.task_dir)
                original.write_bytes(raw)
            added = directory / 'nested' / 'added.bin'
            added.parent.mkdir()
            with self.subTest(field=field, change='add'):
                added.write_bytes(b'\x00\xff')
                with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
                    load_tasks(self.task_dir)
            lock_tasks(self.task_dir)
            with self.subTest(field=field, change='remove'):
                added.unlink()
                with self.assertRaisesRegex(ValueError, 'changed since tasks.lock'):
                    load_tasks(self.task_dir)
            added.parent.rmdir()
            lock_tasks(self.task_dir)

    def test_symlinks_are_refused_in_both_input_trees_and_task_json(self):
        for field in ('workspace', 'oracle'):
            directory = Path(self.tasks[0][field])
            nested = directory / 'nested'
            nested.mkdir()
            for target in (directory / 'expected.txt', self.root / 'outside.txt', directory):
                link = nested / 'link'
                with self.subTest(field=field, target=target):
                    link.symlink_to(target)
                    for load in (lock_tasks, load_tasks):
                        with self.assertRaisesRegex(ValueError, 'symlinks'):
                            load(self.task_dir)
                    link.unlink()
            nested.rmdir()
        raw = self.task_file.read_bytes()
        target = self.task_dir / 'definition.txt'
        target.write_bytes(raw)
        self.task_file.unlink()
        self.task_file.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            load_tasks(self.task_dir)

    def test_symlink_input_directories_and_ancestors_are_refused(self):
        original = json.loads(self.task_file.read_text(encoding='utf-8'))
        for field in ('workspace', 'oracle'):
            for suffix in ('', '/expected.txt/..'):
                link = self.task_dir.parent / 'input-link'
                link.symlink_to(self.tasks[0][field], target_is_directory=True)
                task = {**original, field: '../input-link' + suffix}
                self.task_file.write_text(json.dumps(task), encoding='utf-8')
                with self.subTest(field=field, suffix=suffix), self.assertRaisesRegex(ValueError, 'symlinks'):
                    load_tasks(self.task_dir)
                link.unlink()
        self.task_file.write_text(json.dumps(original), encoding='utf-8')

    def test_input_paths_outside_tasks_root_are_refused(self):
        outside = self.root / 'outside'
        outside.mkdir()
        original = json.loads(self.task_file.read_text(encoding='utf-8'))
        for field in ('workspace', 'oracle'):
            for value in ('../../outside', str(outside)):
                self.task_file.write_text(json.dumps({**original, field: value}), encoding='utf-8')
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'outside tasks root'):
                    lock_tasks(self.task_dir)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'requires FIFO support')
    def test_fifos_in_either_input_tree_are_refused_without_reading(self):
        for field in ('workspace', 'oracle'):
            fifo = Path(self.tasks[0][field]) / 'fifo'
            os.mkfifo(fifo)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'regular files'):
                load_tasks(self.task_dir)
            fifo.unlink()

    def test_device_inputs_are_refused_before_reading(self):
        original = Path.lstat
        device = Path(self.tasks[0]['workspace']) / 'expected.txt'

        def device_stat(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            return os.stat_result((stat.S_IFCHR | 0o600, *info[1:])) if path == device else info

        with mock.patch.object(Path, 'lstat', device_stat):
            with self.assertRaisesRegex(ValueError, 'regular files'):
                load_tasks(self.task_dir)

    def test_changed_inputs_after_loading_are_refused_before_execution(self):
        (Path(self.tasks[0]['workspace']) / 'expected.txt').write_text('changed', encoding='utf-8')
        with mock.patch('second_brain.evals.subprocess.run', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'changed since load_tasks'):
                self.batch()

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
        self.assertEqual(row['task_hash'], self.tasks[0]['task_hash'])
        c = self.batch('C', skill_pack=lambda task: 'EVAL_SKILL_MARKER')
        self.assertEqual(self.decision(baseline, c)['recommend'], 'enable')
        self.assertFalse(self.learning.enabled('weak', 'skills'))

    def test_C_hook_observes_committed_result_and_fresh_workspace_then_cleanup(self):
        self.tasks.append({**self.tasks[0], 'id': 'second-task'})
        seen, prompts = [], []
        original = self.tasks[0]['prompt'] + '\nquotes " and $() and literal {workspace} and {prompt}'
        for task in self.tasks:
            task['prompt'] = original
        self.relock(self.tasks)

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
            self.assertEqual(result['task_hash'], task['task_hash'])

        batch = self.batch('C', skill_pack=lambda task: 'EVAL_SKILL_MARKER', after_run=hook)
        self.assertEqual(metrics(self.learning, batch)['successes'], 2)
        self.assertEqual(len(set(seen)), 2)
        self.assertTrue(all(not path.exists() for path in seen))
        self.assertEqual(prompts, ['EVAL_SKILL_MARKER\n\n' + original] * 2)
        self.assertEqual(self.tasks[0]['prompt'], original)
        source = self.task_dir.parent / 'workspace'
        self.assertFalse((source / 'answer.txt').exists())
        self.assertFalse((source / 'ran.txt').exists())

    def test_agent_overwriting_its_checker_and_expected_input_still_fails(self):
        seen = []
        batch = self.batch('C', after_run=lambda task, result: seen.append(result),
                           agent_argv=AGENT + ['--mode', 'overwrite-check'])
        self.assertEqual(metrics(self.learning, batch)['success_rate'], 0)
        self.assertEqual(seen[0]['agent']['returncode'], 0)
        self.assertEqual(seen[0]['check']['returncode'], 1)
        self.assertEqual(seen[0]['signature'], error_signature('answer does not match expected input\n'))

    def test_oracle_is_copied_to_a_separate_fresh_directory_after_the_agent_exits(self):
        from second_brain import evals
        original_run, original_copy = evals._run, evals._copy_input
        exited, oracle_dirs, workspaces = [], [], []

        def run(argv, directory, timeout):
            if argv[1] == 'check.py':
                self.assertTrue(exited)
                self.assertEqual(Path(directory), oracle_dirs[-1])
                self.assertEqual(Path(argv[2]), workspaces[-1])
                self.assertFalse(Path(directory).is_relative_to(workspaces[-1]))
            result = original_run(argv, directory, timeout)
            if '--workspace' in argv:
                workspaces.append(Path(directory))
                exited.append(True)
            return result

        def copy(source, destination):
            if str(source) == self.tasks[0]['oracle']:
                self.assertTrue(exited)
                oracle_dirs.append(destination)
            original_copy(source, destination)

        with mock.patch.object(evals, '_run', side_effect=run), mock.patch.object(evals, '_copy_input', side_effect=copy):
            self.batch()
            self.batch()
        self.assertEqual(len(set(oracle_dirs)), 2)
        self.assertTrue(all(not path.exists() for path in oracle_dirs + workspaces))

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
        self.stored_batch('base', [True, False] * 10, config='A')
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
        self.stored_batch('base', [False] * 10 + [True] * 10, config='A')
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
        self.stored_batch('base', [False] * 10 + [True] * 10, config='A')
        self.stored_batch('candidate', [True] * 20, tokens=24, ts=2)
        self.assertEqual(self.decision()['recommend'], 'enable')
        self.stored_batch('expensive', [True] * 20, tokens=25, ts=3)
        result = self.decision(candidate='expensive')
        self.assertGreater(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.assertIn('cost', result['reason'])

    def test_missing_or_conflicting_usage_is_null_but_identical_totals_are_known(self):
        for flags, expected in ((['--no-usage'], None), (['--extra-tokens', '11'], None),
                                (['--extra-tokens', '10'], 10), (['--tokens', '0'], 0)):
            with self.subTest(flags=flags):
                batch = self.batch('C', agent_argv=AGENT + flags, skill_pack=lambda task: 'EVAL_SKILL_MARKER')
                row = self.learning.db.execute('SELECT tokens FROM eval_results WHERE batch=?', (batch,)).fetchone()
                self.assertEqual(row['tokens'], expected)
                report = metrics(self.learning, batch)
                self.assertEqual(report['tokens'], expected)
                self.assertEqual(report['cost_per_success']['tokens'], expected)
                self.assertGreater(report['cost_per_success']['seconds'], 0)

    def test_malformed_usage_cannot_supply_a_known_token_total(self):
        from second_brain.evals import _tokens
        for usage in (None, {}, {'tokens': None}, {'tokens': True}, {'tokens': -1},
                      {'tokens': 10.5}, {'tokens': '10'}):
            stdout = json.dumps({'usage': {'tokens': 10}}) + '\n' + json.dumps({'usage': usage})
            with self.subTest(usage=usage):
                self.assertIsNone(_tokens(stdout))
        self.assertIsNone(_tokens('agent log\n[1, 2]\nnull\n{}'))

    def test_null_tokens_in_either_batch_block_enable_including_failed_tasks(self):
        self.stored_batch('base', [False] * 10 + [True] * 10, config='A')
        self.stored_batch('candidate', [True] * 20, ts=2)
        for batch, task_id in (('base', 'task-000'), ('base', 'task-019'), ('candidate', 'task-000')):
            with self.subTest(batch=batch, task_id=task_id):
                with self.learning.db:
                    self.learning.db.execute('UPDATE eval_results SET tokens=NULL WHERE batch=? AND task_id=?',
                                             (batch, task_id))
                self.assertIsNone(metrics(self.learning, batch)['tokens'])
                result = self.decision()
                self.assertGreater(result['ci_low'], 0)
                self.assertEqual(result['decision'], 'disable')
                self.assertNotIn('recommend', result)
                self.assertIn('unknown', result['reason'])
                switch = self.learning.mechanisms()[0]
                self.assertEqual((switch['enabled'], switch['actor']), (0, 'eval'))
                with self.learning.db:
                    self.learning.db.execute('UPDATE eval_results SET tokens=10 WHERE batch=?', (batch,))

    def test_null_baseline_tokens_block_enable_even_without_baseline_successes(self):
        self.stored_batch('base', [False] * 20, tokens=None, config='A')
        self.stored_batch('candidate', [True] * 20, ts=2)
        self.assertEqual(self.decision()['decision'], 'disable')

    def test_time_gate_exact_boundary_and_custom_threshold(self):
        self.stored_batch('base', [False] * 10 + [True] * 10, seconds=5, config='A')
        self.stored_batch('candidate', [True] * 20, seconds=12, ts=2)
        self.assertEqual(self.decision()['recommend'], 'enable')
        self.stored_batch('slow', [True] * 20, seconds=12.01, ts=3)
        result = self.decision(candidate='slow')
        self.assertGreater(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.assertIn('Time cost', result['reason'])
        self.assertEqual(self.decision(candidate='slow', max_time_increase=0.21)['recommend'], 'enable')
        for increase in (-0.1, float('inf'), float('nan')):
            with self.subTest(increase=increase), self.assertRaisesRegex(ValueError, 'max_time_increase'):
                self.decision(max_time_increase=increase)

    def test_zero_and_undefined_baseline_time_use_the_token_cost_semantics(self):
        self.stored_batch('base', [False] * 10 + [True] * 10, seconds=0, config='A')
        self.stored_batch('candidate', [True] * 20, seconds=0, ts=2)
        self.assertEqual(self.decision()['recommend'], 'enable')
        self.stored_batch('slow', [True] * 20, seconds=0.001, ts=3)
        self.assertEqual(self.decision(candidate='slow')['decision'], 'disable')
        self.stored_batch('all-failed', [False] * 20, seconds=2, ts=4, config='A')
        self.assertEqual(self.decision(baseline='all-failed', candidate='slow')['recommend'], 'enable')

    def test_gain_requires_strict_lower_bound_not_just_positive_mean(self):
        self.stored_batch('base', [False, False], config='A')
        self.stored_batch('candidate', [True, False], ts=2)
        result = self.decision()
        self.assertEqual(result['gain'], 0.5)
        self.assertEqual(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.stored_batch('perfect', [True, True], ts=3)
        self.assertEqual(self.decision(candidate='perfect', min_gain=1)['decision'], 'disable')

    def test_decision_pairs_the_complete_task_set_by_id_independently_of_row_order(self):
        self.stored_batch('base', [False, False], ids=['z', 'a'], config='A')
        self.stored_batch('candidate', [True, True], ids=['a', 'z'], ts=2)
        result = self.decision()
        self.assertEqual(result['paired_tasks'], 2)
        self.assertEqual(result['costs']['candidate']['tokens'], 10)
        self.assertEqual(result['recommend'], 'enable')

    def test_mismatched_task_sets_raise_without_touching_any_switch(self):
        self.stored_batch('base', [False, False], ids=['a', 'b'], config='A')
        for index, ids in enumerate((['a'], ['a', 'b', 'extra'], ['different'])):
            candidate = f'candidate-{index}'
            self.stored_batch(candidate, [True] * len(ids), ids=ids, ts=2)
            self.learning.set_mechanism('weak', 'skills', True, 'human enabled', 'human')
            before = self.learning.mechanisms()
            with self.subTest(ids=ids), mock.patch.object(self.learning, 'set_mechanism') as switch:
                with self.assertRaisesRegex(ValueError, 'same task ids'):
                    self.decision(candidate=candidate)
                switch.assert_not_called()
            self.assertEqual(self.learning.mechanisms(), before)

    def test_mismatched_model_tier_or_hash_in_any_row_raises_without_switch_changes(self):
        self.stored_batch('base', [False, False], config='A')
        self.stored_batch('candidate', [True, True], ts=2)
        self.learning.set_mechanism('weak', 'skills', True, 'human enabled', 'human')
        before = self.learning.mechanisms()
        for batch in ('base', 'candidate'):
            for field, value, message in (('agent_model', 'other-model', 'same agent_model'),
                                           ('tier', 'frontier', 'requested tier'),
                                           ('task_hash', 'f' * 64, 'identical task_hash'),
                                           ('task_hash', None, 'identical task_hash')):
                row = self.learning.db.execute('SELECT * FROM eval_results WHERE batch=? LIMIT 1', (batch,)).fetchone()
                with self.learning.db:
                    self.learning.db.execute(f'UPDATE eval_results SET {field}=? WHERE batch=? AND task_id=?',
                                             (value, batch, row['task_id']))
                with self.subTest(batch=batch, field=field, value=value), mock.patch.object(self.learning, 'set_mechanism') as switch:
                    with self.assertRaisesRegex(ValueError, message):
                        self.decision()
                    switch.assert_not_called()
                self.assertEqual(self.learning.mechanisms(), before)
                with self.learning.db:
                    self.learning.db.execute(f'UPDATE eval_results SET {field}=? WHERE batch=? AND task_id=?',
                                             (row[field], batch, row['task_id']))

    def test_missing_hashes_in_both_legacy_batches_are_not_comparable(self):
        self.stored_batch('base', [False], config='A')
        self.stored_batch('candidate', [True], ts=2)
        with self.learning.db:
            self.learning.db.execute('UPDATE eval_results SET task_hash=NULL')
        with mock.patch.object(self.learning, 'set_mechanism') as switch:
            with self.assertRaisesRegex(ValueError, 'identical task_hash'):
                self.decision()
            switch.assert_not_called()

    def test_config_pairs_are_required_for_each_mechanism_and_checked_in_every_row(self):
        self.stored_batch('base', [False, False], config='A')
        self.stored_batch('candidate', [True, True], ts=2)
        self.learning.set_mechanism('weak', 'skills', True, 'human enabled', 'human')
        self.learning.set_mechanism('weak', 'reflection', True, 'human enabled', 'human')
        before = self.learning.mechanisms()
        for mechanism, expected in (('skills', 'A'), ('reflection', 'B')):
            for batch, config in (('base', 'C'), ('candidate', 'B')):
                with self.learning.db:
                    self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=?', (expected, 'base'))
                    self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=?', ('C', 'candidate'))
                    self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=? AND task_id=?',
                                             (config, batch, 'task-000'))
                with self.subTest(mechanism=mechanism, batch=batch), mock.patch.object(self.learning, 'set_mechanism') as switch:
                    with self.assertRaisesRegex(ValueError, f'{mechanism} requires baseline config {expected}'):
                        self.decision(mechanism=mechanism)
                    switch.assert_not_called()
                self.assertEqual(self.learning.mechanisms(), before)
        with self.learning.db:
            self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=?', ('A', 'base'))
            self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=?', ('C', 'candidate'))
        with self.assertRaisesRegex(ValueError, 'reflection requires baseline config B'):
            self.decision(mechanism='reflection')
        with self.learning.db:
            self.learning.db.execute('UPDATE eval_results SET config=? WHERE batch=?', ('B', 'base'))
        with self.assertRaisesRegex(ValueError, 'skills requires baseline config A'):
            self.decision()
        self.assertEqual(self.decision(mechanism='reflection')['recommend'], 'enable')
        self.assertEqual(self.learning.mechanisms(), before)

    def test_zero_baseline_cost_and_no_successes_costs(self):
        self.stored_batch('base', [False] * 10 + [True] * 10, tokens=0, config='A')
        self.stored_batch('candidate', [True] * 20, tokens=1, ts=2)
        result = self.decision()
        self.assertGreater(result['ci_low'], 0)
        self.assertEqual(result['decision'], 'disable')
        self.assertIn('Token cost', result['reason'])
        self.stored_batch('free', [True] * 20, tokens=0, ts=3)
        self.assertEqual(self.decision(candidate='free')['recommend'], 'enable')
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
            self.relock([task])
            batch = run_batch(self.tasks, config='C', tier='weak', agent_model='fake', agent_argv=AGENT,
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
        self.assertEqual((report['success_rate'], report['tokens']), (1, None))
        self.tasks[0]['check']['expect_exit'] = 1
        self.relock(self.tasks)
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
