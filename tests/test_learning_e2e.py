"""The whole loop through the host CLI: weak run → trigger → reviewer → human review →
skill pack → A/B evaluation. Fake model and agent scripts, no network."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from second_brain.brain import Brain
from second_brain.cli import main

FIXTURES = Path(__file__).parent / 'fixtures' / 'evals'
FAKE_LLM = '''import json, sys
prompt = sys.stdin.read()
if 'Stage 1' in prompt:
    print('The copy failed twice for the same reason; the working fix is reproducible.')
else:
    print(json.dumps({"action": "add", "title": "Copy files for the evaluated skill",
                      "body": "Copy a file using the evaluated skill: EVAL_SKILL_MARKER copy expected.txt to answer.txt.",
                      "applies_to": "copy a file using the evaluated skill", "evidence": "check.py passed",
                      "supersedes": None}))
'''


class LearningLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = ['--vault', str(self.root / 'vault'), '--state', str(self.root / 'state')]
        (self.root / 'state').mkdir()
        (self.root / 'state' / 'agents.json').write_text(json.dumps({'models': {'qwen-small': 'weak'}}))

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *argv, code=0):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                main(self.base + list(argv))
                got = 0
            except SystemExit as exc:
                got = exc.code
        self.assertEqual(got, code, out.getvalue())
        return json.loads(out.getvalue())

    def write(self, name, data):
        path = self.root / name
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding='utf-8')
        return path

    def test_weak_model_learns_a_skill_that_a_measured_batch_rewards(self):
        brain = Brain(self.root / 'vault', self.root / 'state')
        old = brain.create('Old copy advice', 'Copy with a shell loop.', review='reviewed')
        brain.sync(); brain.close()
        self.cli('mechanism', '--tier', 'weak', '--name', 'reflection', '--on', '--reason', 'pilot')
        error = 'Traceback\n  File "/tmp/w1/check.py", line 4\nAssertionError: answer.txt differs'
        first = self.cli('ingest-run', '--model', 'qwen-small', str(self.write('r1.json', {
            'run_id': 'r1', 'agent_model': 'qwen-small', 'task': 'copy a file', 'outcome': 'failure',
            'errors': [error], 'transcript': 'must be dropped'})))
        self.assertEqual(first[0]['triggers'], [])
        second = self.cli('ingest-run', '--model', 'qwen-small', str(self.write('r2.json', {
            'run_id': 'r2', 'agent_model': 'qwen-small', 'task': 'copy a file using the evaluated skill',
            'outcome': 'success', 'steps': 3, 'errors': [error.replace('/tmp/w1', '/tmp/w2')],
            'contradicts': [old]})))
        self.assertEqual(second[0]['tier'], 'weak')
        self.assertEqual(set(second[0]['triggers']), {'repeated_error', 'contradiction'})
        frontier = self.cli('ingest-run', '--model', 'claude-opus-5-5', str(self.write('r3.json', {
            'run_id': 'r3', 'agent_model': 'claude-opus-5-5', 'task': 'copy', 'outcome': 'success',
            'steps': 2, 'user_correction': True})))
        self.assertEqual((frontier[0]['tier'], frontier[0]['triggers']), ('frontier', []))

        llm = self.write('llm.json', [sys.executable, str(self.write('fake_llm.py', FAKE_LLM))])
        reviewed = self.cli('review', '--llm', str(llm), '--llm-model', 'fake-llm')
        drafts = {r['note_id'] for r in reviewed if r['status'] == 'done'}
        self.assertEqual(len(drafts), 1)  # the second trigger proposes the same draft: deduplicated
        skill = drafts.pop()
        brain = Brain(self.root / 'vault', self.root / 'state')
        brain.sync()
        note = brain.get(skill)
        self.assertEqual((note['review'], note['meta']['memory_type'], note['meta']['agent_model']),
                         ('proposed', 'skill', 'reviewer'))
        self.assertEqual(self.cli('skill-pack', '--model', 'qwen-small', '--task', 'copy a file', '--run-id', 'x')['skill_ids'], [])

        brain.edit(skill, note['content'].replace('review: "proposed"', 'review: "reviewed"'), note['hash'])  # the human
        brain.close()
        self.cli('mechanism', '--tier', 'weak', '--name', 'skills', '--on', '--reason', 'pilot')
        self.cli('promote', '--skill', skill, '--stage', 'full')
        pack = self.cli('skill-pack', '--model', 'qwen-small', '--task', 'how do I copy a file?', '--run-id', 'x')
        self.assertEqual(pack['skill_ids'], [skill])
        self.assertIn('EVAL_SKILL_MARKER', pack['text'])

        tasks = self.root / 'evals' / 'tasks'
        shutil.copytree(FIXTURES / 'tasks', tasks)
        shutil.copytree(FIXTURES / 'workspace', self.root / 'evals' / 'workspace')
        self.cli('eval-lock', str(tasks))
        agent = self.write('agent.json', [sys.executable, str(FIXTURES / 'fake_agent.py'), '--prompt', '{prompt}',
                                          '--workspace', '{workspace}'])
        a = self.cli('eval-run', '--tasks', str(tasks), '--config', 'A', '--model', 'qwen-small', '--agent', str(agent))
        b = self.cli('eval-run', '--tasks', str(tasks), '--config', 'B', '--model', 'qwen-small', '--agent', str(agent))
        c = self.cli('eval-run', '--tasks', str(tasks), '--config', 'C', '--model', 'qwen-small', '--agent', str(agent))
        self.assertEqual((a['success_rate'], b['success_rate'], c['success_rate']), (0.0, 0.0, 1.0))
        reflection = self.cli('eval-decide', '--baseline', b['batch_id'], '--candidate', c['batch_id'],
                              '--tier', 'weak', '--mechanism', 'reflection')
        self.assertEqual(reflection['decision'], 'recommend_enable')
        # skills: all skills (C) against none (A); reflection: reflection-made skills on top (B → C).
        skills_all = self.cli('eval-decide', '--baseline', a['batch_id'], '--candidate', c['batch_id'],
                              '--tier', 'weak', '--mechanism', 'skills')
        self.assertEqual(skills_all['decision'], 'recommend_enable')
        mechanisms = {(m['name'], m['enabled'], m['actor']) for m in self.cli('mechanism')}
        self.assertEqual(mechanisms, {('reflection', 1, 'human:cli'), ('skills', 1, 'human:cli')})
        # Mismatched pairs are refused outright (exit 1 with the error), without touching switches.
        with self.assertRaises(ValueError):
            self.cli('eval-decide', '--baseline', a['batch_id'], '--candidate', b['batch_id'],
                     '--tier', 'weak', '--mechanism', 'reflection')
        self.cli('promote', '--skill', skill, '--stage', 'off')  # the human withdraws the skill
        c2 = self.cli('eval-run', '--tasks', str(tasks), '--config', 'C', '--model', 'qwen-small', '--agent', str(agent))
        no_gain = self.cli('eval-decide', '--baseline', b['batch_id'], '--candidate', c2['batch_id'],
                           '--tier', 'weak', '--mechanism', 'reflection')
        self.assertEqual(no_gain['decision'], 'disable')
        mechanisms = {(m['name'], m['enabled'], m['actor']) for m in self.cli('mechanism')}
        self.assertEqual(mechanisms, {('reflection', 0, 'eval'), ('skills', 1, 'human:cli')})

        report = self.cli('lint')
        self.assertEqual(report['errors'], [])
        actors = {e['actor'] for e in Brain(self.root / 'vault', self.root / 'state').ledger.entries()
                  if e['phase'] == 'intent'}
        self.assertEqual(actors, {'human:local', 'agent:reviewer'})
        models = {e.get('agent_model') for e in Brain(self.root / 'vault', self.root / 'state').ledger.entries()
                  if e['phase'] == 'intent' and e['actor'] == 'agent:reviewer'}
        self.assertEqual(models, {'fake-llm'})

    def test_skill_configs_accept_any_task_id(self):
        tasks = self.root / 'evals' / 'tasks'
        shutil.copytree(FIXTURES / 'tasks', tasks)
        shutil.copytree(FIXTURES / 'workspace', self.root / 'evals' / 'workspace')
        task = json.loads((tasks / 'skill.json').read_text(encoding='utf-8'))
        (tasks / 'skill.json').write_text(json.dumps({**task, 'id': 'копия файла 1'}), encoding='utf-8')
        self.cli('eval-lock', str(tasks))
        agent = self.write('agent.json', [sys.executable, str(FIXTURES / 'fake_agent.py'), '--prompt', '{prompt}',
                                          '--workspace', '{workspace}'])
        result = self.cli('eval-run', '--tasks', str(tasks), '--config', 'B', '--model', 'qwen-small', '--agent', str(agent))
        self.assertEqual(result['tasks'], 1)


if __name__ == '__main__':
    unittest.main()
