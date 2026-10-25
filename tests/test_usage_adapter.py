import json
from pathlib import Path
import subprocess
import sys
import unittest

from second_brain.evals import _tokens
from second_brain.usage_adapter import tokens

REPO = Path(__file__).resolve().parent.parent
CLAUDE = json.dumps({'type': 'result', 'result': 'ok', 'usage': {
    'input_tokens': 2, 'cache_creation_input_tokens': 10758, 'cache_read_input_tokens': 21540, 'output_tokens': 4}})
CODEX = '\n'.join(json.dumps(e) for e in [
    {'type': 'thread.started'}, {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'cached_input_tokens': 50,
                                                                     'output_tokens': 20, 'reasoning_output_tokens': 5}}])
OPENCODE = '\n'.join(json.dumps(e) for e in [
    {'type': 'step_finish', 'part': {'tokens': {'input': 10, 'output': 3, 'reasoning': 1, 'cache': {'read': 900}}}},
    {'type': 'text', 'part': {'text': 'ok'}},
    {'type': 'step_finish', 'part': {'tokens': {'input': 12, 'output': 4, 'reasoning': 0, 'cache': {'read': 900}}}}])


class UsageAdapterTests(unittest.TestCase):
    def test_provider_formats_count_input_output_reasoning_without_cache(self):
        self.assertEqual(tokens(CLAUDE, 'claude'), 6)
        self.assertEqual(tokens(CODEX, 'codex'), 125)
        self.assertEqual(tokens(OPENCODE, 'opencode'), 30)
        self.assertIsNone(tokens('plain text, no json', 'claude'))
        self.assertIsNone(tokens(json.dumps({'type': 'result'}), 'claude'))

    def test_wrapper_passes_output_and_exit_code_and_feeds_the_eval_parser(self):
        fake = f'import sys; print({CLAUDE!r}); sys.exit(3)'
        done = subprocess.run([sys.executable, '-m', 'second_brain.usage_adapter', '--format', 'claude', '--',
                               sys.executable, '-c', fake], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(done.returncode, 3)
        self.assertIn('"result": "ok"', done.stderr)
        self.assertEqual(_tokens(done.stdout), 6)  # exactly one usage line: eval-run trusts it

    def test_no_usage_leaves_cost_unknown(self):
        done = subprocess.run([sys.executable, '-m', 'second_brain.usage_adapter', '--format', 'codex', '--',
                               sys.executable, '-c', 'print("hi")'], cwd=REPO, capture_output=True, text=True)
        self.assertIsNone(_tokens(done.stdout))


if __name__ == '__main__':
    unittest.main()
