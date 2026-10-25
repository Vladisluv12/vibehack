from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from second_brain.brain import Brain
from second_brain.mcp_server import build, propose_draft


class FakeMCP:
    """Stands in for fastmcp: records tools so the real tool bodies run in tests."""
    def __init__(self, name, instructions=''):
        self.tools = {}

    def tool(self, annotations=None):
        def register(func):
            self.tools[func.__name__] = func
            return func
        return register


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.brain = Brain(root/'vault', root/'state')

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def edges(self, ident):
        return {(r['target'], r['relation']) for r in self.brain.db.execute(
            'SELECT target, relation FROM links WHERE source=?', (ident,))}

    def test_draft_text_links_are_suggestions_not_graph_edges(self):
        source = self.brain.create('Источник', 'факт', kind='source', review='unreviewed')
        other = self.brain.create('Старая гипотеза', 'текст', review='reviewed')
        self.brain.sync()
        body = (f'Новая версия.\n\n- contradicts [[{other}]]\n- supersedes [[Старая гипотеза]]\n'
                f'```\n[[{source}]] пример в коде\n```\n')
        result = propose_draft(self.brain, 'Черновик', body, [source])

        self.assertEqual(result['review'], 'proposed')
        self.assertEqual(self.edges(result['id']), {(source, 'relates_to')})
        self.assertEqual(result['proposed_links'], [other, 'Старая гипотеза'])
        note = self.brain.get(result['id'])
        self.assertTrue(note['meta']['declared_only'])
        self.assertEqual(note['meta']['proposed_links'], [other, 'Старая гипотеза'])
        self.assertEqual(self.brain.sync()['warning_count'], 0)

    def test_draft_without_text_links_has_no_suggestions(self):
        result = propose_draft(self.brain, 'Черновик', 'без ссылок', [])
        self.assertEqual(result['proposed_links'], [])
        self.assertNotIn('proposed_links', self.brain.get(result['id'])['meta'])
        self.assertEqual(self.edges(result['id']), set())

    def test_draft_rejects_unknown_source_and_oversize(self):
        with self.assertRaises(Exception):
            propose_draft(self.brain, 'Черновик', 'текст', ['0' * 32])
        with self.assertRaises(ValueError):
            propose_draft(self.brain, 'Черновик', 'x' * 100001, [])
        with self.assertRaises(ValueError):
            propose_draft(self.brain, 'Черновик', 'текст', ['0' * 32] * 31)

    def test_mcp_tool_uses_declared_only_drafts(self):
        source = self.brain.create('Источник', 'факт', kind='source', review='unreviewed')
        other = self.brain.create('Другая', 'текст', review='reviewed')
        self.brain.close()
        with mock.patch.dict(sys.modules, {'fastmcp': types.SimpleNamespace(FastMCP=FakeMCP)}):
            tools = build(self.brain.vault, self.brain.state).tools
        result = tools['create_draft']('Черновик', f'- contradicts [[{other}]]', [source])
        self.brain = Brain(self.brain.vault, self.brain.state)
        self.brain.sync()
        self.assertEqual(result['proposed_links'], [other])
        self.assertEqual(self.edges(result['id']), {(source, 'relates_to')})


if __name__ == '__main__':
    unittest.main()
