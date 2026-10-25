import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from second_brain import activity
from second_brain.activity import record, since
from second_brain.brain import Brain
from second_brain.cli import main
from second_brain.tools import Tools


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.vault, self.state = root / 'vault', root / 'state'
        brain = Brain(self.vault, self.state)
        self.hub = brain.create('Галлюцинации агентов', 'Без проверки ошибки агента цитируются как факты.')
        self.check = brain.create('Статусы проверки', 'Человек утверждает записи агента.', links=[self.hub])
        brain.sync()
        brain.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_log_is_bounded_redacted_and_cursor_based(self):
        first = since(self.state)
        self.assertEqual((first['cursor'], first['items'], first['initial']), (0, [], True))
        record(self.state, 'claude code!', 'memory.search', [self.hub, 'not-an-id'], 'token=sk-abcdefghijklmn что-то')
        record(self.state, 'codex', 'memory.read', [self.check])
        latest = since(self.state, 0)
        self.assertEqual([i['type'] for i in latest['items']], ['memory.search', 'memory.read'])
        self.assertEqual(latest['items'][0]['actor'], 'claudecode')
        self.assertEqual(latest['items'][0]['ids'], [self.hub])
        self.assertNotIn('sk-abcdefghijklmn', latest['items'][0]['query'])
        self.assertTrue(all(i['live'] for i in latest['items']))
        self.assertEqual(since(self.state, latest['cursor'])['items'], [])
        self.assertFalse(any(i['live'] for i in since(self.state)['items']))
        with self.assertRaises(ValueError):
            record(self.state, 'x', 'memory.delete')
        with mock.patch.object(activity, 'KEEP', 3):
            for _ in range(10):
                record(self.state, 'x', 'memory.read', [self.hub])
        self.assertLessEqual(len(since(self.state, 0)['items']), 4)

    def test_tools_record_every_access(self):
        tools = Tools(self.vault, self.state, 'claude-code')
        found = tools.search_knowledge('как агент выдумывает факты', top_k=3)  # strict BM25 misses, hybrid fallback finds
        self.assertEqual(found['items'][0]['id'], self.hub)
        self.assertEqual(tools.get_document(self.hub)['title'], 'Галлюцинации агентов')
        self.assertEqual(len(tools.get_related(self.check)['relations']), 1)
        draft = tools.create_draft('Вывод агента', 'Нужна проверка человеком.', [self.hub, self.check])
        events = since(self.state, 0)['items']
        self.assertEqual([e['type'] for e in events], ['memory.search', 'memory.read', 'memory.traverse', 'memory.draft'])
        self.assertEqual(events[2]['ids'], [self.check, self.hub])
        self.assertEqual(events[3]['ids'], [draft['id'], self.hub, self.check])
        self.assertTrue(all(e['actor'] == 'claude-code' for e in events))
        brain = Brain(self.vault, self.state)
        try:
            note = brain.get(draft['id'])
            self.assertEqual((note['space'], note['review'], note['meta']['source']), ('agent', 'proposed', 'agent:claude-code'))
        finally:
            brain.close()

    def test_cli_tool_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(['--vault', str(self.vault), '--state', str(self.state), 'tool', 'get_document',
                  '--args', json.dumps({'document_id': self.hub}), '--agent', 'script'])
        self.assertEqual(json.loads(out.getvalue())['id'], self.hub)
        self.assertEqual(since(self.state, 0)['items'][-1]['actor'], 'script')

    @unittest.skipUnless(__import__('importlib').util.find_spec('fastmcp'), 'fastmcp not installed')
    def test_mcp_gateway_exposes_all_tools(self):
        import asyncio
        from second_brain.mcp_server import build
        server = build(self.vault, self.state, 'claude-code')
        names = {t.name for t in asyncio.run(server.list_tools())}
        self.assertEqual(names, {'search_knowledge', 'get_document', 'get_related', 'create_draft',
                                 'report_usage', 'get_skill_pack', 'reflect'})


if __name__ == '__main__':
    unittest.main()
