import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from second_brain.brain import Brain
from second_brain.learning import Learning
from second_brain.mcp_server import build, propose_draft, search
from tests.test_drafts import FakeMCP


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.brain = Brain(self.root/'vault', self.root/'state')
        self.learning = Learning(self.root/'state')

    def tearDown(self):
        self.learning.close()
        self.brain.close()
        self.temp.cleanup()

    def test_same_draft_twice_is_deduplicated(self):
        source = self.brain.create('Источник', 'факт', kind='source', review='reviewed'); self.brain.sync()
        first = propose_draft(self.brain, 'Вывод', 'Кэш  ускоряет\nсборку', [source])
        again = propose_draft(self.brain, 'вывод', 'Кэш ускоряет сборку', [source])
        self.assertFalse(first['dedup'])
        self.assertTrue(again['dedup'])
        self.assertEqual(first['id'], again['id'])
        self.assertEqual(self.brain.sync()['notes'], 2)
        self.assertNotEqual(propose_draft(self.brain, 'Вывод', 'Другое', [source])['id'], first['id'])

    def test_skill_drafts_need_applicability_evidence_and_sources(self):
        source = self.brain.create('Лог', 'тест прошёл', kind='source', review='unreviewed'); self.brain.sync()
        for kwargs in ({'applies_to': 'Fedora, Python 3.14'}, {'evidence': 'make test'},
                       {'applies_to': 'x', 'evidence': 'y', 'ids': []}):
            ids = kwargs.pop('ids', [source])
            with self.assertRaises(ValueError):
                propose_draft(self.brain, 'Навык', 'шаги', ids, memory_type='skill', **kwargs)
        ok = propose_draft(self.brain, 'Навык', 'шаги', [source], memory_type='skill',
                           applies_to='Fedora, Python 3.14', evidence='make test зелёный', agent_model='qwen', tier='weak')
        meta = self.brain.get(ok['id'])['meta']
        self.assertEqual((meta['memory_type'], meta['agent_tier'], meta['agent_model']), ('skill', 'weak', 'qwen'))
        self.assertEqual(ok['evidence_basis'], 'self-referential')
        with self.assertRaises(ValueError):
            propose_draft(self.brain, 'Навык', 'шаги', [source], memory_type='fact')

    def test_search_labels_hypotheses_ranks_them_after_reviewed_and_hides_eval(self):
        draft = self.brain.create('Кэш proposed', 'кэширование сборки', review='proposed')
        fact = self.brain.create('Кэш reviewed', 'кэширование сборки', review='reviewed')
        self.brain.create('Задача оценки', 'кэширование сборки', review='reviewed', source='eval:task-7')
        self.brain.sync()
        result = search(self.brain, self.learning, 'кэширование', run_id='run-1', agent_model='qwen')
        self.assertEqual([n['id'] for n in result['items']], [fact, draft])
        self.assertEqual([n['unverified'] for n in result['items']], [False, True])
        self.assertFalse(result['truncated'])
        self.assertTrue(search(self.brain, self.learning, 'кэширование', top_k=1)['truncated'])

    def test_usage_separates_served_from_helpful(self):
        notes = [self.brain.create(f'Заметка {i}', 'общий термин', review='reviewed') for i in range(3)]
        self.brain.sync()
        for run in ('r1', 'r2', 'r3', 'r4', 'r5'):
            search(self.brain, self.learning, 'термин', run_id=run)
            self.learning.report_usage(run, [notes[0]], 'success')
        self.assertEqual(self.learning.usefulness(notes[0]), {'served_runs': 5, 'helped_runs': 5})
        stale = {r['note_id'] for r in self.learning.stale_candidates(min_served=5)}
        self.assertEqual(stale, {notes[1], notes[2]})
        self.assertEqual(search(self.brain, self.learning, 'термин')['items'][0]['id'] in notes, True)
        self.assertEqual(self.learning.db.execute("SELECT count(*) FROM served WHERE run_id=''").fetchone()[0], 0)
        with self.assertRaises(ValueError):
            self.learning.report_usage('r9', [], 'great')
        # Forged or repeated reports change nothing.
        self.assertFalse(self.learning.report_usage('r1', [notes[1]], 'success')['recorded'])
        self.assertEqual(self.learning.report_usage('invented', [notes[1]], 'success')['recorded'], False)
        search(self.brain, self.learning, 'термин', run_id='r6', agent_model='qwen')
        self.assertEqual(self.learning.report_usage('r6', [notes[1]], 'success', model='other')['recorded'], False)
        self.assertEqual(self.learning.usefulness(notes[1])['helped_runs'], 0)

    def test_server_takes_tier_from_host_file_and_writes_as_agent(self):
        (self.root/'state'/'agents.json').write_text(json.dumps({'models': {'qwen3.8-flash': 'weak'}}))
        source = self.brain.create('Лог', 'факт', kind='source', review='reviewed')
        self.brain.close()
        with mock.patch.dict(sys.modules, {'fastmcp': types.SimpleNamespace(FastMCP=FakeMCP)}):
            tools = build(self.brain.vault, self.brain.state, 'qwen3.8-flash').tools
        result = tools['create_draft']('Навык', 'шаги', [source], 'skill', 'везде', 'тест')
        self.brain = Brain(self.root/'vault', self.root/'state'); self.brain.sync()
        self.assertEqual(self.brain.get(result['id'])['meta']['agent_tier'], 'weak')
        actors = {e['actor'] for e in self.brain.ledger.entries() if e['phase'] == 'intent'}
        self.assertIn('agent:qwen3.8-flash', actors)


if __name__ == '__main__':
    unittest.main()
