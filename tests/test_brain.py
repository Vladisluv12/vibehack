import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from second_brain.brain import Brain, digest, restore_backup


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.brain = Brain(self.root/'vault', self.root/'state')

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_provenance_dedup_graph_search_external_edit(self):
        project = self.brain.create('Проект Steam', 'Трекер цен', kind='project', key='project')
        source = self.root/'research.md'
        source.write_text('# Документ\n## Кэш\nКэширование результатов\n', encoding='utf-8')
        result = self.brain.import_markdown(source, split=True, links=[project])
        self.assertEqual(result, self.brain.import_markdown(source, split=True, links=[project]))
        self.assertEqual(self.brain.sync()['notes'], 3)
        self.assertEqual(self.brain.list('Кэширование')['total'], 2)
        self.assertEqual(len(self.brain.graph()['links']), 3)
        note = self.brain.get(result['id'])
        self.assertEqual(note['meta']['source_hash'], digest(source.read_bytes()))
        self.assertEqual(note['review'], 'unreviewed')
        p = self.brain.vault/note['path'];p.write_text(p.read_text(encoding='utf-8')+'\nНовая мысль', encoding='utf-8')
        self.brain.sync()
        self.assertEqual(self.brain.list('мысль')['total'], 1)

    def test_edit_conflict_secrets_and_path_safety(self):
        ident = self.brain.create('Заметка', 'original');self.brain.sync()
        note = self.brain.get(ident)
        (self.brain.vault/note['path']).write_text(note['content']+'external', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Conflict'):
            self.brain.edit(ident, note['content']+'edit', note['hash'])
        with self.assertRaises(ValueError):
            self.brain.edit(ident, note['content']+' API_KEY=sk-abcdefghijkl', note['hash'])
        for path in ('../outside.md', '/tmp/outside.md', '.hidden/n.md'):
            with self.assertRaises(ValueError):self.brain.safe(path)
        for path in ('C:/outside.md', 'C:outside.md', r'\\server\share\outside.md'):
            if os.name == 'nt':
                with self.assertRaises(ValueError):self.brain.safe(path)
        try:
            (self.brain.vault/'escape').symlink_to(self.root, target_is_directory=True)
        except OSError:  # Windows without Developer Mode cannot create symlinks
            return
        with self.assertRaises(ValueError):self.brain.safe('escape/outside.md')

    def test_backup_roundtrip_and_traversal(self):
        self.brain.create('Локальная память', 'Важный вывод')
        backup = self.brain.backup()
        result = restore_backup(backup['path'], self.root/'restored')
        self.assertEqual(result['notes'],1)
        a = next(self.brain.vault.rglob('*.md')).read_bytes()
        self.assertEqual(a,next((self.root/'restored').rglob('*.md')).read_bytes())
        with self.assertRaises(ValueError):restore_backup(backup['path'],self.root/'restored')
        bad=self.root/'bad.zip'
        with zipfile.ZipFile(bad,'w') as z:
            z.writestr('../bad.md','bad');z.writestr('manifest.json',json.dumps({'../bad.md':digest('bad')}))
        with self.assertRaises(ValueError):restore_backup(bad,self.root/'unsafe')
        self.assertFalse((self.root/'unsafe').exists())

    def test_pagination_ambiguous_link_and_duplicate_ids(self):
        for i in range(3):self.brain.create('Same',str(i))
        self.brain.create('Link','- relates_to [[Same]]')
        report=self.brain.sync();self.assertEqual(report['warning_count'],1)
        self.assertEqual(self.brain.list(limit=2)['next'],2)
        self.assertEqual(len(self.brain.list(offset=2,limit=2)['items']),2)
        p=next(self.brain.vault.rglob('*.md'))
        (self.brain.vault/'duplicate.md').write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.brain.sync()

    def test_rename_reimport_preserves_id_and_user_edits(self):
        ident=self.brain.create('Stable','before',key='stable');self.brain.sync()
        old=self.brain.vault/self.brain.get(ident)['path']
        moved=self.brain.vault/'Renamed.md';old.rename(moved)
        moved.write_text(moved.read_text(encoding='utf-8')+'\nUser edit', encoding='utf-8')
        self.assertEqual(self.brain.create('Stable','before',key='stable'),ident)
        self.assertEqual(self.brain.sync()['notes'],1)
        self.assertIn('User edit',self.brain.get(ident)['body'])

    def test_invalid_scalar_metadata_and_nested_fences(self):
        (self.brain.vault/'external.md').write_text('---\ntype: []\nreview: {}\n---\nBody', encoding='utf-8')
        victim=self.brain.create('Victim','real')
        self.brain.create('Transcript','Example\n```\n[[Victim]]',kind='session',key='fence')
        self.assertEqual(self.brain.sync()['notes'],3)
        self.assertEqual(self.brain.graph()['links'],[])

    def test_byte_exact_crlf_hash_backup_and_versions(self):
        source=self.root/'crlf.md';source.write_bytes(b'# Source\r\n\r\nOriginal\r\n')
        ident=self.brain.import_markdown(source)['id'];self.brain.sync()
        self.assertEqual(self.brain.get(ident)['meta']['raw_sha256'],digest(source.read_bytes()))
        note=self.brain.get(ident)
        self.brain.edit(ident,note['content']+'\nEdit',note['hash'])
        version=next((self.brain.vault/'Versions').rglob('*.md'))
        self.assertEqual(version.read_bytes(),note['content'].encode())
        self.assertEqual(self.brain.sync()['notes'],1)
        backup=self.brain.backup();restore_backup(backup['path'],self.root/'restored')
        for p in self.brain.vault.rglob('*.md'):
            self.assertEqual(p.read_bytes(),(self.root/'restored'/p.relative_to(self.brain.vault)).read_bytes())

    def test_spaces_split_human_and_agent_notes(self):
        human = self.brain.create('Решение команды', 'Подробное обоснование')
        draft = self.brain.create('Черновик агента', 'Кратко', source='agent:codex', links=[human])
        tagged = self.brain.create('Явно агент', 'x', extra={'space': 'agent'})
        self.brain.sync()
        self.assertEqual(self.brain.graph()['spaces'], {'agent': 2, 'human': 1})
        self.assertEqual({n['id'] for n in self.brain.list(space='agent')['items']}, {draft, tagged})
        self.assertEqual([n['space'] for n in self.brain.list('Решение')['items']], ['human'])
        self.assertEqual(self.brain.get(draft)['space'], 'agent')
        self.assertEqual(self.brain.graph(space='human')['links'], [])
        self.assertEqual(len(self.brain.graph()['links']), 1)
        self.assertEqual({n['id'] for n in self.brain.list(review='proposed', space='agent')['items']}, {draft, tagged})
        self.assertEqual(self.brain.list('Черновик', review='reviewed')['total'], 0)
        with self.assertRaises(ValueError):
            self.brain.list(space='robots')
        with self.assertRaises(ValueError):
            self.brain.list(review='maybe')

    def test_tags_from_frontmatter_yaml_lists_and_inline(self):
        flow = self.brain.create('Поток', 'Текст #Идея и `#код`', extra={'tags': ['поиск', 'BM25']})
        (self.brain.vault / 'obsidian.md').write_text(
            '---\ntitle: "Obsidian"\ntags:\n  - поиск\n  - project/alpha\n---\nСм. [x](#anchor), #123 и #идея\n```\n#fence\n```\n',
            encoding='utf-8')
        source = self.root / 'ext.md'
        source.write_text('# Внешний\n#чужой-тег', encoding='utf-8')
        imported = self.brain.import_markdown(source)['id']
        self.brain.sync()
        self.assertEqual(self.brain.get(flow)['tags'], ['bm25', 'идея', 'поиск'])
        self.assertEqual(self.brain.get(imported)['tags'], [])
        self.assertEqual(self.brain.list(tag='#Поиск')['total'], 2)
        self.assertEqual({n['title'] for n in self.brain.list(tag='project/alpha')['items']}, {'Obsidian'})
        self.assertEqual(self.brain.tags()['tags'][:2], [{'tag': 'идея', 'count': 2}, {'tag': 'поиск', 'count': 2}])
        self.assertNotIn('fence', {t['tag'] for t in self.brain.tags()['tags']})
        with self.assertRaises(ValueError):
            self.brain.list(tag='bad tag')

    def test_bm25_title_ranking(self):
        best=self.brain.create('Qdrant retrieval','Exact match')
        self.brain.create('AAA other','many unrelated words '*150+' Qdrant retrieval')
        self.brain.sync()
        self.assertEqual(self.brain.list('Qdrant retrieval')['items'][0]['id'],best)
        self.assertIn('score',self.brain.list('Qdrant')['items'][0])

    def test_project_owners_use_only_explicit_links_or_metadata(self):
        a=self.brain.create('A','',kind='project',source='file:///work/a')
        b=self.brain.create('B','',kind='project')
        direct=self.brain.create('Direct','',links=[a])
        both=self.brain.create('Both','',links=[a,b])
        meta=self.brain.create('Meta','',extra={'project':'/work/a'})
        self.brain.create('Loose','mentions A')
        self.brain.sync()
        nodes={n['title']:n for n in self.brain.graph()['nodes']}
        self.assertEqual(nodes['Direct']['project_id'],a)
        self.assertEqual(nodes['Meta']['project_id'],a)
        self.assertIsNone(nodes['Both']['project_id'])
        self.assertIsNone(nodes['Loose']['project_id'])
        self.assertEqual(nodes['A']['project_materials'],3)


if __name__ == '__main__':unittest.main()
