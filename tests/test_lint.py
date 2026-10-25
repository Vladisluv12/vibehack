import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError

from second_brain.brain import Brain
from second_brain.lint import exit_code, lint


class LintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.brain = Brain(root / 'vault', root / 'state')
        self.addCleanup(self.brain.close)

    def note(self, number, body='', **metadata):
        ident = f'{number:032x}'
        meta = dict(id=ident, title=f'Note {number}', type='note', permalink=f'note/{ident}',
                    review='unreviewed', source='', source_hash='')
        meta.update(metadata)
        path = f'note-{number:04}.md'
        text = '---\n' + ''.join(f'{k}: {json.dumps(v)}\n' for k, v in meta.items()) + '---\n' + body
        (self.brain.vault / path).write_text(text, encoding='utf-8')
        return ident

    def test_each_error_and_warning_has_source_and_stable_code(self):
        self.note(1, title='Duplicate')
        self.note(2, title='dUPLICATE')
        bad = self.note(3, '[[missing]] [[Duplicate]]', review='proposed', source='  ', source_hash='ABC',
                        supersedes='f' * 32, memory_type='skill', proposed_links=['nowhere'])
        self.note(4, '[[ignored external link]]', declared_only=True, external_content=False,
                  link_policy='declared', declared_links=['e' * 32])
        self.brain.sync()
        report = lint(self.brain)
        self.assertEqual({e['code'] for e in report['errors']}, {
            'unresolved_link', 'ambiguous_link', 'unresolved_declared_link', 'missing_supersedes',
            'invalid_source_hash', 'missing_applies_to', 'missing_evidence', 'missing_declared_links'})
        self.assertEqual({w['code'] for w in report['warnings']},
                         {'duplicate_title', 'unresolved_proposed_link', 'proposed_without_source'})
        self.assertTrue(any(e['note_id'] == bad and e['path'] == 'note-0003.md' for e in report['errors']))
        for group in ('errors', 'warnings', 'candidates'):
            self.assertEqual(report['counts'][group], len(report[group]))
            self.assertTrue(all(set(item) == {'code', 'note_id', 'path', 'detail'} for item in report[group]))
        self.assertEqual(exit_code(report), 1)
        self.assertEqual(report, lint(self.brain))

    def test_orphans_are_candidates_and_links_match_projection_policy(self):
        target = self.note(1, title='Target', source_hash='a' * 64)
        self.note(2, '[[TARGET#heading|label]] [[note-0001.md]]')
        orphan = self.note(3, '```\n[[missing]]\n```\n~~~\n[[also missing]]\n~~~')
        self.note(4, type='project')
        self.note(5, type='session')
        self.note(6, '[[untrusted]]', external_content=True, declared_only=False,
                  link_policy='declared', declared_links=[target], supersedes=target)
        self.note(7, '[[suggestion]]', declared_only=True, external_content=False,
                  link_policy='declared', declared_links=[target], memory_type='experience',
                  applies_to='tests', evidence='oracle passed', proposed_links=['Target'])
        self.brain.sync()
        report = lint(self.brain)
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['warnings'], [])
        self.assertEqual([n['note_id'] for n in report['candidates']], [orphan])
        self.assertEqual(report['counts']['links'], 3)
        self.assertEqual(exit_code(report), 0)

    def test_full_index_exceeds_both_capped_views(self):
        for number in range(1, 522):
            self.note(number, f'[[missing-{number}]]')
        target = self.note(522)
        self.note(523, f'[[{target}]]')
        self.assertEqual(len(self.brain.sync()['warnings']), 100)
        self.assertEqual(len(self.brain.graph()['nodes']), 500)
        with mock.patch.object(self.brain, 'sync', side_effect=AssertionError('capped sync report')), \
                mock.patch.object(self.brain, 'graph', side_effect=AssertionError('capped graph')):
            report = lint(self.brain)
        self.assertEqual(report['counts']['notes'], 523)
        self.assertEqual(report['counts']['links'], 1)
        self.assertEqual(len(report['errors']), 521)
        self.assertEqual(len(report['candidates']), 521)

    def test_duplicate_ids_and_unsynced_changes_are_reported_without_mutation(self):
        target = self.note(1)
        self.note(2, f'[[{target}]]')
        self.brain.sync()
        self.note(3, id=target)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.brain.sync()
        report = lint(self.brain)
        self.assertEqual(len([e for e in report['errors'] if e['code'] == 'duplicate_id']), 2)
        self.assertEqual(report['counts']['notes'], 3)
        self.assertEqual(report['counts']['links'], 1)
        self.assertEqual(self.brain.db.execute('SELECT count(*) FROM notes').fetchone()[0], 2)
        (self.brain.vault / 'note-0003.md').unlink()
        self.note(2, '[[new missing link]]')
        report = lint(self.brain)
        self.assertEqual([e['code'] for e in report['errors']], ['unresolved_link'])
        self.assertEqual(len(report['candidates']), 2)

    def test_source_hash_requires_exact_lowercase_hex(self):
        for number, value in enumerate(['', '0' * 64, 'a' * 64, 'A' * 64, 'g' * 64, 'a' * 63, ' ', 42], 1):
            self.note(number, source_hash=value)
        self.assertEqual([e['note_id'] for e in lint(self.brain)['errors']], [f'{n:032x}' for n in range(4, 9)])

    def test_network_is_opt_in_and_fetch_is_injected_and_deduplicated(self):
        for number, source in enumerate(['https://good.test', 'https://gone.test', 'https://bad.test',
                                         'https://down.test', 'file:///missing', 'eval:case', 'https://gone.test'], 1):
            self.note(number, source=source)

        def fetch(url):
            if url == 'https://down.test':
                raise OSError('unreachable')
            return {'https://good.test': 200, 'https://gone.test': 404, 'https://bad.test': 503}[url]

        fetcher = mock.Mock(side_effect=fetch)
        with mock.patch('second_brain.lint.urlopen', side_effect=AssertionError('no network')):
            self.assertEqual(lint(self.brain, fetch=fetcher)['warnings'], [])
            fetcher.assert_not_called()
            report = lint(self.brain, online=True, fetch=fetcher)
        self.assertEqual(fetcher.call_count, 4)
        self.assertEqual([w['code'] for w in report['warnings']], [
            'source_http_error', 'source_http_error', 'source_unreachable', 'source_http_error'])
        self.assertEqual(exit_code(report), 0)

    def test_default_fetch_uses_timeout_and_get_if_head_is_unsupported(self):
        self.note(1, source='https://example.test')
        response = mock.MagicMock()
        response.__enter__.return_value.status = 204
        error = HTTPError('https://example.test', 405, 'HEAD unsupported', {}, None)
        with mock.patch('second_brain.lint.urlopen', side_effect=[error, response]) as opener:
            self.assertEqual(lint(self.brain, online=True)['warnings'], [])
        self.assertEqual([call.args[0].get_method() for call in opener.call_args_list], ['HEAD', 'GET'])
        self.assertTrue(all(call.kwargs['timeout'] == 10 for call in opener.call_args_list))


if __name__ == '__main__':
    unittest.main()
