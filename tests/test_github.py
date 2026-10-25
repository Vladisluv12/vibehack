import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from urllib.request import Request

from second_brain.brain import MAX_NOTE, Brain
from second_brain.github import _GitHubOnly, import_github, markdown_members, parse_repo


def archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type, info.linkname = tarfile.SYMTYPE, '../../etc/passwd.md'
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


ROOT = 'acme-tools-1a2b3c/'
ENTRIES = [
    (ROOT + 'docs/setup.md', '# Setup\nRun it. [[Victim]]\n'.encode()),
    (ROOT + 'README.md', '# Tools\n\nОписание.\n'.encode()),
    (ROOT + '.github/ISSUE.md', b'hidden'),
    (ROOT + 'node_modules/pkg/README.md', b'vendored'),
    (ROOT + 'a/../../escape.md', b'traversal'),
    (ROOT + 'link.md', None),
    (ROOT + 'big.md', b'x' * (MAX_NOTE + 1)),
    (ROOT + 'latin1.md', 'caf\xe9'.encode('latin-1')),
    (ROOT + 'logo.png', b'\x89PNG'),
]


SHA1, SHA2 = 'a' * 40, 'b' * 40


def github(commit, entries):
    def fetcher(url, limit):
        if url == 'https://api.github.com/repos/acme/tools':
            return json.dumps({'default_branch': 'main', 'description': 'Acme tooling'}).encode()
        if url == 'https://api.github.com/repos/acme/tools/branches/main':
            return json.dumps({'name': 'main', 'commit': {'sha': commit}}).encode()
        if url == f'https://api.github.com/repos/acme/tools/tarball/{commit}':
            return archive(entries)
        raise AssertionError(url)
    return fetcher


fake_fetch = github(SHA1, ENTRIES)


class GitHubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.brain = Brain(root / 'vault', root / 'state')

    def tearDown(self):
        self.brain.close()
        self.temp.cleanup()

    def test_parse_repo_accepts_github_forms_only(self):
        for url in ('https://github.com/acme/tools', 'github.com/acme/tools.git', 'http://www.github.com/acme/tools/', 'acme/tools'):
            self.assertEqual(parse_repo(url), ('acme', 'tools'))
        for url in ('https://github.com.evil.test/acme/tools', 'https://evil.test/acme/tools', 'acme/..', 'acme/tools/extra',
                    'https://github.com/-acme/tools', 'file:///etc/passwd', '', None):
            with self.assertRaises(ValueError):
                parse_repo(url)

    def test_redirects_stay_on_github(self):
        handler, request = _GitHubOnly(), Request('https://api.github.com/repos/acme/tools/tarball/main')
        self.assertIsNotNone(handler.redirect_request(request, None, 302, 'Found', {}, 'https://codeload.github.com/acme/tools/x'))
        for target in ('https://evil.test/x', 'http://codeload.github.com/x'):
            with self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, 'Found', {}, target)

    def test_members_are_bounded_regular_markdown(self):
        files, skipped, omitted = markdown_members(archive(ENTRIES))
        self.assertEqual([f[0] for f in files], ['README.md', 'docs/setup.md'])
        self.assertEqual((skipped, omitted), (2, 0))

    def test_import_creates_project_cluster_of_unreviewed_sources(self):
        result = import_github(self.brain, 'https://github.com/acme/tools', fake_fetch)
        self.assertEqual((result['imported'], result['branch']), (2, 'main'))
        self.brain.sync()
        project = self.brain.get(result['project'])
        self.assertEqual((project['title'], project['kind'], project['review']), ('acme/tools', 'project', 'unreviewed'))
        readme = self.brain.get(result['sources'][0])
        self.assertEqual(readme['title'], 'tools/README.md')
        self.assertEqual(readme['meta']['source'], f'https://github.com/acme/tools/blob/{SHA1}/README.md')
        self.assertEqual(readme['meta']['commit'], SHA1)
        self.assertEqual(readme['review'], 'unreviewed')
        graph = self.brain.graph()
        run_edges = {(result['run'], t) for t in [result['project'], *result['sources']]}
        self.assertEqual({(l['source'], l['target']) for l in graph['links']},
                         {(s, result['project']) for s in result['sources']} | run_edges)
        self.assertTrue(all(n['project_id'] == result['project'] for n in graph['nodes']))
        self.assertEqual(self.brain.list('Описание')['total'], 1)
        again = import_github(self.brain, 'acme/tools', fake_fetch)
        self.assertEqual((again['project'], again['sources'], again['run']), (result['project'], result['sources'], result['run']))
        self.assertEqual(self.brain.sync()['notes'], 4)

    def test_new_commit_versions_changed_files_and_keeps_old_ones(self):
        first = import_github(self.brain, 'acme/tools', fake_fetch)
        changed = [(ROOT + 'docs/setup.md', b'# Setup\nRun it twice.\n'), (ROOT + 'README.md', ENTRIES[1][1])]
        second = import_github(self.brain, 'acme/tools', github(SHA2, changed))
        self.brain.sync()
        self.assertEqual(second['commit'], SHA2)
        self.assertEqual(second['changed'], 1)
        self.assertEqual(second['project'], first['project'])
        self.assertNotEqual(second['run'], first['run'])
        old_readme, old_setup = first['sources']
        new_readme, new_setup = second['sources']
        self.assertEqual(new_readme, old_readme)
        self.assertNotEqual(new_setup, old_setup)
        setup = self.brain.get(new_setup)
        self.assertEqual(setup['meta']['supersedes'], old_setup)
        self.assertEqual(setup['meta']['source'], f'https://github.com/acme/tools/blob/{SHA2}/docs/setup.md')
        self.assertEqual(self.brain.get(old_setup)['meta']['commit'], SHA1)
        self.assertIn('Run it twice', setup['content'])
        self.assertEqual(self.brain.sync()['notes'], 6)
        third = import_github(self.brain, 'acme/tools', github('c' * 40, [(ROOT + 'docs/setup.md', b'# v3\n')]))
        self.brain.sync()
        self.assertEqual(self.brain.get(third['sources'][0])['meta']['supersedes'], new_setup)

    def test_branch_metadata_must_name_a_commit(self):
        def bad(url, limit):
            if url.endswith('/branches/main'):
                return json.dumps({'commit': {'sha': 'main'}}).encode()
            return fake_fetch(url, limit)
        with self.assertRaises(ValueError):
            import_github(self.brain, 'acme/tools', bad)


if __name__ == '__main__':
    unittest.main()
