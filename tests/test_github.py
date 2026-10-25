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


def fake_fetch(url, limit):
    if url == 'https://api.github.com/repos/acme/tools':
        return json.dumps({'default_branch': 'main', 'description': 'Acme tooling'}).encode()
    if url == 'https://api.github.com/repos/acme/tools/tarball/main':
        return archive(ENTRIES)
    raise AssertionError(url)


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
        self.assertEqual(readme['meta']['source'], 'https://github.com/acme/tools/blob/main/README.md')
        self.assertEqual(readme['review'], 'unreviewed')
        graph = self.brain.graph()
        self.assertEqual({(l['source'], l['target']) for l in graph['links']}, {(s, result['project']) for s in result['sources']})
        self.assertTrue(all(n['project_id'] == result['project'] for n in graph['nodes']))
        self.assertEqual(self.brain.list('Описание')['total'], 1)
        again = import_github(self.brain, 'acme/tools', fake_fetch)
        self.assertEqual((again['project'], again['sources']), (result['project'], result['sources']))
        self.assertEqual(self.brain.sync()['notes'], 3)


if __name__ == '__main__':
    unittest.main()
