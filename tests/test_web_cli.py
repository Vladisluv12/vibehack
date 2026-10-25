import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request

from second_brain.cli import main
from second_brain.web import Extension, make_server
from test_github import fake_fetch


class WebCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vault, self.state = self.root/'vault', self.root/'state'

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(['--vault', str(self.vault), '--state', str(self.state), *args])
        return json.loads(out.getvalue())

    def test_demo_vault_syncs_without_warnings_and_is_searchable(self):
        report = self.cli('demo')
        self.assertEqual(report['notes'], 8)
        self.assertEqual(report['warning_count'], 0)
        self.assertGreater(report['links'], 5)
        self.assertEqual(self.cli('search', 'proposed')['total'], 2)
        self.assertEqual(self.cli('demo')['notes'], 8)

    def test_showcase_vault_is_dense_and_clean(self):
        report = self.cli('demo', '--showcase')
        self.assertGreaterEqual(report['notes'], 60)
        self.assertGreater(report['links'], report['notes'])
        self.assertEqual(report['warning_count'], 0)

    def test_server_requires_token_and_serves_brain_only(self):
        self.cli('demo')
        server = make_server(self.vault, self.state, 0)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f'http://127.0.0.1:{port}'
            page = urllib.request.urlopen(base+'/').read().decode()
            self.assertIn('id="queue-link"', page)
            token = re.search(r'name="control-token" content="([^"]+)"', page)[1]

            def call(path, body=None, headers=None):
                request = urllib.request.Request(base+path, data=json.dumps(body).encode() if body is not None else None,
                                                 headers={'X-Control-Token': token, **(headers or {})})
                return json.loads(urllib.request.urlopen(request).read())

            with self.assertRaises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(base+'/api/brain/notes')
            self.assertEqual(denied.exception.code, 403)
            with self.assertRaises(urllib.error.HTTPError) as foreign:
                call('/api/brain/notes', headers={'Origin': 'http://evil.test'})
            self.assertEqual(foreign.exception.code, 403)
            self.assertEqual(call('/api/brain/notes')['total'], 8)
            self.assertEqual(call('/api/extension'), {'connected': False, 'name': ''})
            with self.assertRaises(urllib.error.HTTPError) as absent:
                call('/api/activity')
            self.assertEqual(absent.exception.code, 404)
            created = call('/api/brain/create', {'title': 'Новая', 'body': 'текст', 'review': 'reviewed'})
            self.assertEqual(call('/api/brain/note?id='+created['id'])['review'], 'proposed')
            with mock.patch('second_brain.github.fetch', fake_fetch):
                imported = call('/api/brain/import-github', {'url': 'https://github.com/acme/tools'})
            self.assertEqual(imported['imported'], 2)
            self.assertEqual(call('/api/brain/note?id='+imported['project'])['title'], 'acme/tools')
            with self.assertRaises(urllib.error.HTTPError) as bad_repo:
                call('/api/brain/import-github', {'url': 'https://evil.test/acme/tools'})
            self.assertEqual(bad_repo.exception.code, 400)
            bad_repo.exception.close()
            with self.assertRaises(urllib.error.HTTPError) as missing:
                call('/api/tasks', {})
            self.assertEqual(missing.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()


class FakeAgents(Extension):
    name = 'Agent Control'
    pages = {'/queue': '<meta name="control-token" content="__CONTROL_TOKEN__">queue'}

    def info(self):
        return {'pages': {'queue': '/queue'}}

    def get(self, path, params, open_brain):
        if path == '/api/activity':
            brain = open_brain()
            try:
                return {'items': [], 'notes': brain.list()['total']}
            finally:
                brain.close()
        if path == '/api/boom':
            raise ValueError('token=sk-abcdefghijkl leaked')

    def post(self, path, data, open_brain):
        if path == '/api/tasks':
            return 201, {'echo': data}


class ExtensionTests(unittest.TestCase):
    def test_extension_routes_pages_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = make_server(root/'vault', root/'state', 0, extension=FakeAgents())
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                queue = urllib.request.urlopen(base+'/queue').read().decode()
                token = re.search(r'content="([^"]+)"', queue)[1]
                self.assertNotIn('__CONTROL_TOKEN__', queue)

                def call(path, body=None):
                    request = urllib.request.Request(base+path, data=json.dumps(body).encode() if body is not None else None,
                                                     headers={'X-Control-Token': token})
                    with urllib.request.urlopen(request) as response:
                        return response.status, json.loads(response.read())

                self.assertEqual(call('/api/extension')[1], {'connected': True, 'name': 'Agent Control', 'pages': {'queue': '/queue'}})
                self.assertEqual(call('/api/activity')[1], {'items': [], 'notes': 0})
                self.assertEqual(call('/api/tasks', {'x': 1}), (201, {'echo': {'x': 1}}))
                for path, code in (('/api/boom', 400), ('/api/unknown', 404)):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        call(path)
                    self.assertEqual(error.exception.code, code)
                    self.assertNotIn('sk-abcdefghijkl', error.exception.read().decode())
                    error.exception.close()
            finally:
                server.shutdown()
                server.server_close()


if __name__ == '__main__':
    unittest.main()
