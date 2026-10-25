import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from second_brain.cli import main
from second_brain.web import make_server


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

    def test_server_requires_token_and_serves_brain_only(self):
        self.cli('demo')
        server = make_server(self.vault, self.state, 0)
        port = server.server_port
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            base = f'http://127.0.0.1:{port}'
            page = urllib.request.urlopen(base+'/').read().decode()
            self.assertNotIn('/queue', page)
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
            self.assertEqual(call('/api/activity')['items'], [])
            created = call('/api/brain/create', {'title': 'Новая', 'body': 'текст', 'review': 'reviewed'})
            self.assertEqual(call('/api/brain/note?id='+created['id'])['review'], 'proposed')
            with self.assertRaises(urllib.error.HTTPError) as missing:
                call('/api/tasks', {})
            self.assertEqual(missing.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
