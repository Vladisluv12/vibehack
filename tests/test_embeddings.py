from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from second_brain.embeddings import endpoint_embed, load_settings, public_settings, update_settings


class FakeEndpoint(BaseHTTPRequestHandler):
    """OpenAI, Ollama and TEI shapes; vectors encode text length so order is checkable."""
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeEndpoint.seen.append((self.path, self.headers.get('Authorization'), body))
        texts = body.get('input') or body.get('inputs')
        vectors = [[float(len(t)), 1.0] for t in texts]
        if self.path == '/v1/embeddings':
            payload = {'data': [{'index': i, 'embedding': v} for i, v in reversed(list(enumerate(vectors)))]}
        elif self.path == '/api/embed':
            payload = {'embeddings': vectors}
        elif self.path == '/embed':
            payload = vectors
        elif self.path == '/bad/embeddings':
            payload = {'data': [{'index': 0, 'embedding': ['x']}]}
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeEndpoint)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_settings_validate_persist_and_hide_the_key(self):
        self.assertEqual(load_settings(self.state)['embeddings']['provider'], 'local')
        self.assertEqual(load_settings(self.state)['semantic']['threshold'], 'auto')
        saved = update_settings(self.state, {'embeddings': {'provider': 'openai', 'url': self.base + '/v1', 'model': 'm', 'api_key': 'sk-secret-value'}})
        self.assertEqual(saved['embeddings']['api_key'], 'sk-secret-value')
        shown = public_settings(load_settings(self.state))
        self.assertNotIn('sk-secret-value', json.dumps(shown))
        self.assertTrue(shown['embeddings']['api_key_set'])
        update_settings(self.state, {'embeddings': {'model': 'other'}, 'semantic': {'threshold': 0.7}})
        self.assertEqual(load_settings(self.state)['embeddings']['api_key'], 'sk-secret-value')
        update_settings(self.state, {'embeddings': {'api_key': ''}})
        self.assertFalse(public_settings(load_settings(self.state))['embeddings']['api_key_set'])
        for bad in ({'embeddings': {'provider': 'magic'}}, {'embeddings': {'url': 'file:///etc/passwd'}},
                    {'embeddings': {'model': ''}}, {'semantic': {'threshold': 2}}, {'semantic': {'spaces': []}},
                    {'semantic': {'top_k': 0}}, {'embeddings': {'api_key': 'a\nb'}}):
            with self.assertRaises(ValueError):
                update_settings(self.state, bad)
        self.assertEqual(load_settings(self.state)['embeddings']['model'], 'other')

    def test_endpoint_shapes(self):
        texts = ['a', 'bbb']
        common = {'model': 'm', 'api_key': 'k'}
        for provider, url in (('openai', self.base + '/v1'), ('ollama', self.base), ('tei', self.base)):
            self.assertEqual(endpoint_embed(texts, {**common, 'provider': provider, 'url': url}), [[1.0, 1.0], [3.0, 1.0]])
        self.assertEqual(FakeEndpoint.seen[-3][1], 'Bearer k')
        self.assertEqual(FakeEndpoint.seen[-3][2], {'model': 'm', 'input': texts, 'encoding_format': 'float'})
        for url in (self.base + '/bad', self.base + '/missing'):
            with self.assertRaises(ValueError):
                endpoint_embed(texts, {**common, 'provider': 'openai', 'url': url})
        with self.assertRaises(ValueError):
            endpoint_embed(texts, {**common, 'provider': 'openai', 'url': 'http://127.0.0.1:9/v1'})


if __name__ == '__main__':
    unittest.main()
