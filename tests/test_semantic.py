from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest

from second_brain.brain import Brain
from second_brain.embeddings import load_settings, update_settings
from second_brain.semantic import Semantic, chunk_note, clean_body, mutual_neighbours, percentile


def bag_of_words(text, dims=64):
    """Deterministic stand-in for a model: similar wording → similar vectors."""
    vector = [0.0] * dims
    for word in re.findall(r'\w+', text.lower()):
        vector[int(hashlib.md5(word[:5].encode()).hexdigest(), 16) % dims] += 1.0
    return vector


class BagEndpoint(BaseHTTPRequestHandler):
    requests = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        BagEndpoint.requests += 1
        texts = json.loads(self.rfile.read(int(self.headers['Content-Length'])))['input']
        data = json.dumps({'data': [{'index': i, 'embedding': bag_of_words(t)} for i, t in enumerate(texts)]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


NOTES = [
    ('Кэширование результатов', 'Кэширование результатов поиска ускоряет повторные запросы к индексу. ' * 3),
    ('Кэш поисковой выдачи', 'Кэш поисковой выдачи хранит результаты запросов к индексу, чтобы не считать заново. ' * 3),
    ('Рецепт борща', 'Свёкла, капуста, картофель и говядина варятся два часа на медленном огне. ' * 3),
    ('Отпуск на море', 'Билеты на поезд, гостиница у пляжа и экскурсии по побережью летом. ' * 3),
]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.state = root / 'state'
        self.brain = Brain(root / 'vault', self.state)
        self.ids = [self.brain.create(title, body) for title, body in NOTES]
        self.agent = self.brain.create('Черновик агента про кэш', NOTES[0][1], source='agent:codex')
        self.brain.sync()
        self.semantic = Semantic(self.brain, self.state)

    def tearDown(self):
        self.semantic.close()
        self.brain.close()
        self.temp.cleanup()

    def pairs(self, result):
        return {frozenset((l['source'], l['target'])) for l in result['links']}

    def test_text_cleaning_and_chunks(self):
        body = 'См. [[abc]] и [[def|своё имя]].\n\n## Relations\n- relates_to [[abc]]\n'
        self.assertEqual(clean_body(body, {'abc': 'Кэш'}), 'См. Кэш и своё имя.')
        self.assertEqual(chunk_note('Т', 'коротко'), ['Т\n\nкоротко'])
        long = '# Раздел\n' + ('слово ' * 400) + '\n## Подраздел\n' + ('другое ' * 50)
        chunks = chunk_note('Заметка', long)
        self.assertTrue(chunks[0].startswith('Заметка > Раздел\n\n'))
        self.assertTrue(chunks[-1].startswith('Заметка > Раздел > Подраздел\n\n'))
        self.assertTrue(all(len(c) < 1100 for c in chunks))

    def test_mutual_neighbours_and_percentile(self):
        # 0 is a hub: everyone's nearest, but 0's own top-1 is only 1.
        scores = {0: {1: .9, 2: .8, 3: .7}, 1: {0: .9, 2: .1}, 2: {0: .8, 1: .1, 3: .6}, 3: {0: .7, 2: .6}}
        self.assertEqual(mutual_neighbours(scores, 4, 1, 5, 0.0, set()), {(0, 1): .9})
        self.assertEqual(mutual_neighbours(scores, 4, 1, 5, 0.0, {(0, 1)}), {(0, 2): .8})
        self.assertEqual(percentile([0.9], 100), 0.0)
        self.assertEqual(percentile([0.9, 0.8, 0.7], 3, q=0.5), 0.8)

    def test_local_fallback_suggests_similar_human_notes_only(self):
        settings = load_settings(self.state)
        result = self.semantic.suggestions(settings)
        self.assertIn(frozenset(self.ids[:2]), self.pairs(result))
        self.assertTrue(all(self.agent not in pair for pair in self.pairs(result)))
        self.assertFalse(result['cached'])
        self.assertTrue(self.semantic.suggestions(settings)['cached'])
        self.semantic.dismiss(self.ids[1], self.ids[0])
        self.assertNotIn(frozenset(self.ids[:2]), self.pairs(self.semantic.suggestions(settings)))

    def test_accepting_a_link_removes_the_suggestion(self):
        note = self.brain.add_link(self.ids[0], self.ids[1])
        self.assertIn(f'- relates_to [[{self.ids[1]}]]', note['content'])
        self.assertEqual(self.brain.add_link(self.ids[0], self.ids[1])['hash'], note['hash'])
        self.assertNotIn(frozenset(self.ids[:2]), self.pairs(self.semantic.suggestions(load_settings(self.state))))
        with self.assertRaises(ValueError):
            self.brain.add_link(self.ids[0], self.ids[0])

    def test_endpoint_vectors_are_cached_per_model(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), BagEndpoint)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            settings = update_settings(self.state, {'embeddings': {'provider': 'openai', 'url': f'http://127.0.0.1:{server.server_port}/v1', 'model': 'bag'}})
            result = self.semantic.suggestions(settings)
            self.assertIn(frozenset(self.ids[:2]), self.pairs(result))
            self.assertGreater(result['fetched'], 0)
            link = next(l for l in result['links'] if frozenset((l['source'], l['target'])) == frozenset(self.ids[:2]))
            self.assertEqual(len(link['why']), 2)
            self.brain.create('Новая заметка', 'Совсем другой текст про горы и лыжи. ' * 3)
            self.brain.sync()
            again = self.semantic.suggestions(settings)
            self.assertEqual(again['fetched'], 1)  # only the new short note: its head and single chunk coincide
            settings = update_settings(self.state, {'embeddings': {'model': 'bag-v2'}})
            self.assertGreater(self.semantic.suggestions(settings)['fetched'], 1)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
