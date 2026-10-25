"""Semantic link suggestions between notes.

Design, reimplemented from published approaches (no code copied):
- clean text: drop relation lines, replace [[id]] by the target's title, so
  existing links don't leak into similarity;
- heading-aware chunks (~1000 chars) prefixed with "title > H1 > H2", overlap
  only where a section is cut mid-text (Basic Memory, Khoj, Smart Connections);
- plus one "head" vector per note: title + first ~900 chars; short notes are a
  single chunk;
- pair score = 0.5·cos(heads) + 0.5·mean(S(A→B), S(B→A)), where S is the mean of
  the top-3 best chunk matches; candidates come from head similarity;
- mutual k-nearest neighbours (hubness guard), at most top_k suggestions per
  note, already linked and dismissed pairs excluded;
- threshold "auto" = max(model floor, 98th percentile of the vault's pairwise
  head similarity), because score scales differ per model.
Vectors are cached in `state/semantic.sqlite` by sha256 of the model identity
and the chunk text. Without an endpoint, TF-IDF over light-stemmed words and
character 3–5-grams gives a lexical (not semantic) fallback.
"""
from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import heapq
import json
import math
from operator import mul
from pathlib import Path
import re
import sqlite3

from .brain import SPACE_SQL
from .embeddings import endpoint_embed

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 120
HEAD_CHARS = 900
SHORT_NOTE = 300
MAX_CHUNKS = 48
MAX_NOTES = 2000
CANDIDATES = 30
NEIGHBOURS = 10
FLOOR = {'local': 0.08, 'endpoint': 0.3}
DUPLICATE = 0.95
HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
RELATIONS = re.compile(r'\n## Relations\s*\n', re.I)
RELATION_LINE = re.compile(r'(?m)^\s*-\s+[a-z_]+\s+\[\[[^\]\n]+\]\]\s*$')
WIKI = re.compile(r'\[\[([^\]\n]+)\]\]')


def clean_body(body, titles):
    body = RELATIONS.split(body.replace('\r\n', '\n'))[0]
    body = RELATION_LINE.sub('', body)

    def title(found):
        target, _, label = found[1].partition('|')
        return label.strip() or titles.get(target.split('#')[0].strip(), target.strip())
    return WIKI.sub(title, body).strip()


def chunk_note(title, body, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Heading-aware windows of a cleaned body, each prefixed with title > headings."""
    if len(body) < SHORT_NOTE:
        return [(title + '\n\n' + re.sub(r'\s+', ' ', body)).strip()]
    sections, path, lines = [], [], []

    def close():
        text = re.sub(r'\s+', ' ', ' '.join(lines)).strip()
        if text:
            sections.append((list(path), text))
        lines.clear()

    for line in body.split('\n'):
        found = HEADING.match(line)
        if found:
            close()
            path[:] = path[:len(found[1]) - 1] + [found[2].strip()]
        else:
            lines.append(line)
    close()
    chunks = []
    for heads, text in sections:
        prefix = ' > '.join([title, *heads])
        start = 0
        while start < len(text) and len(chunks) < MAX_CHUNKS:
            end = min(len(text), start + size)
            if end < len(text):
                cut = max(text.rfind('. ', start + size // 2, end), text.rfind(' ', start + size // 2, end))
                end = cut + 1 if cut > start else end
            chunks.append(prefix + '\n\n' + text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks or [title]


def normalise(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def dot(a, b):
    return sum(map(mul, a, b))


def directional(chunks_a, chunks_b):
    """S(A→B): mean of the top-3 best matches of A's chunks in B, plus the best pair."""
    best = []
    for i, a in enumerate(chunks_a):
        score, j = max((dot(a, b), j) for j, b in enumerate(chunks_b))
        best.append((score, i, j))
    best.sort(reverse=True)
    top = best[:3]
    return sum(s for s, _, _ in top) / len(top), best[0]


def ngram_vectors(texts):
    """TF-IDF over light-stemmed words and in-word character 3–5-grams; sparse, L2-normalised."""
    grams = []
    for text in texts:
        counts = Counter()
        for word in re.findall(r'\w+', text.lower().replace('ё', 'е')):
            if len(word) < 3 or word.isdigit():
                continue
            counts['w:' + word[:6]] += 2
            padded = f'<{word}>'
            for n in (3, 4, 5):
                counts.update(padded[i:i + n] for i in range(len(padded) - n + 1))
        grams.append(counts)
    df = Counter(g for counts in grams for g in counts)
    total = len(texts)
    vectors = []
    for counts in grams:
        # Grams in more than half the notes carry no signal for "related".
        weights = {g: (1 + math.log(c)) * math.log((1 + total) / (1 + df[g])) for g, c in counts.items() if df[g] <= max(2, total // 2)}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vectors.append({g: w / norm for g, w in weights.items()})
    return vectors


def sparse_similarity(vectors):
    """{i: {j: cosine}} for pairs sharing grams, via an inverted index."""
    index = {}
    for i, vector in enumerate(vectors):
        for gram, weight in vector.items():
            index.setdefault(gram, []).append((i, weight))
    result = {}
    for i, vector in enumerate(vectors):
        scores = Counter()
        for gram, weight in vector.items():
            for j, other in index[gram]:
                if j != i:
                    scores[j] += weight * other
        result[i] = dict(scores)
    return result


def percentile(values, total, q=0.98):
    """q-quantile of `values` padded with zeros up to `total` observations."""
    if total <= 0:
        return 0.0
    rank = int(q * total)
    ordered = sorted(values, reverse=True)
    position = total - 1 - rank
    return ordered[position] if 0 <= position < len(ordered) else 0.0


def mutual_neighbours(scores, n, k_neighbours, top_k, threshold, excluded):
    """scores: {i: {j: s}} symmetric. Keep i–j if each is in the other's top-k."""
    tops = {i: {j for _, j in heapq.nlargest(k_neighbours, ((s, j) for j, s in scores.get(i, {}).items() if (min(i, j), max(i, j)) not in excluded))}
            for i in range(n)}
    pairs = {}
    for i in range(n):
        ranked = sorted(((scores[i][j], j) for j in tops[i] if i in tops.get(j, ()) and scores[i][j] >= threshold), reverse=True)
        for s, j in ranked[:top_k]:
            pairs[(min(i, j), max(i, j))] = s
    return pairs


class Semantic:
    def __init__(self, brain, state: Path):
        self.brain = brain
        self.db = sqlite3.connect(Path(state) / 'semantic.sqlite', timeout=30)
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB);
        CREATE TABLE IF NOT EXISTS dismissed (a TEXT, b TEXT, PRIMARY KEY(a,b));
        CREATE TABLE IF NOT EXISTS results (fingerprint TEXT PRIMARY KEY, created REAL DEFAULT (julianday('now')), data TEXT);
        ''')

    def close(self):
        self.db.close()

    def dismiss(self, a, b):
        for ident in (a, b):
            self.brain.get(ident)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO dismissed VALUES (?,?)', tuple(sorted((a, b))))
        return {'dismissed': [a, b]}

    def _notes(self, spaces):
        marks = ','.join('?' * len(spaces))
        rows = self.brain.db.execute(f'SELECT n.id,n.title,n.body,n.hash FROM notes n WHERE {SPACE_SQL} IN ({marks}) ORDER BY n.id LIMIT ?',
                                     (*spaces, MAX_NOTES + 1)).fetchall()
        if len(rows) > MAX_NOTES:
            raise ValueError(f'Semantic suggestions support up to {MAX_NOTES} notes in the chosen spaces')
        return rows

    def _embed(self, texts, config):
        """Normalised cached vectors for every text; missing ones fetched in batches."""
        identity = json.dumps([config['provider'], config['url'], config['model'], config['document_prefix']])
        keys = [hashlib.sha256((identity + '\0' + t).encode()).hexdigest() for t in texts]
        cached = {}
        unique = list(dict.fromkeys(keys))
        for i in range(0, len(unique), 500):
            part = unique[i:i + 500]
            cached.update(self.db.execute(f'SELECT key,vec FROM vectors WHERE key IN ({",".join("?" * len(part))})', part))
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in cached]
        for i in range(0, len(missing), config['batch_size']):
            batch = missing[i:i + config['batch_size']]
            vectors = endpoint_embed([config['document_prefix'] + t for _, t in batch], config)
            with self.db:
                for (key, _), vector in zip(batch, vectors):
                    cached[key] = array('f', normalise(vector)).tobytes()
                    self.db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)', (key, cached[key]))
        return [list(array('f', cached[k])) for k in keys], len(missing)

    def _score_endpoint(self, texts, config, excluded):
        heads_text = [(t + '\n\n' + b[:HEAD_CHARS]).strip() for t, b in texts]
        chunks_text = [chunk_note(t, b) for t, b in texts]
        flat = heads_text + [c for group in chunks_text for c in group]
        vectors, fetched = self._embed(flat, config)
        n = len(texts)
        heads, chunks, at = vectors[:n], [], n
        for group in chunks_text:
            chunks.append(vectors[at:at + len(group)])
            at += len(group)
        head_scores = {i: {} for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                head_scores[i][j] = head_scores[j][i] = dot(heads[i], heads[j])
        floor = max(FLOOR['endpoint'], percentile([s for i in head_scores for j, s in head_scores[i].items() if j > i], n * (n - 1) // 2))
        scores, why = {i: {} for i in range(n)}, {}
        for i in range(n):
            for j in heapq.nlargest(CANDIDATES, head_scores[i], key=head_scores[i].get):
                if j in scores[i]:
                    continue
                ab, (_, ca, cb) = directional(chunks[i], chunks[j])
                ba, _ = directional(chunks[j], chunks[i])
                scores[i][j] = scores[j][i] = 0.5 * head_scores[i][j] + 0.25 * (ab + ba)
                why[(min(i, j), max(i, j))] = (chunks_text[i][ca], chunks_text[j][cb]) if i < j else (chunks_text[j][cb], chunks_text[i][ca])
        return scores, floor, why, fetched

    def suggestions(self, settings):
        config, params = settings['embeddings'], settings['semantic']
        notes = self._notes(params['spaces'])
        titles = dict(self.brain.db.execute('SELECT id,title FROM notes'))
        linked = {tuple(sorted(r)) for r in self.brain.db.execute('SELECT source,target FROM links')}
        dismissed = set(self.db.execute('SELECT a,b FROM dismissed'))
        fingerprint = hashlib.sha256(json.dumps([config['provider'], config['url'], config['model'], config['document_prefix'], params,
                                                 [(r['id'], r['hash']) for r in notes], sorted(titles.items()),
                                                 sorted(linked), sorted(dismissed)]).encode()).hexdigest()
        row = self.db.execute('SELECT data FROM results WHERE fingerprint=?', (fingerprint,)).fetchone()
        if row:
            return {**json.loads(row[0]), 'cached': True}
        ids = [r['id'] for r in notes]
        index = {ident: i for i, ident in enumerate(ids)}
        excluded = {(min(index[a], index[b]), max(index[a], index[b])) for a, b in linked | dismissed if a in index and b in index}
        texts = [(r['title'], clean_body(r['body'], titles)) for r in notes]
        n = len(ids)
        if config['provider'] == 'local':
            scores = sparse_similarity(ngram_vectors([f'{t} {t} {b}' for t, b in texts]))
            floor = max(FLOOR['local'], percentile([s for i in scores for j, s in scores[i].items() if j > i], n * (n - 1) // 2))
            why, fetched = {}, 0
        else:
            scores, floor, why, fetched = self._score_endpoint(texts, config, excluded)
        threshold = floor if params['threshold'] == 'auto' else params['threshold']
        pairs = mutual_neighbours(scores, n, NEIGHBOURS, params['top_k'], threshold, excluded)
        links = []
        for (i, j), s in sorted(pairs.items(), key=lambda x: -x[1]):
            link = {'source': ids[i], 'target': ids[j], 'score': round(s, 4)}
            if config['provider'] != 'local' and s >= DUPLICATE:
                link['duplicate'] = True
            if (i, j) in why:
                link['why'] = [part.split('\n\n', 1)[-1][:240] for part in why[(i, j)]]
            links.append(link)
        result = {'links': links, 'provider': config['provider'], 'model': config['model'], 'notes': n,
                  'spaces': params['spaces'], 'threshold': round(threshold, 4), 'auto': params['threshold'] == 'auto', 'fetched': fetched}
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO results(fingerprint,data) VALUES (?,?)', (fingerprint, json.dumps(result, ensure_ascii=False)))
            self.db.execute('DELETE FROM results WHERE fingerprint NOT IN (SELECT fingerprint FROM results ORDER BY created DESC LIMIT 8)')
        return {**result, 'cached': False}


def test_connection(settings):
    config = settings['embeddings']
    if config['provider'] == 'local':
        return {'provider': 'local', 'model': 'TF-IDF по n-граммам', 'dims': 'разреженная'}
    vector = endpoint_embed([config['document_prefix'] + 'Проверка связи с эндпоинтом эмбеддингов'], config)[0]
    return {'provider': config['provider'], 'model': config['model'], 'dims': len(vector)}
