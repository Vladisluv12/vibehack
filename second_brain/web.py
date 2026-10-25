from __future__ import annotations

import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from .brain import ID, MAX_TAGS, Brain, normalise_tag
from .activity import safe_record, since
from .chat import answer as chat_answer
from .embeddings import chat_complete, load_settings, public_settings, update_settings
from .github import import_github
from .semantic import Semantic, test_connection
from .redact import redact

HERE = Path(__file__).parent
ICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8 1.5 14.5 8 8 14.5 1.5 8z" fill="none" stroke="#3567d6" stroke-width="2"/></svg>'


class Extension:
    """Optional plugin slot, e.g. Agent Control. The brain never imports it.

    `get`/`post` return None for paths they do not own. `pages` are extra HTML
    documents served like the dashboard, with __CONTROL_TOKEN__ substituted.
    """
    name = ''
    pages: dict = {}
    post_limit = 30_000

    def info(self):
        return {}

    def get(self, path, params, open_brain):
        return None

    def post(self, path, data, open_brain):
        return None


def make_server(vault, state, port=8765, extension=None, open_brain=None):
    token = secrets.token_urlsafe(32)
    page = (HERE / "dashboard.html").read_text(encoding="utf-8").replace("__CONTROL_TOKEN__", token)
    pages = {path: text.replace("__CONTROL_TOKEN__", token) for path, text in (extension.pages if extension else {}).items()}
    status = {'connected': bool(extension), 'name': extension.name if extension else '',
              **(extension.info() if extension else {})}

    state_dir = Path(state).expanduser()

    def brain():
        if open_brain:
            return open_brain()
        return Brain(Path(vault).expanduser(), Path(state).expanduser())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, mime="application/json"):
            payload = body.encode() if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime+"; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def check(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self.send(403, {"error": "Invalid host"})
                return False
            if urlsplit(self.path).path not in {'/', '/assets/app.js', '/assets/app.css', '/assets/icon.svg', *pages}:
                if not hmac.compare_digest(self.headers.get("X-Control-Token", ""), token):
                    self.send(403, {"error": "Local control token required"})
                    return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://"+x for x in expected}:
                self.send(403, {"error": "Foreign origin"})
                return False
            return True

        def do_GET(self):
            if not self.check():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                return self.send(200, page, "text/html")
            if url.path == '/assets/icon.svg':
                return self.send(200, ICON, 'image/svg+xml')
            if url.path in {'/assets/app.js', '/assets/app.css'}:
                path = HERE/'static'/url.path.rsplit('/', 1)[1]
                return self.send(200, path.read_text(encoding="utf-8"), 'text/javascript' if path.suffix == '.js' else 'text/css')
            if url.path in pages:
                return self.send(200, pages[url.path], "text/html")
            if url.path == '/api/extension':
                return self.send(200, status)
            if url.path == '/api/settings':
                try:
                    return self.send(200, public_settings(load_settings(state_dir)))
                except ValueError as exc:
                    return self.send(400, {'error': str(exc)})
            if not url.path.startswith('/api/brain/'):
                return self.extend('get', url.path, parse_qs(url.query))
            b = brain()
            try:
                params = parse_qs(url.query)
                action = url.path.rsplit('/', 1)[1]
                if action == 'notes':
                    result = b.list(params.get('q', [''])[0], params.get('kind', [''])[0],
                                    int(params.get('offset', [0])[0]), int(params.get('limit', [50])[0]),
                                    params.get('space', [''])[0], params.get('review', [''])[0], params.get('tag', [''])[0])
                elif action == 'note':
                    result = b.get(params.get('id', [''])[0])
                elif action == 'graph':
                    result = b.graph(params.get('q', [''])[0], params.get('kind', [''])[0], space=params.get('space', [''])[0])
                elif action == 'activity':
                    after = params.get('after', [None])[0]
                    result = since(state_dir, int(after) if after not in (None, '') else None)
                    ids = {i for item in result['items'] for i in item['ids']}
                    result['titles'] = {r[0]: r[1] for r in b.db.execute(
                        f"SELECT id,title FROM notes WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))} if ids else {}
                elif action == 'tags':
                    result = b.tags(params.get('space', [''])[0])
                elif action == 'search':
                    semantic = Semantic(b, state_dir)
                    try:
                        result = semantic.search(params.get('q', [''])[0], params.get('mode', ['bm25'])[0], load_settings(state_dir),
                                                 params.get('space', [''])[0], min(50, max(1, int(params.get('limit', [20])[0]))))
                    finally:
                        semantic.close()
                elif action == 'semantic':
                    semantic = Semantic(b, state_dir)
                    try:
                        result = semantic.suggestions(load_settings(state_dir))
                    finally:
                        semantic.close()
                else:
                    return self.send(404, {'error': 'Not found'})
                return self.send(200, result)
            except (ValueError, TypeError, OSError) as exc:
                return self.send(400, {'error': redact(str(exc))})
            finally:
                b.close()

        def extend(self, method, path, payload):
            if not extension:
                return self.send(404, {'error': 'Not found'})
            try:
                result = getattr(extension, method)(path, payload, brain)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self.send(400, {'error': redact(str(exc))})
            if result is None:
                return self.send(404, {'error': 'Not found'})
            code, body = result if method == 'post' else (200, result)
            return self.send(code, body)

        def body(self, limit):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= limit:
                raise ValueError("Request too large or empty")
            return json.loads(self.rfile.read(size))

        def do_POST(self):
            if not self.check():
                return
            if self.path in ('/api/settings', '/api/settings/test', '/api/settings/test-chat'):
                try:
                    data = self.body(20_000)
                    if self.path == '/api/settings':
                        return self.send(200, public_settings(update_settings(state_dir, data)))
                    settings = load_settings(state_dir)
                    if self.path == '/api/settings/test-chat':
                        reply = chat_complete([{'role': 'user', 'content': 'Ответь одним словом: связь есть?'}], settings['chat'])
                        return self.send(200, {'model': settings['chat']['model'], 'reply': reply[:200]})
                    return self.send(200, test_connection(settings))
                except (ValueError, TypeError, OSError) as exc:
                    return self.send(400, {'error': redact(str(exc))})
            if not self.path.startswith('/api/brain/'):
                if not extension:
                    return self.send(404, {'error': 'Not found'})
                try:
                    data = self.body(extension.post_limit)
                except (ValueError, TypeError) as exc:
                    return self.send(400, {'error': redact(str(exc))})
                return self.extend('post', urlsplit(self.path).path, data)
            b = brain()
            try:
                data = self.body(2_100_000)
                action = self.path.rsplit('/', 1)[1]
                if action == 'sync':
                    result = b.sync()
                elif action == 'backup':
                    result = b.backup()
                elif action == 'edit':
                    result = b.edit(data['id'], data['content'], data['hash'])
                elif action == 'import-github':
                    result = import_github(b, data['url'])
                    b.sync()
                elif action == 'chat':
                    semantic = Semantic(b, state_dir)
                    try:
                        result = chat_answer(b, semantic, load_settings(state_dir), data.get('messages'), data.get('space') or '')
                    finally:
                        semantic.close()
                    question = data['messages'][-1]['content']
                    safe_record(state_dir, 'chat', 'memory.search', [s['id'] for s in result['sources']], question)
                    safe_record(state_dir, 'chat', 'chat.answer', [s['id'] for s in result['sources'] if s['cited']], question)
                elif action == 'link':
                    result = b.add_link(data['source'], data['target'])
                elif action == 'dismiss':
                    semantic = Semantic(b, state_dir)
                    try:
                        result = semantic.dismiss(data['source'], data['target'])
                    finally:
                        semantic.close()
                elif action == 'create':
                    tags = data.get('tags') or []
                    if not isinstance(tags, list) or len(tags) > MAX_TAGS or any(normalise_tag(t) is None for t in tags):
                        raise ValueError('Tags must be words like "поиск" or "project/alpha"')
                    links = data.get('links') or []
                    if not isinstance(links, list) or len(links) > 30 or any(not isinstance(i, str) or not ID.fullmatch(i) for i in links):
                        raise ValueError('Invalid links')
                    for ident in links:
                        b.get(ident)
                    # A saved chat answer is model output: it lands in the agent space for review.
                    source = 'agent:chat' if data.get('origin') == 'chat' else ''
                    ident = b.create(data['title'], data['body'], kind=data.get('kind', 'note'), review='proposed', source=source,
                                     links=links, extra={'tags': sorted({normalise_tag(t) for t in tags})} if tags else None)
                    b.sync()
                    result = {'id': ident}
                else:
                    return self.send(404, {'error': 'Not found'})
                return self.send(200, result)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                return self.send(400, {'error': redact(str(exc))})
            finally:
                b.close()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
