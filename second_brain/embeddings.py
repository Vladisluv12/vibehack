"""Embedding settings and a small client for user-configured endpoints.

Settings live in the state directory (never in the vault). The API key is
write-only from the panel's point of view: it is stored, used, and reported
only as "set" or "not set".
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import urllib.error
from urllib.parse import urlsplit
import urllib.request

from .brain import atomic_write

PROVIDERS = {'local', 'openai', 'ollama', 'tei'}
CHAT_PROVIDERS = {'openai', 'ollama'}
SEARCH_MODES = {'bm25', 'semantic', 'hybrid'}
SPACES = {'human', 'agent'}
DEFAULTS = {
    'embeddings': {'provider': 'local', 'url': '', 'model': '', 'api_key': '', 'document_prefix': '', 'query_prefix': '', 'batch_size': 32},
    'semantic': {'spaces': ['human'], 'threshold': 'auto', 'top_k': 5},
    'search': {'mode': 'bm25'},
    'chat': {'provider': 'ollama', 'url': '', 'model': '', 'api_key': '', 'context_notes': 6},
}
MAX_RESPONSE = 50_000_000


def _merge(saved):
    result = json.loads(json.dumps(DEFAULTS))
    for section, values in (saved or {}).items():
        if section in result and isinstance(values, dict):
            result[section].update({k: v for k, v in values.items() if k in result[section]})
    return result


def load_settings(state: Path):
    path = Path(state) / 'settings.json'
    try:
        return _merge(json.loads(path.read_text(encoding='utf-8')))
    except FileNotFoundError:
        return _merge({})
    except (OSError, ValueError):
        raise ValueError('settings.json is unreadable; fix or delete it') from None


def public_settings(settings):
    """What the browser may see: everything except the key itself."""
    shown = json.loads(json.dumps(settings))
    for section in ('embeddings', 'chat'):
        shown[section]['api_key_set'] = bool(shown[section].pop('api_key'))
    return shown


def _check_endpoint(config, label):
    parts = urlsplit(config['url'])
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError(f'{label} URL must start with http:// or https://')


def update_settings(state: Path, changes):
    """Validate and persist. An omitted or null api_key keeps the stored one; "" clears it."""
    if not isinstance(changes, dict):
        raise ValueError('Invalid settings')
    current = load_settings(state)
    merged = _merge({s: {**current[s], **(changes.get(s) or {})} for s in current})
    for section in ('embeddings', 'chat'):
        if (changes.get(section) or {}).get('api_key') is None:
            merged[section]['api_key'] = current[section]['api_key']
    e, s, c = merged['embeddings'], merged['semantic'], merged['chat']
    if e['provider'] not in PROVIDERS:
        raise ValueError('Unknown embedding provider')
    for config, keys in ((e, ('url', 'model', 'api_key', 'document_prefix', 'query_prefix')), (c, ('url', 'model', 'api_key'))):
        for key in keys:
            if not isinstance(config[key], str) or len(config[key]) > 500 or '\n' in config[key] or '\r' in config[key]:
                raise ValueError(f'Invalid {key}')
        config['url'] = config['url'].strip()
    if e['provider'] != 'local':
        _check_endpoint(e, 'Endpoint')
        if e['provider'] != 'tei' and not e['model'].strip():
            raise ValueError('Model name is required for this provider')
    if type(e['batch_size']) is not int or not 1 <= e['batch_size'] <= 256:
        raise ValueError('Batch size must be between 1 and 256')
    if not isinstance(s['spaces'], list) or not s['spaces'] or not set(s['spaces']) <= SPACES:
        raise ValueError('Choose at least one space')
    if s['threshold'] != 'auto' and (not isinstance(s['threshold'], (int, float)) or isinstance(s['threshold'], bool) or not 0 <= s['threshold'] <= 1):
        raise ValueError('Threshold must be "auto" or between 0 and 1')
    if type(s['top_k']) is not int or not 1 <= s['top_k'] <= 20:
        raise ValueError('Suggestions per note must be between 1 and 20')
    s['spaces'] = sorted(set(s['spaces']))
    if merged['search']['mode'] not in SEARCH_MODES:
        raise ValueError('Unknown search mode')
    if c['provider'] not in CHAT_PROVIDERS:
        raise ValueError('Unknown chat provider')
    if c['url']:
        _check_endpoint(c, 'Chat')
        if not c['model'].strip():
            raise ValueError('Chat model name is required')
    if type(c['context_notes']) is not int or not 1 <= c['context_notes'] <= 12:
        raise ValueError('Notes per answer must be between 1 and 12')
    atomic_write(Path(state) / 'settings.json', json.dumps(merged, ensure_ascii=False, indent=2))
    return merged


def _post(url, payload, api_key, label='Embedding endpoint', timeout=120):
    headers = {'Content-Type': 'application/json', 'User-Agent': 'second-brain'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'{label} answered HTTP {exc.code}') from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f'{label} is unreachable: {getattr(exc, "reason", exc)}') from None
    if len(data) > MAX_RESPONSE:
        raise ValueError(f'{label}: response exceeds size limit')
    try:
        return json.loads(data)
    except ValueError:
        raise ValueError(f'{label} returned invalid JSON') from None


def chat_complete(messages, config):
    """One non-streaming chat completion from an OpenAI-compatible or Ollama endpoint."""
    if not config['url']:
        raise ValueError('Chat model is not configured: open settings')
    base = config['url'].rstrip('/')
    if config['provider'] == 'ollama':
        url = base if base.endswith('/api/chat') else base + '/api/chat'
        body = _post(url, {'model': config['model'], 'messages': messages, 'stream': False, 'think': False,
                           'options': {'temperature': 0.2}}, config['api_key'], 'Chat endpoint', 300)
        text = (body.get('message') or {}).get('content') if isinstance(body, dict) else None
    else:
        url = base if base.endswith('/chat/completions') else base + '/chat/completions'
        body = _post(url, {'model': config['model'], 'messages': messages, 'temperature': 0.2, 'stream': False},
                     config['api_key'], 'Chat endpoint', 300)
        choices = body.get('choices') if isinstance(body, dict) else None
        text = ((choices or [{}])[0].get('message') or {}).get('content') if isinstance(choices, list) and choices else None
    if not isinstance(text, str):
        raise ValueError('Chat endpoint returned an unexpected response')
    # Reasoning models may still emit their scratchpad; it is not part of the answer.
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def endpoint_embed(texts, config):
    """Vectors for `texts` from an OpenAI-compatible, Ollama or TEI endpoint."""
    base, provider = config['url'].rstrip('/'), config['provider']
    if provider == 'openai':
        url = base if base.endswith('/embeddings') else base + '/embeddings'
        body = _post(url, {'model': config['model'], 'input': texts, 'encoding_format': 'float'}, config['api_key'])
        rows = body.get('data') if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ValueError('Unexpected embedding response')
        vectors = [r.get('embedding') for r in sorted(rows, key=lambda r: r.get('index', 0)) if isinstance(r, dict)]
    elif provider == 'ollama':
        url = base if base.endswith('/api/embed') else base + '/api/embed'
        body = _post(url, {'model': config['model'], 'input': texts, 'truncate': True}, config['api_key'])
        vectors = body.get('embeddings') if isinstance(body, dict) else None
    elif provider == 'tei':
        url = base if base.endswith('/embed') else base + '/embed'
        vectors = _post(url, {'inputs': texts, 'truncate': True, 'normalize': True}, config['api_key'])
    else:
        raise ValueError('Not an endpoint provider')
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError('Embedding endpoint returned the wrong number of vectors')
    dims = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector or (dims and len(vector) != dims):
            raise ValueError('Embedding endpoint returned malformed vectors')
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
            raise ValueError('Embedding endpoint returned non-numeric values')
        dims = len(vector)
    return vectors
