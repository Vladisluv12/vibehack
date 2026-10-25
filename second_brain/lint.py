"""Deterministic, uncapped vault checks. Only ``online=True`` permits URL requests.

Read the entire projection, reusing it when file hashes match. Markdown is also
scanned because duplicate IDs cannot be represented by the notes primary key,
and a failed sync must not prevent lint from explaining how to repair the vault.
Codes are stable snake_case strings; every finding identifies its source note.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import ipaddress
import socket
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .brain import LINK, LINK_METADATA, declared_links, digest, parse_note, without_fences
from .redact import redact


def _target(value):
    value = value.strip()
    if value.startswith('[[') and value.endswith(']]'):
        value = value[2:-2]
    return value.split('|')[0].split('#')[0].strip()


def _nonempty(value):
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def public_url(url):
    """Sources are untrusted: only http(s) URLs whose every address is public.

    Blocks loopback, private, link-local (cloud metadata) and reserved ranges, also
    after each redirect. A resolver that answers differently on connect (DNS
    rebinding) is not fully excluded by a check before the request.
    """
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError('only http(s) URLs with a host are checked')
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80))
    except OSError as exc:
        raise ValueError(f'unresolvable host: {exc}') from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split('%')[0]).is_global:
            raise ValueError(f'non-public address {info[4][0]}')
    return url


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, public_url(newurl))


def urlopen(request, timeout):
    public_url(request.full_url)
    return build_opener(_PublicRedirects).open(request, timeout=timeout)


def _fetch_status(url):
    """Redirects are followed only to public addresses; servers that reject HEAD get a GET instead."""
    for method in ('HEAD', 'GET'):
        try:
            with urlopen(Request(url, method=method), timeout=10) as response:
                return response.status
        except HTTPError as exc:
            status = exc.code
            exc.close()
            if method != 'HEAD' or status not in (405, 501):
                return status


def _snapshot(brain, warnings):
    with brain.lock():
        projected = {r['path']: dict(r) for r in brain.db.execute('SELECT * FROM notes ORDER BY path, id')}
        edges = {tuple(r) for r in brain.db.execute('SELECT source, target, relation FROM links')}
        notes, unchanged = [], True
        for path in sorted(brain.vault.rglob('*.md')):
            rel = path.relative_to(brain.vault).as_posix()
            if Path(rel).parts[0] == 'Versions' or any(p.startswith('.') for p in Path(rel).parts):
                continue
            old = projected.get(rel)
            try:
                raw = brain._read(brain.safe(rel))
                hashed = digest(raw)
                if old and old['hash'] == hashed:
                    note = {**old, 'meta': json.loads(old['meta'])}
                else:
                    meta, body = parse_note(redact(raw), rel)
                    note = dict(id=meta['id'], path=rel, title=meta['title'], kind=meta['type'],
                                review=meta['review'], meta=meta, body=body)
                    unchanged = False
                notes.append(note)
            except (OSError, ValueError) as exc:
                unchanged = False
                warnings.append(dict(code='unreadable_note', note_id=old['id'] if old else '',
                                     path=rel, detail=str(exc)))
        return notes, edges if unchanged and len(notes) == len(projected) else None


def lint(brain, *, online=False, fetch=None) -> dict:
    """Return errors, warnings, orphan candidates and totals, without changing notes.

    Link resolution and fenced-code handling match Brain's projection. Orphans
    use all canonical edges, never proposed links or the capped graph view.
    Unsynced edits are reparsed, including a vault whose duplicate IDs block sync.
    ``fetch(url)`` returns an HTTP status or raises on an unreachable URL.
    """
    report = {'errors': [], 'warnings': [], 'candidates': []}
    notes, projected_edges = _snapshot(brain, report['warnings'])
    aliases, titles, identities = defaultdict(set), defaultdict(list), defaultdict(list)
    for note in notes:
        ident, rel, meta = note['id'], note['path'], note['meta']
        identities[ident].append(note)
        titles[note['title'].casefold()].append(note)
        for key in (ident, note['title'], str(meta['permalink']), rel, rel.removesuffix('.md'), Path(rel).stem):
            aliases[key.casefold()].add(ident)

    def add(group, code, note, detail):
        report[group].append(dict(code=code, note_id=note['id'], path=note['path'], detail=detail))

    for group, code, duplicates in (('errors', 'duplicate_id', identities), ('warnings', 'duplicate_title', titles)):
        for members in duplicates.values():
            if len(members) > 1:
                paths = ', '.join(n['path'] for n in members)
                for note in members:
                    add(group, code, note, paths)

    edges = set()
    for note in notes:
        meta = note['meta']
        hashed = meta.get('source_hash')
        if hashed not in (None, '') and (not isinstance(hashed, str) or not re.fullmatch('[a-f0-9]{64}', hashed)):
            add('errors', 'invalid_source_hash', note, 'source_hash must be 64 lowercase hexadecimal characters')
        if note['review'] == 'proposed' and not _nonempty(meta.get('source')):
            # Brain.create makes human notes proposed with no source by default: worth a look, not a failure.
            add('warnings', 'proposed_without_source', note, 'proposed note without a source')
        supersedes = meta.get('supersedes')
        if _nonempty(supersedes) and (not isinstance(supersedes, str) or supersedes not in identities):
            add('errors', 'missing_supersedes', note, f'supersedes target does not exist: {supersedes}')
        if meta.get('memory_type') in ('skill', 'experience') or note['kind'] in ('skill', 'experience'):
            for field in ('applies_to', 'evidence', 'declared_links'):
                if not _nonempty(meta.get(field)):
                    add('errors', 'missing_' + field, note, f'{field} is required for skills and experience')

        declared = meta.get('declared_links', [])
        if isinstance(declared, (list, tuple)):
            for target in declared:
                if not isinstance(target, str) or target not in identities:
                    add('errors', 'unresolved_declared_link', note, f'declared target does not exist: {target}')

        # Invalid external policy must not turn untrusted text into assertions.
        has_policy = bool(LINK_METADATA.intersection(meta))
        if has_policy:
            try:
                external, only = meta.get('external_content'), meta.get('declared_only', False)
                if (type(external) is not bool or type(only) is not bool or not (external or only)
                        or meta.get('link_policy') != 'declared'):
                    raise ValueError('Invalid link policy')
                targets = declared_links(declared)
            except ValueError:
                targets = []
            edges.update((note['id'], target, 'relates_to') for target in targets if target in identities)
        if not meta.get('declared_only') and not meta.get('external_content'):
            for line in without_fences(note['body']).splitlines():
                relation = re.match(r'^\s*-\s+([a-z_]+)\s+\[\[', line)
                for found in LINK.finditer(line):
                    target = _target(found[1])
                    matches = aliases.get(target.casefold(), set())
                    if len(matches) == 1:
                        if not has_policy:
                            edges.add((note['id'], next(iter(matches)), relation[1] if relation else 'relates_to'))
                    else:
                        add('errors', 'ambiguous_link' if matches else 'unresolved_link', note, target)

        proposed = meta.get('proposed_links', [])
        if isinstance(proposed, (list, tuple)):
            for target in proposed:
                if isinstance(target, str) and not aliases.get(_target(target).casefold()):
                    add('warnings', 'unresolved_proposed_link', note, target)

    if projected_edges is not None:
        edges = projected_edges
    connected = {ident for source, target, _ in edges for ident in (source, target)}
    for note in notes:
        if note['id'] not in connected and note['kind'] not in ('project', 'session'):
            add('candidates', 'orphan', note, 'no incoming or outgoing canonical links')

    if online:
        by_url = defaultdict(list)
        for note in notes:
            source = note['meta'].get('source')
            if isinstance(source, str):
                # Checking the prefix avoids malformed URLs raising before fetch.
                if source.strip().lower().startswith(('http://', 'https://')):
                    by_url[source.strip()].append(note)
        fetch = fetch or _fetch_status
        for url, members in sorted(by_url.items()):
            try:
                if urlsplit(url).scheme.lower() not in ('http', 'https'):
                    continue
                status = fetch(url)
                if type(status) is not int or not 100 <= status <= 599:
                    raise ValueError('fetch must return an HTTP status integer')
                code, detail = 'source_http_error', f'{url}: HTTP {status}'
                if status < 400:
                    continue
            except HTTPError as exc:
                code, detail = 'source_http_error', f'{url}: HTTP {exc.code}'
                exc.close()
            except Exception as exc:
                code, detail = 'source_unreachable', f'{url}: {exc}'
            for note in members:
                add('warnings', code, note, detail)

    for findings in report.values():
        findings.sort(key=lambda item: (item['path'], item['code'], item['note_id'], item['detail']))
    report['counts'] = {'notes': len(notes), 'links': len(edges),
                        **{group: len(report[group]) for group in ('errors', 'warnings', 'candidates')}}
    return report


def exit_code(report) -> int:
    return int(bool(report['errors']))
