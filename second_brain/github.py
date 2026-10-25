"""Import the Markdown files of a GitHub repository as unreviewed sources.

The only network access in the package, and only on explicit request. The
repository tarball is read in memory: nothing is extracted to disk, only regular
`.md` members are considered, and every size is bounded. Imported text is
external content: its wiki links never become graph relations.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import PurePosixPath
import re
import tarfile
import urllib.error
from urllib.parse import quote, urlsplit
import urllib.request

from .brain import MAX_NOTE, digest

REPO_URL = re.compile(r'^(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]{1,100}?)(?:\.git)?/?$')
REPO_SHORT = re.compile(r'^([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]{1,100})$')
HOSTS = {'api.github.com', 'codeload.github.com'}
MAX_META = 1_000_000
MAX_ARCHIVE = 50_000_000
MAX_UNPACKED = 1_000_000_000
MAX_MEMBERS = 50_000
MAX_FILES = 200
MAX_TOTAL = 20_000_000
SKIP_DIRS = {'node_modules', 'vendor', 'third_party'}


def parse_repo(url):
    """`https://github.com/owner/repo`, `github.com/owner/repo(.git)` or `owner/repo`."""
    text = url.strip() if isinstance(url, str) else ''
    found = REPO_URL.fullmatch(text) or REPO_SHORT.fullmatch(text)
    if not found or found[2] in {'.', '..'}:
        raise ValueError('Expected a GitHub repository like https://github.com/owner/repo')
    return found[1], found[2].removesuffix('.git')


class _GitHubOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != 'https' or parts.hostname not in HOSTS:
            raise ValueError('GitHub redirected outside GitHub')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, limit):
    headers = {'User-Agent': 'second-brain', 'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['GITHUB_TOKEN']
    try:
        with urllib.request.build_opener(_GitHubOnly).open(urllib.request.Request(url, headers=headers), timeout=60) as response:
            data = response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError('Repository not found (private repositories need GITHUB_TOKEN)') from None
        if exc.code in (401, 403, 429):
            raise ValueError('GitHub refused the request: rate limit or token lacks access') from None
        raise ValueError(f'GitHub answered HTTP {exc.code}') from None
    except urllib.error.URLError:
        raise ValueError('GitHub is unreachable: check the network connection') from None
    if len(data) > limit:
        raise ValueError('GitHub response exceeds size limit')
    return data


def markdown_members(archive):
    """[(repo-relative path, text, raw bytes)] plus counts of skipped and omitted files."""
    files, skipped, omitted, unpacked, total = [], 0, 0, 0, 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for count, member in enumerate(tar):
            unpacked += max(0, member.size)
            if count >= MAX_MEMBERS or unpacked > MAX_UNPACKED:
                raise ValueError('Repository archive exceeds limits')
            parts = PurePosixPath(member.name).parts[1:]  # drop GitHub's "<owner>-<repo>-<sha>/" root
            if not member.isfile() or not parts or not parts[-1].lower().endswith('.md'):
                continue
            if any(p in ('', '..') or p.startswith('.') or p in SKIP_DIRS for p in parts):
                continue
            if member.size > MAX_NOTE:
                skipped += 1
                continue
            if len(files) >= MAX_FILES or total + member.size > MAX_TOTAL:
                omitted += 1
                continue
            raw = tar.extractfile(member).read(MAX_NOTE + 1)
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                skipped += 1
                continue
            total += len(raw)
            files.append(('/'.join(parts), text, raw))
    files.sort(key=lambda f: (f[0].count('/') > 0, not f[0].lower().startswith('readme'), f[0].lower()))
    return files, skipped, omitted


SHA = re.compile(r'^[0-9a-f]{40}$')


def previous_version(brain, home, path):
    row = brain.db.execute(
        "SELECT id, meta FROM notes WHERE kind='source' AND json_extract(meta,'$.repository')=? "
        "AND json_extract(meta,'$.repo_path')=? AND id NOT IN (SELECT json_extract(meta,'$.supersedes') FROM notes "
        "WHERE json_extract(meta,'$.supersedes') IS NOT NULL) ORDER BY json_extract(meta,'$.created') DESC, id LIMIT 1",
        (home, path)).fetchone()
    return (row['id'], json.loads(row['meta']).get('source_hash', '')) if row else (None, '')


def import_github(brain, url, fetcher=None):
    """One import run per commit. Branch names move, so sources cite the resolved
    commit; unchanged files keep their note, a changed file gets a new note that
    names the version it supersedes. Nothing is overwritten or deleted."""
    fetcher = fetcher or fetch
    owner, repo = parse_repo(url)
    api = f'https://api.github.com/repos/{owner}/{repo}'
    meta = json.loads(fetcher(api, MAX_META))
    branch = meta.get('default_branch') if isinstance(meta, dict) else None
    if not isinstance(branch, str) or not branch or len(branch) > 255:
        raise ValueError('GitHub returned unexpected repository metadata')
    description = meta.get('description') if isinstance(meta.get('description'), str) else ''
    head = json.loads(fetcher(f'{api}/branches/{quote(branch, safe="")}', MAX_META))
    commit = head.get('commit', {}).get('sha') if isinstance(head, dict) and isinstance(head.get('commit'), dict) else None
    if not isinstance(commit, str) or not SHA.fullmatch(commit):
        raise ValueError('GitHub returned unexpected branch metadata')
    archive = fetcher(f'{api}/tarball/{commit}', MAX_ARCHIVE)
    files, skipped, omitted = markdown_members(archive)
    home = f'https://github.com/{owner}/{repo}'
    body = (description.strip() + '\n\n' if description.strip() else '') + 'Репозиторий GitHub. Каждый импорт — отдельная запись сессии с коммитом и списком файлов.'
    project = brain.create(f'{owner}/{repo}', body, kind='project', review='unreviewed', source=home,
                           key='github:' + home, external_content=True,
                           extra={'repository': home, 'branch': branch})
    brain.sync()
    sources, changed = [], 0
    for path, text, raw in files:
        previous, previous_hash = previous_version(brain, home, path)
        if previous and previous_hash == digest(raw):
            sources.append(previous)
            continue
        extra = {'repository': home, 'repo_path': path, 'commit': commit}
        if previous:
            extra['supersedes'] = previous
            changed += 1
        sources.append(brain.create(f'{repo}/{path}'[:200], text, kind='source', review='unreviewed',
                                    source=f'{home}/blob/{commit}/{quote(path)}', source_hash=digest(raw),
                                    # A new version is new even if it equals an older one (A→B→A, C→B):
                                    # its identity includes the version it replaces.
                                    key=f'{home}#{path}@{digest(raw)}' + (f'@after:{previous}' if previous else ''),
                                    links=[project],
                                    external_content=True, extra=extra))
    run = brain.create(f'{owner}/{repo} @ {commit[:12]}',
                       f'Импорт ветки `{branch}`, коммит `{commit}`. Markdown-файлов: {len(files)}, '
                       f'изменилось: {changed}, пропущено: {skipped}, не вошло по лимиту: {omitted}.',
                       kind='session', review='unreviewed', source=f'{home}/tree/{commit}',
                       source_hash=digest(archive), key=f'github-run:{home}@{commit}', links=[project, *dict.fromkeys(sources)],
                       external_content=True, extra={'repository': home, 'branch': branch, 'commit': commit})
    return {'project': project, 'run': run, 'repository': home, 'branch': branch, 'commit': commit,
            'imported': len(sources), 'changed': changed, 'sources': sources, 'skipped': skipped, 'omitted': omitted}
