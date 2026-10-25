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


def import_github(brain, url, fetcher=None):
    fetcher = fetcher or fetch
    owner, repo = parse_repo(url)
    meta = json.loads(fetcher(f'https://api.github.com/repos/{owner}/{repo}', MAX_META))
    branch = meta.get('default_branch') if isinstance(meta, dict) else None
    if not isinstance(branch, str) or not branch or len(branch) > 255:
        raise ValueError('GitHub returned unexpected repository metadata')
    description = meta.get('description') if isinstance(meta.get('description'), str) else ''
    archive = fetcher(f'https://api.github.com/repos/{owner}/{repo}/tarball/{quote(branch, safe="")}', MAX_ARCHIVE)
    files, skipped, omitted = markdown_members(archive)
    home = f'https://github.com/{owner}/{repo}'
    body = (description.strip() + '\n\n' if description.strip() else '') + f'Репозиторий GitHub, ветка `{branch}`. Markdown-файлов импортировано: {len(files)}.'
    project = brain.create(f'{owner}/{repo}', body, kind='project', review='unreviewed', source=home,
                           source_hash=digest(archive), key='github:' + home, external_content=True,
                           extra={'repository': home, 'branch': branch})
    sources = []
    for path, text, raw in files:
        source = f'{home}/blob/{quote(branch, safe="")}/{quote(path)}'
        sources.append(brain.create(f'{repo}/{path}'[:200], text, kind='source', review='unreviewed', source=source,
                                    source_hash=digest(raw), key=source + digest(raw), links=[project],
                                    external_content=True, extra={'repository': home, 'repo_path': path}))
    return {'project': project, 'repository': home, 'branch': branch, 'imported': len(sources),
            'sources': sources, 'skipped': skipped, 'omitted': omitted}
