"""Canonical Markdown vault and disposable, local graph/search projection.

No provider calls. External content is data; imported notes are unreviewed.
Only explicit document imports read outside the vault (CLI, never HTTP).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import time
from urllib.parse import quote
import uuid
import zipfile

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None
    import msvcrt

from .redact import redact

LOCK_NAME = ".second-brain.lock"
MAX_NOTE = 2_000_000
MAX_VAULT = 100_000_000
KINDS = {"project", "research", "source", "decision", "session", "concept", "note"}
REVIEWS = {"unreviewed", "proposed", "reviewed"}
SPACES = {"human", "agent"}
# Who wrote a note: explicit `space` metadata, else agent drafts carry an `agent:` source.
SPACE_SQL = ("(CASE WHEN json_extract(n.meta,'$.space') IN ('human','agent') THEN json_extract(n.meta,'$.space') "
             "WHEN json_extract(n.meta,'$.source') LIKE 'agent:%' THEN 'agent' ELSE 'human' END)")
FOLDERS = {k: k.capitalize() + "s" for k in KINDS}
ID = re.compile(r"^[a-f0-9]{32}$")
LINK = re.compile(r"\[\[([^\]\n]+)\]\]")
MAX_DECLARED_LINKS = 512
LINK_METADATA = {"external_content", "declared_only", "link_policy", "declared_links"}
OWNED_METADATA = {"id", "title", "type", "permalink", "review", "created", "source", "source_hash"}


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def declared_links(value):
    """Explicit wrapper-owned IDs; source text cannot supply graph assertions."""
    if not isinstance(value, (list, tuple)) or len(value) > MAX_DECLARED_LINKS:
        raise ValueError("Invalid declared links")
    if any(not isinstance(ident, str) or not ID.fullmatch(ident) for ident in value):
        raise ValueError("Declared links require canonical note IDs")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate declared links")
    return list(value)


def unsafe_relative(rel):
    """Absolute, drive-relative (Windows `/x`, `C:x`), parent or hidden paths."""
    return rel.is_absolute() or bool(rel.anchor) or ".." in rel.parts or any(p.startswith(".") for p in rel.parts)


def path_uri(value):
    """file:// URI matching import sources; POSIX metadata also resolves on Windows."""
    if Path(value).is_absolute():
        return Path(value).as_uri()
    if PurePosixPath(value).is_absolute():
        return 'file://' + quote(value)
    return value


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkstemp creates the file 0600. Byte-exact: no newline translation on Windows.
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                # Windows refuses to replace a file another reader holds open.
                if os.name != "nt" or attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def parse_note(text, relative):
    """Read our scalar YAML subset, while tolerating ordinary Obsidian notes.

    No YAML execution. Complex metadata remains in the original Markdown, and
    isn't interpreted by this projection. Basic Memory has its own YAML parser.
    """
    text = text.replace('\r\n', '\n')
    meta, body = {}, text
    if text.startswith("---\n") and "\n---\n" in text[4:]:
        head, body = text[4:].split("\n---\n", 1)
        listing, blank = None, set()
        for line in head.splitlines():
            item = re.match(r"^\s+-\s+(.+)$", line)
            if listing and item:
                meta[listing].append(item[1].strip().strip("'\""))
                continue
            listing = None
            if re.match(r"^[a-z_][a-z0-9_]*:", line):
                key, value = line.split(":", 1)
                value = value.strip()
                if not value:
                    meta[key], listing = [], key
                    blank.add(key)
                    continue
                try:
                    meta[key] = json.loads(value)
                except ValueError:
                    meta[key] = value.strip("'\"")
        meta.update({key: '' for key in blank if meta.get(key) == []})
    title = meta.get("title") or Path(relative).stem
    if not isinstance(title, str):
        title = Path(relative).stem
    ident = meta.get("id")
    if not isinstance(ident, str) or not ID.fullmatch(ident):
        ident = digest("external:" + relative)[:32]
    kind = meta.get("type", "note")
    meta.update(id=ident, title=title[:200], type=kind if isinstance(kind,str) and kind in KINDS else "note")
    if not isinstance(meta.get('review'),str) or meta.get("review") not in REVIEWS:
        meta["review"] = "unreviewed"
    meta.setdefault("permalink", relative.removesuffix(".md"))
    return meta, body


INLINE_TAG = re.compile(r"(?<![\w&#/(\[])#([\w][\w/-]{0,63})")
MAX_TAGS = 64


def normalise_tag(value):
    tag = str(value).strip().lstrip('#').strip().lower()
    if not re.fullmatch(r"[\w][\w/-]{0,63}", tag) or tag.isdigit():
        return None
    return tag


def note_tags(meta, body):
    """Frontmatter `tags` (list, or comma/space separated string) plus inline #tags.

    Inline tags in code, or in imported external text, are data and ignored.
    """
    declared = meta.get('tags', [])
    if isinstance(declared, str):
        declared = re.split(r"[,\s]+", declared)
    tags = [normalise_tag(t) for t in declared if isinstance(t, (str, int))] if isinstance(declared, list) else []
    if not LINK_METADATA.intersection(meta):
        plain = re.sub(r"`[^`\n]*`", " ", without_fences(body))
        tags += [normalise_tag(t) for t in INLINE_TAG.findall(plain)]
    return sorted({t for t in tags if t})[:MAX_TAGS]


def space_of(meta):
    if meta.get("space") in SPACES:
        return meta["space"]
    return "agent" if str(meta.get("source", "")).startswith("agent:") else "human"


def without_fences(body):
    lines, fence, width = [], None, 0
    for line in body.splitlines():
        found = re.match(r'^\s{0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if found and found[1][0] == fence and len(found[1]) >= width and not found[2].strip():
                fence = None
        elif found:
            fence, width = found[1][0], len(found[1])
        else:
            lines.append(line)
    return '\n'.join(lines)


class Brain:
    def __init__(self, vault: Path, state: Path):
        if vault.is_symlink():
            raise ValueError("Vault cannot be a symlink")
        self.vault = vault.absolute()
        self.state = state
        self.vault.mkdir(parents=True, exist_ok=True, mode=0o700)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(state / "brain.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, path TEXT UNIQUE,
          title TEXT, kind TEXT, review TEXT, hash TEXT, meta TEXT, body TEXT);
        CREATE TABLE IF NOT EXISTS tags (note TEXT, tag TEXT, PRIMARY KEY(note, tag));
        CREATE TABLE IF NOT EXISTS links (source TEXT, target TEXT, relation TEXT,
          PRIMARY KEY(source,target,relation));
        CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED,title,body, tokenize='unicode61');
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def lock(self):
        with (self.vault / LOCK_NAME).open("a+") as stream:
            if fcntl:
                fcntl.flock(stream, fcntl.LOCK_EX)
                yield
                return
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.02)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def safe(self, relative):
        rel = Path(relative)
        if unsafe_relative(rel):
            raise ValueError("Invalid vault path")
        path = self.vault / rel
        for part in [path, *path.parents]:
            if part == self.vault:
                break
            if part.is_symlink():
                raise ValueError("Symlink in vault")
        if path.suffix != ".md":
            raise ValueError("Only Markdown notes are supported")
        return path

    def _read(self, path):
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Expected a regular file')
        if info.st_size > MAX_NOTE:
            raise ValueError("Note exceeds size limit")
        return path.read_bytes().decode('utf-8')

    def sync(self):
        with self.lock():
            return self._sync()

    def _sync(self):
        records, warnings, total = [], [], 0
        for path in sorted(self.vault.rglob("*.md")):
            if len(records) >= 10000:
                raise ValueError("Vault exceeds 10000 note projection limit")
            rel = path.relative_to(self.vault).as_posix()
            if Path(rel).parts[0] == 'Versions':
                continue
            if any(p.startswith(".") for p in Path(rel).parts):
                continue
            try:
                raw = self._read(self.safe(rel))
                total += len(raw.encode())
                if total > MAX_VAULT:
                    raise OverflowError("Vault projection exceeds 100 MB")
                meta, body = parse_note(redact(raw), rel)
                records.append((meta, body, rel, digest(raw)))
            except (OSError, ValueError) as exc:
                warnings.append({"path": rel, "error": str(exc)})
        ids = [r[0]["id"] for r in records]
        duplicate = {x for x in ids if ids.count(x) > 1}
        if duplicate:
            raise ValueError("Duplicate note IDs; fix files before sync")
        known_ids = set(ids)
        aliases = {}
        for meta, body, rel, hashed in records:
            for key in {meta['id'], meta['title'], str(meta['permalink']), rel, rel.removesuffix('.md'), Path(rel).stem}:
                aliases.setdefault(key.casefold(), set()).add(meta['id'])
        with self.db:
            self.db.execute("DELETE FROM notes")
            self.db.execute("DELETE FROM links")
            self.db.execute("DELETE FROM search")
            self.db.execute("DELETE FROM tags")
            for meta, body, rel, hashed in records:
                self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?,?,?,?)", (
                    meta['id'], rel, meta['title'], meta['type'], meta['review'], hashed,
                    json.dumps(meta, ensure_ascii=False), body))
                self.db.execute("INSERT INTO search VALUES (?,?,?)", (meta['id'], meta['title'], body))
                self.db.executemany("INSERT OR IGNORE INTO tags VALUES (?,?)", [(meta['id'], t) for t in note_tags(meta, body)])
                # External text stays readable and searchable, but its wiki
                # syntax is data. Invalid policy metadata must not fall back
                # to interpreting source text as canonical graph relations.
                if ('external_content' in meta or 'declared_only' in meta
                        or 'link_policy' in meta or 'declared_links' in meta):
                    try:
                        external = meta.get('external_content')
                        declared = meta.get('declared_only', False)
                        if (type(external) is not bool or type(declared) is not bool
                                or not (external or declared) or meta.get('link_policy') != 'declared'):
                            raise ValueError("Invalid external link policy")
                        targets = declared_links(meta.get('declared_links'))
                    except ValueError as exc:
                        warnings.append({"path": rel, "error": str(exc)})
                        continue
                    for target in targets:
                        if target in known_ids:
                            self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?)", (
                                meta['id'], target, 'relates_to'))
                        else:
                            warnings.append({"path": rel, "link": target, "error": "unresolved declared link"})
                    continue
                # Code fences contain quoted examples and transcripts, not graph assertions.
                plain = without_fences(body)
                for line in plain.splitlines():
                    relation = re.match(r"^\s*-\s+([a-z_]+)\s+\[\[", line)
                    for found in LINK.finditer(line):
                        target = found[1].split('|')[0].split('#')[0].strip()
                        matches = aliases.get(target.casefold(), set())
                        if len(matches) == 1:
                            self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?)", (
                                meta['id'], next(iter(matches)), relation[1] if relation else "relates_to"))
                        else:
                            warnings.append({"path": rel, "link": target, "error": "ambiguous" if matches else "unresolved"})
        return {"notes": len(records), "links": self.db.execute('SELECT count(*) FROM links').fetchone()[0], "warnings": warnings[:100], "warning_count": len(warnings)}

    def list(self, query="", kind="", offset=0, limit=100, space="", review="", tag=""):
        if kind and kind not in KINDS:
            raise ValueError("Unknown note kind")
        if space and space not in SPACES:
            raise ValueError("Unknown space")
        if review and review not in REVIEWS:
            raise ValueError("Unknown review status")
        if tag and normalise_tag(tag) is None:
            raise ValueError("Invalid tag")
        tag = normalise_tag(tag) if tag else ""
        offset, limit = max(0, int(offset)), min(500, max(1, int(limit)))
        args = [kind, kind, space, space, review, review, tag, tag]
        where = (f"(?='' OR n.kind=?) AND (?='' OR {SPACE_SQL}=?) AND (?='' OR n.review=?) "
                 "AND (?='' OR n.id IN (SELECT note FROM tags WHERE tag=?))")
        tokens = re.findall(r"\w+", query[:300], re.UNICODE)
        match = " AND ".join('"'+t+'"*' for t in tokens[:12])
        if tokens:
            total = self.db.execute("SELECT count(*) FROM notes n WHERE " + where + " AND n.id IN (SELECT id FROM search WHERE search MATCH ?)", (*args, match)).fetchone()[0]
            # FTS5 rank defaults to BM25; lower values are better. Keep the
            # ranking in the matched virtual-table query, not a correlated one.
            rows = [dict(r) for r in self.db.execute(f'''SELECT n.id,n.path,n.title,n.kind,n.review,n.hash,{SPACE_SQL} AS space,
                bm25(search,0,5,1) AS score, snippet(search,2,'','', ' … ',24) AS excerpt
                FROM search JOIN notes n ON n.id=search.id
                WHERE {where} AND search MATCH ? ORDER BY score,n.id LIMIT ? OFFSET ?''', (*args,match,limit,offset))]
        else:
            total = self.db.execute("SELECT count(*) FROM notes n WHERE " + where, args).fetchone()[0]
            rows = [dict(r) for r in self.db.execute(f"SELECT id,path,title,kind,review,hash,{SPACE_SQL} AS space FROM notes n WHERE " + where + " ORDER BY title,id LIMIT ? OFFSET ?", (*args, limit, offset))]
        return {"items": rows, "total": total, "offset": offset, "next": offset+len(rows) if offset+len(rows)<total else None}

    def get(self, ident):
        row = self.db.execute("SELECT * FROM notes WHERE id=?", (ident,)).fetchone()
        if not row:
            raise ValueError("Unknown note")
        result = dict(row)
        raw = self._read(self.safe(result['path']))
        meta,body=parse_note(redact(raw),result['path'])
        if meta['id']!=ident:
            raise ValueError('Note identity changed externally; sync the vault')
        result.update(content=redact(raw), hash=digest(raw), meta=meta,body=body,title=meta['title'],kind=meta['type'],review=meta['review'],space=space_of(meta))
        result['relations'] = [dict(r) for r in self.db.execute("""SELECT l.*, s.title AS source_title, t.title AS target_title
            FROM links l LEFT JOIN notes s ON s.id=l.source LEFT JOIN notes t ON t.id=l.target
            WHERE l.source=? OR l.target=?""", (ident, ident))]
        result['tags'] = [r[0] for r in self.db.execute("SELECT tag FROM tags WHERE note=? ORDER BY tag", (ident,))]
        return result

    def create(self, title, body, *, kind="note", review="proposed", source="", source_hash="", key=None, links=(), extra=None,
               external_content=False, declared_only=False):
        if not isinstance(kind,str) or kind not in KINDS or not isinstance(review,str) or review not in REVIEWS or not isinstance(title, str) or not title.strip() or len(title)>200 or not isinstance(body,str):
            raise ValueError("Invalid title, kind or review state")
        if type(external_content) is not bool or type(declared_only) is not bool:
            raise ValueError("Link policy flags must be booleans")
        if extra is not None and not isinstance(extra, dict):
            raise ValueError("Invalid note metadata")
        extra = extra or {}
        declared_policy = external_content or declared_only
        if LINK_METADATA.intersection(extra) or (declared_policy and OWNED_METADATA.intersection(extra)):
            raise ValueError("Cannot override owned external metadata")
        if declared_policy and any(not isinstance(key, str) or not re.fullmatch(r'[a-z_][a-z0-9_]*', key) for key in extra):
            raise ValueError("Invalid metadata field name")
        if external_content:
            if review != 'unreviewed':
                raise ValueError("External imports require unreviewed status")
        if declared_policy:
            links = declared_links(links)
        ident = digest(key)[:32] if key else uuid.uuid4().hex
        rel = FOLDERS[kind] + '/' + ident + '.md'
        meta = {"id": ident, "title": title, "type": kind, "permalink": f"{kind}/{ident}",
                "review": review, "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "source": source, "source_hash": source_hash, **extra}
        if declared_policy:
            meta.update(external_content=external_content, declared_only=declared_only,
                        link_policy='declared', declared_links=links)
        text = '---\n' + ''.join(k+': '+json.dumps(v, ensure_ascii=False)+'\n' for k,v in meta.items()) + '---\n\n'
        text += body
        if links:
            text += '\n\n## Relations\n' + ''.join('- relates_to [['+x+']]\n' for x in links)
        text = redact(text)
        if len(text.encode()) > MAX_NOTE:
            raise ValueError("Note exceeds size limit")
        with self.lock():
            path = self.safe(rel)
            if key and not path.exists():
                # Identity survives an Obsidian move, including moves since
                # the previous index build. Read just headers, not transcripts.
                for candidate in self.vault.rglob('*.md'):
                    relative = candidate.relative_to(self.vault).as_posix()
                    if Path(relative).parts[0] == 'Versions':
                        continue
                    try:
                        with self.safe(relative).open('r', encoding='utf-8', newline='') as stream:
                            head = stream.read(32768)
                        if parse_note(head,relative)[0]['id'] == ident:
                            return ident
                    except (OSError,ValueError):
                        continue
            if not path.exists():
                atomic_write(path, text)
        return ident

    def edit(self, ident, content, expected_hash):
        row = self.db.execute('SELECT path FROM notes WHERE id=?', (ident,)).fetchone()
        if not row or not isinstance(content, str) or len(content.encode()) > MAX_NOTE:
            raise ValueError("Invalid note or content size")
        if redact(content) != content:
            raise ValueError("Credentials are not allowed in notes")
        meta, _ = parse_note(content, row['path'])
        if meta['id'] != ident:
            raise ValueError("Note ID cannot be changed")
        with self.lock():
            path = self.safe(row['path'])
            before = self._read(path)
            if digest(before) != expected_hash:
                raise ValueError("Conflict: note changed outside this editor; reload it")
            version = self.safe(f'Versions/{ident}/{expected_hash}.md')
            if not version.exists():
                atomic_write(version, before)
            atomic_write(path, content)
        self.sync()
        return self.get(ident)

    def tags(self, space=""):
        if space and space not in SPACES:
            raise ValueError("Unknown space")
        rows = self.db.execute(f"""SELECT t.tag, count(*) AS count FROM tags t JOIN notes n ON n.id=t.note
            WHERE (?='' OR {SPACE_SQL}=?) GROUP BY t.tag ORDER BY count DESC, t.tag LIMIT 300""", (space, space))
        return {"tags": [dict(r) for r in rows]}

    def add_link(self, source, target, relation="relates_to"):
        """Accept a suggested link: append it to a note's Relations, as a versioned edit.

        Imported notes only keep wrapper-declared links, so the link is written
        on the other note when the first one is external.
        """
        if source == target or not re.fullmatch(r"[a-z_]{1,40}", relation):
            raise ValueError("Invalid link")
        notes = [self.get(source), self.get(target)]
        writable = [n for n in notes if not LINK_METADATA.intersection(n['meta'])]
        if not writable:
            raise ValueError("Both notes are imported; link them from a note of your own")
        note = writable[0]
        other = target if note['id'] == source else source
        line = f"- {relation} [[{other}]]"
        content = note['content']
        if any(l.strip() == line for l in content.splitlines()):
            return note
        if re.search(r"(?m)^## Relations\s*$", content):
            content = content.rstrip('\n') + '\n' + line + '\n'
        else:
            content = content.rstrip('\n') + '\n\n## Relations\n' + line + '\n'
        return self.edit(note['id'], content, note['hash'])

    def graph(self, query="", kind="", limit=500, space=""):
        result = self.list(query, kind, limit=limit, space=space)
        nodes = result['items']
        owners = self.project_owners()
        counts = {}
        for owner in owners.values():counts[owner] = counts.get(owner,0)+1
        for node in nodes:
            node['project_id'] = owners.get(node['id'])
            if node['kind']=='project':node['project_materials']=counts.get(node['id'],1)
        ids = {n['id'] for n in nodes}
        links = [dict(r) for r in self.db.execute('SELECT * FROM links') if r['source'] in ids and r['target'] in ids]
        spaces = {r[0]: r[1] for r in self.db.execute(f"SELECT {SPACE_SQL}, count(*) FROM notes n GROUP BY 1")}
        return {"nodes": nodes, "links": links, "total": result['total'], "limited": result['total'] > len(nodes),
                "spaces": {s: spaces.get(s, 0) for s in sorted(SPACES)},
                "projects": dict(self.db.execute("SELECT id,title FROM notes WHERE kind='project' LIMIT 500").fetchall())}

    def project_owners(self):
        """Group only by explicit metadata or an unambiguous direct project link."""
        rows = list(self.db.execute('''SELECT id,kind,json_extract(meta,'$.source') AS source,
            json_extract(meta,'$.project') AS project FROM notes LIMIT 10001'''))
        if len(rows) > 10000:
            raise ValueError('Knowledge mapping exceeds node limit')
        projects = {r['id'] for r in rows if r['kind'] == 'project'}
        by_source = {}
        for row in rows:
            if row['kind'] == 'project' and isinstance(row['source'], str) and row['source']:
                by_source.setdefault(row['source'], []).append(row['id'])
        owners = {ident: ident for ident in projects}
        for row in rows:
            if row['id'] in owners or not isinstance(row['project'], str):
                continue
            matches = by_source.get(path_uri(row['project']), [])
            if len(matches) == 1:
                owners[row['id']] = matches[0]
        direct = {}
        for edge in self.db.execute("SELECT source,target FROM links WHERE relation IN ('relates_to','belongs_to')"):
            if edge['target'] in projects and edge['source'] not in projects:
                direct.setdefault(edge['source'], set()).add(edge['target'])
        for note, targets in direct.items():
            if note not in owners and len(targets) == 1:
                owners[note] = next(iter(targets))
        return owners

    def import_markdown(self, path: Path, *, split=False, links=()):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != '.md':
            raise ValueError("Import requires a regular Markdown file")
        raw = self._read(path)
        hashed = digest(raw)
        normalized_hash = digest(raw.replace('\r\n','\n'))
        source = path.resolve().as_uri()
        title = path.stem
        ident = self.create(title, raw, kind="source", source=source, source_hash=hashed,
                            review="unreviewed", key=source+hashed, links=links,
                            external_content=True,
                            extra={'source_id':digest(source)[:32], 'raw_sha256':hashed,'normalized_sha256':normalized_hash,'parser':'markdown-v1'})
        children = []
        if split:
            sections = re.split(r'(?m)^## (.+)\n', raw)
            for i in range(1, min(len(sections)-1, 201), 2):
                if not sections[i+1].strip():
                    continue
                children.append(self.create(sections[i][:160], sections[i+1], kind="research",
                    source=source, source_hash=hashed, key=source+hashed+str(i),
                    review="unreviewed", external_content=True, links=[ident, *links], extra={"source_section": sections[i]}))
        section_count = len(re.findall(r'(?m)^## .+\n',raw)) if split else 0
        return {"id": ident, "sections": children, 'sections_total':section_count,'sections_omitted':max(0,section_count-len(children))}

    def backup(self):
        target = self.state / 'backups' / (time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.zip')
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        manifest = {}
        with self.lock(), zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            total = 0
            for p in sorted(self.vault.rglob('*.md')):
                rel = p.relative_to(self.vault).as_posix()
                raw = self._read(self.safe(rel)).encode()
                total += len(raw)
                if total > MAX_VAULT:
                    raise ValueError('Backup exceeds size limit')
                manifest[rel] = digest(raw)
                archive.writestr(rel, raw)
            archive.writestr('manifest.json', json.dumps(manifest))
        os.chmod(target, 0o600)
        return {"path": str(target), "notes": len(manifest), "sha256": digest(target.read_bytes()), "scope":"Markdown notes and Versions only; original external assets are not included"}


def restore_backup(archive_path, destination):
    """Restore only into a new directory; verify every member before any write."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Restore destination must not exist')
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if len(entries)>10001 or sum(x.file_size for x in entries)>MAX_VAULT+2_000_000:
            raise ValueError('Backup exceeds limits')
        names = [x.filename for x in entries]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        manifest = json.loads(archive.read('manifest.json'))
        if set(names) != {*manifest, 'manifest.json'}:
            raise ValueError('Unexpected backup member')
        data = {}
        for name, expected in manifest.items():
            path = Path(name)
            if unsafe_relative(path) or path.suffix != '.md':
                raise ValueError('Unsafe backup path')
            raw = archive.read(name)
            if len(raw)>MAX_NOTE or digest(raw)!=expected:
                raise ValueError('Backup checksum mismatch')
            data[name] = raw.decode('utf-8')
    destination.mkdir(parents=True, mode=0o700)
    for name, content in data.items():
        atomic_write(destination/name, content)
    return {"path": str(destination), "notes": len(data)}
