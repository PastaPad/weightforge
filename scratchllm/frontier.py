"""Durable crawl queue. Only the in-flight page may need another visit on resume."""
import hashlib
import fcntl
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from .text_policy import EXTRACTION_VERSION
from .sources import seed_url


def state_path(output):
    return Path(str(output) + ".crawl.sqlite3")


@contextmanager
def corpus_lock(output):
    path = Path(str(Path(output).resolve()) + '.crawl.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This corpus is already being collected by another worker') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prefix_digest(path, size):
    digest = hashlib.sha256()
    if size:
        with Path(path).open('rb') as stream:
            while size:
                block = stream.read(min(size, 1024 * 1024))
                if not block:
                    raise ValueError("Corpus is shorter than its saved crawl state")
                digest.update(block)
                size -= len(block)
    return digest.hexdigest()


def read_state(output):
    path = state_path(output).resolve()
    if not path.is_file():
        raise ValueError("No saved crawl queue for this corpus. Start a fresh collection first.")
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0.05)) as db:
            data = {key: json.loads(value) for key, value in db.execute('SELECT key, value FROM metadata')}
            data['pending'] = db.execute("SELECT count(*) FROM pages WHERE status != 'done'").fetchone()[0]
    except sqlite3.Error as exc:
        raise ValueError(f'Cannot read saved crawl queue: {exc}') from exc
    if data.get('extraction') != EXTRACTION_VERSION or data.get('version') != 1:
        raise ValueError("Saved crawl queue uses a different text format; start a fresh collection")
    if not {'output', 'settings', 'stats', 'corpus_size', 'corpus_digest'} <= data.keys():
        raise ValueError('Saved crawl queue is incomplete')
    if data.get('kiwix'):
        data['pending'] += max(0, data['kiwix']['entries'] - data['kiwix']['cursor'])
    return data


class Frontier:
    def __init__(self, output, settings, resume=False):
        self.output = Path(output).resolve()
        self.path = state_path(self.output)
        if resume:
            saved = read_state(self.output)
            if saved['output'] != str(self.output):
                raise ValueError("Saved queue belongs to a different corpus path")
            if prefix_digest(self.output, saved['corpus_size']) != saved['corpus_digest']:
                raise ValueError("Corpus changed since collection; use a fresh corpus instead of this queue")
            original = saved['settings']
            if any(settings[key] != value for key, value in original.items() if key not in {'pages_per_site', 'wikipedia_all'}):
                raise ValueError("Resume requires the saved collection settings")
            settings['pages_per_site'] = max(settings['pages_per_site'], original['pages_per_site'])
            if original.get('wikipedia_all'):
                settings['wikipedia_all'] = True
            self.stats = saved['stats']
        else:
            if self.path.exists():
                raise ValueError("A saved crawl queue already exists. Use Resume collection or a fresh corpus file.")
            self.stats = {}
        self.settings = settings
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, host TEXT NOT NULL,
                depth INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending');
            CREATE INDEX IF NOT EXISTS pending_pages ON pages(host, status, id);
            CREATE INDEX IF NOT EXISTS page_states ON pages(status);
        ''')
        with self.db:
            self.put('version', 1)
            self.put('extraction', EXTRACTION_VERSION)
            self.put('output', str(self.output))
            self.put('settings', settings)
            if resume:
                self.db.execute("UPDATE pages SET status='pending' WHERE status='active'")
            else:
                for host in settings['hosts']:
                    if host == 'wikipedia.org' and settings.get('wikipedia_archive'):
                        self.put('kiwix', dict(cursor=0, entries=settings['wikipedia_archive']['entries']))
                        continue
                    self.db.execute('INSERT INTO pages(url,host,depth) VALUES(?,?,0)', (seed_url(host), host))
                size = self.output.stat().st_size if self.output.exists() else 0
                self.put('corpus_size', size)
                self.put('corpus_digest', prefix_digest(self.output, size))
                self.put('stats', {})

    def put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, json.dumps(value)))

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def next(self, host):
        row = self.db.execute("SELECT url,depth FROM pages WHERE host=? AND status='pending' ORDER BY id LIMIT 1", (host,)).fetchone()
        if row:
            with self.db:
                self.db.execute("UPDATE pages SET status='active' WHERE url=?", (row[0],))
        return row

    def discover(self, host, links):
        count = self.db.execute('SELECT count(*) FROM pages WHERE host=?', (host,)).fetchone()[0]
        limit = max(0, self.settings['pages_per_site'] * 50 - count)
        with self.db:
            for url, depth in links:
                if not limit:
                    break
                inserted = self.db.execute('INSERT OR IGNORE INTO pages(url,host,depth) VALUES(?,?,?)', (url, host, depth)).rowcount
                limit -= inserted

    def pending(self):
        kiwix = self.get('kiwix', {})
        return (self.db.execute("SELECT count(*) FROM pages WHERE status != 'done'").fetchone()[0]
                + max(0, kiwix.get('entries', 0) - kiwix.get('cursor', 0)))

    def checkpoint(self, stats, size, digest, finished=None, release=None, kiwix=None):
        with self.db:
            if finished:
                self.db.execute("UPDATE pages SET status='done' WHERE url=?", (finished,))
            if release:
                self.db.execute("UPDATE pages SET status='pending' WHERE url=?", (release,))
            self.put('stats', stats)
            self.put('corpus_size', size)
            self.put('corpus_digest', digest)
            if kiwix is not None:
                self.put('kiwix', kiwix)

    def close(self):
        self.db.close()
