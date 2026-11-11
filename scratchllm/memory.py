"""Local, session-scoped episodic memory. SQLite FTS; no embeddings or model fitting."""
import json
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from threading import RLock

from .dialogue import message_text

SESSION = re.compile(r'[A-Za-z0-9_-]{1,64}')
STOPWORDS = set('a an and are as at be by can do for from how i in is it me my of on or please that the this to was we what when where which who why with you your'.split())


class MemoryStore:
    def __init__(self, path, max_episodes=10000):
        if max_episodes < 1:
            raise ValueError('Memory capacity must be positive')
        self.path, self.max_episodes = Path(path), max_episodes
        self.lock, self.initialized = RLock(), False

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            with self.lock:
                if not self.initialized:
                    db.execute('PRAGMA journal_mode=WAL')
                    db.executescript('''
                        CREATE TABLE IF NOT EXISTS sessions (
                            id TEXT PRIMARY KEY, goal TEXT NOT NULL, revision INTEGER NOT NULL,
                            updated REAL NOT NULL);
                        CREATE TABLE IF NOT EXISTS episodes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                            user_text TEXT NOT NULL, assistant_text TEXT NOT NULL,
                            checkpoint TEXT NOT NULL, diagnostics TEXT NOT NULL,
                            feedback INTEGER, feedback_note TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
                        CREATE INDEX IF NOT EXISTS episodes_session ON episodes(session_id, id);
                        CREATE VIRTUAL TABLE IF NOT EXISTS episode_search USING fts5(
                            user_text, assistant_text, feedback_note, content='episodes', content_rowid='id');
                        CREATE TRIGGER IF NOT EXISTS episode_insert AFTER INSERT ON episodes BEGIN
                            INSERT INTO episode_search(rowid,user_text,assistant_text,feedback_note)
                                VALUES (new.id,new.user_text,new.assistant_text,new.feedback_note);
                        END;
                        CREATE TRIGGER IF NOT EXISTS episode_delete AFTER DELETE ON episodes BEGIN
                            INSERT INTO episode_search(episode_search,rowid,user_text,assistant_text,feedback_note)
                                VALUES ('delete',old.id,old.user_text,old.assistant_text,old.feedback_note);
                        END;
                        CREATE TRIGGER IF NOT EXISTS episode_update AFTER UPDATE ON episodes BEGIN
                            INSERT INTO episode_search(episode_search,rowid,user_text,assistant_text,feedback_note)
                                VALUES ('delete',old.id,old.user_text,old.assistant_text,old.feedback_note);
                            INSERT INTO episode_search(rowid,user_text,assistant_text,feedback_note)
                                VALUES (new.id,new.user_text,new.assistant_text,new.feedback_note);
                        END;
                    ''')
                    self.initialized = True
            with db:
                yield db

    @staticmethod
    def check_session(session_id):
        if not SESSION.fullmatch(session_id):
            raise ValueError('Session ID must be 1–64 letters, digits, underscores or hyphens')

    @staticmethod
    def episode(row):
        value = dict(row)
        value['diagnostics'] = json.loads(value['diagnostics'])
        return value

    def snapshot(self, session_id):
        self.check_session(session_id)
        with self.connect() as db:
            db.execute('BEGIN')
            session = db.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
            counts = db.execute('SELECT COUNT(*) AS total, COUNT(CASE WHEN feedback=1 THEN 1 END) AS helpful, '
                                'COUNT(CASE WHEN feedback=-1 THEN 1 END) AS unhelpful FROM episodes WHERE session_id=?', (session_id,)).fetchone()
            recent = db.execute('SELECT * FROM episodes WHERE session_id=? ORDER BY id DESC LIMIT 8', (session_id,)).fetchall()
        return dict(session_id=session_id, goal=session['goal'] if session else '', revision=session['revision'] if session else 0,
                    feedback=dict(counts), recent=[self.episode(row) for row in reversed(recent)], capacity=self.max_episodes)

    def retrieve(self, session_id, query, exclude=(), limit=4):
        self.check_session(session_id)
        words = list(dict.fromkeys(word for word in re.findall(r'[^\W_]+', query.casefold())
                                   if word not in STOPWORDS and len(word) > 1))[:16]
        if not words:
            return []
        # Only quoted lexical terms reach MATCH; user-supplied operators are never executed.
        match = ' OR '.join('"' + word + '"' for word in words)
        omitted = ','.join('?' for _ in exclude) or 'NULL'
        exclusion = f'AND e.id NOT IN ({omitted})' if exclude else ''
        with self.connect() as db:
            rows = db.execute(f'''SELECT e.* FROM episode_search JOIN episodes e ON e.id=episode_search.rowid
                WHERE episode_search MATCH ? AND e.session_id=? AND COALESCE(e.feedback,0)>=0 {exclusion}
                ORDER BY bm25(episode_search), e.id DESC LIMIT ?''', (match, session_id, *exclude, min(8, max(1, limit)))).fetchall()
        return [self.episode(row) for row in rows]

    def record(self, session_id, prompt, answer, checkpoint, diagnostics, goal, revision):
        self.check_session(session_id)
        prompt, answer, goal = message_text(prompt), message_text(answer), message_text(goal)
        if not prompt or not answer:
            raise ValueError('Only complete nonempty exchanges can enter memory')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM sessions WHERE id=?', (session_id,)).fetchone()
            if (row['revision'] if row else 0) != revision:
                raise ValueError('Memory changed during generation; retry with the current session')
            db.execute('INSERT INTO sessions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET '
                       'goal=excluded.goal, revision=excluded.revision, updated=excluded.updated',
                       (session_id, goal, revision + 1, time.time()))
            entry = db.execute('INSERT INTO episodes(session_id,user_text,assistant_text,checkpoint,diagnostics,created) '
                               'VALUES (?,?,?,?,?,?)', (session_id, prompt, answer, checkpoint, json.dumps(diagnostics), time.time()))
            db.execute('DELETE FROM episodes WHERE session_id=? AND id IN '
                       '(SELECT id FROM episodes WHERE session_id=? ORDER BY id DESC LIMIT -1 OFFSET ?)',
                       (session_id, session_id, self.max_episodes))
            return entry.lastrowid

    def feedback(self, session_id, episode_id, helpful, note=''):
        self.check_session(session_id)
        with self.connect() as db:
            result = db.execute('UPDATE episodes SET feedback=?,feedback_note=? WHERE session_id=? AND id=?',
                                (1 if helpful else -1, message_text(note), session_id, episode_id))
            if result.rowcount != 1:
                raise ValueError('Memory entry does not belong to this session')
            db.execute('UPDATE sessions SET revision=revision+1, updated=? WHERE id=?', (time.time(), session_id))

    def forget(self, session_id):
        self.check_session(session_id)
        with self.connect() as db:
            # Keep a revision tombstone, so a simultaneous inference cannot recreate deleted memories.
            db.execute('INSERT INTO sessions VALUES (?, ?, 1, ?) ON CONFLICT(id) DO UPDATE SET '
                       "goal='', revision=revision+1, updated=excluded.updated", (session_id, '', time.time()))
            db.execute('DELETE FROM episodes WHERE session_id=?', (session_id,))
