"""Small atomic status snapshots for long-running crawl operations."""
import asyncio
import json
import time
from pathlib import Path


class CrawlProgress:
    def __init__(self, path, budget):
        self.path = Path(path) if path else None
        self.state = dict(phase="starting", detail="Starting crawler", url=None, host=None,
                          attempted=0, completed=0, saved=0, skipped=0, failed=0,
                          blocked_resources=0, filtered_links=0,
                          saved_pages=0, duplicates=0, too_short=0, characters=0, text_bytes=0, sources={},
                          queued=0, budget=budget, started_at=time.time(), phase_started_at=time.time())

    def update(self, phase=None, detail=None, **fields):
        if phase and phase != self.state["phase"]:
            self.state["phase_started_at"] = time.time()
        if phase:
            self.state["phase"] = phase
        if detail is not None:
            self.state["detail"] = detail
            print(f"[{time.strftime('%H:%M:%S')}] {detail}", flush=True)
        self.state.update(fields, updated_at=time.time(), heartbeat_at=time.time())
        self.write()

    def count(self, key, amount=1):
        self.state[key] = self.state.get(key, 0) + amount
        host = self.state.get('host')
        if host:
            source = self.state['sources'].setdefault(host, {})
            source[key] = source.get(key, 0) + amount

    def counts(self):
        keys = ('attempted', 'completed', 'saved', 'saved_pages', 'skipped', 'failed', 'blocked_resources',
                'filtered_links', 'duplicates', 'too_short', 'characters', 'text_bytes', 'sources')
        return {key: self.state[key] for key in keys}

    def write(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state) + "\n")
            tmp.replace(self.path)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(10)
            self.state["heartbeat_at"] = time.time()
            self.write()
            elapsed = int(time.time() - self.state["phase_started_at"])
            print(f"[{time.strftime('%H:%M:%S')}] ACTIVE {self.state['phase']} ({elapsed}s): "
                  f"{self.state['url'] or self.state['detail']}", flush=True)
