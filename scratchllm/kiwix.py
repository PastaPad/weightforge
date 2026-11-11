"""Kiwix Wikipedia archives: verified downloads, metadata, and offline article reading."""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from threading import Event
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from libzim.reader import Archive

from .frontier import corpus_lock
from .corpus import content_hash
from .links import TEXT_POLICY_VERSION
from .progress import CrawlProgress
from .sources import WIKIPEDIA_NAMESPACES
from .text_policy import EXTRACTION_VERSION

BASE_URL = 'https://download.kiwix.org/zim/wikipedia/'
DEFAULT_FILE = 'wikipedia_en_top_nopic_2026-09.zim'
DEFAULT_PATH = 'data/kiwix/' + DEFAULT_FILE
ARCHIVE_NAME = re.compile(r'wikipedia_(?:en|en-simple)_[a-z0-9_-]+_\d{4}-\d{2}\.zim')


def download_name(url):
    parsed = urlsplit(url)
    name = parsed.path.removeprefix('/zim/wikipedia/')
    if (parsed.scheme != 'https' or parsed.netloc != 'download.kiwix.org' or parsed.query or parsed.fragment
            or not parsed.path.startswith('/zim/wikipedia/') or not ARCHIVE_NAME.fullmatch(name)):
        raise ValueError('Choose an English Wikipedia .zim URL from https://download.kiwix.org/zim/wikipedia/')
    return name


def catalog():
    with urlopen(BASE_URL, timeout=20) as response:
        html = response.read(2_000_000).decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    newest = {}
    for anchor in soup.select('a[href]'):
        name = anchor['href']
        if not ARCHIVE_NAME.fullmatch(name) or '_nopic_' not in name:
            continue
        series = name.rsplit('_', 1)[0]
        tail = str(anchor.next_sibling or '').split()
        row = dict(name=name, url=BASE_URL + name, size=tail[-1] if tail else '?')
        if series not in newest or name > newest[series]['name']:
            newest[series] = row
    return sorted(newest.values(), key=lambda row: (not row['name'].startswith('wikipedia_en_top_nopic_'), row['name']))


def open_archive(path, *, partial=False):
    path = Path(path).resolve()
    if not path.is_file() or not (path.suffix.lower() == '.zim' or partial and path.name.endswith('.zim.part')):
        raise ValueError('Select a complete local Wikipedia .zim file (download it first)')
    try:
        archive = Archive(path)
        metadata = {key: archive.get_metadata(key).decode('utf-8') for key in
                    ('Title', 'Name', 'Language', 'Date', 'Source', 'Creator', 'Publisher', 'Flavour') if key in archive.metadata_keys}
    except (RuntimeError, UnicodeError) as exc:
        raise ValueError(f'Cannot read Wikipedia ZIM: {exc}') from exc
    source = metadata.get('Source', '').removeprefix('https://').removeprefix('http://').rstrip('/')
    if source not in {'en.wikipedia.org', 'simple.wikipedia.org'} or metadata.get('Creator') != 'Wikipedia':
        raise ValueError('Use a Kiwix English or Simple English Wikipedia archive')
    stat = path.stat()
    info = dict(path=str(path), uuid=str(archive.uuid), checksum=archive.checksum, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, entries=archive.entry_count, article_entries=archive.article_count,
                source=source, metadata=metadata)
    return archive, info


def archive_info(path):
    _, info = open_archive(path)
    return info


def permutation(length, seed):
    """A stable, constant-memory spread across the archive, rather than link-following."""
    if length < 1:
        raise ValueError('Archive contains no entries')
    rng = random.Random(seed)
    start, stride = rng.randrange(length), rng.randrange(1, length + 1)
    while math.gcd(stride, length) != 1:
        stride = stride % length + 1
    return start, stride


class _HeadComplete(Exception):
    pass


class _RedirectHeader(HTMLParser):
    redirect = False

    def handle_starttag(self, tag, attributes):
        if tag == 'meta' and (dict(attributes).get('http-equiv') or '').casefold() == 'refresh':
            self.redirect = True
            raise _HeadComplete
        if tag == 'body':
            raise _HeadComplete

    def handle_endtag(self, tag):
        if tag == 'head':
            raise _HeadComplete


def html_redirect(html):
    parser = _RedirectHeader()
    try:
        parser.feed(html)
    except _HeadComplete:
        pass
    return parser.redirect


def article_entry(archive, index, source):
    entry = archive._get_entry_by_id(index)
    if entry.is_redirect:
        return None
    path = entry.path
    if not archive.has_new_namespace_scheme:
        if not path.startswith('A/'):
            return None
        path = path[2:]
    if (not path or path.startswith(('-', '_', '/')) or path in {'index', 'index.html', 'Main_Page'}
            or path.lstrip(':').split(':', 1)[0].replace('_', ' ').casefold() in WIKIPEDIA_NAMESPACES):
        return None
    item = entry.get_item()
    if item.mimetype.split(';', 1)[0].lower() not in {'text/html', 'application/xhtml+xml'}:
        return None  # Never read media/attachment bytes.
    if item.size > 20_000_000:
        return None
    html = bytes(item.content).decode('utf-8')
    if html_redirect(html):
        return None  # mwoffliner stores fragment redirects as HTML refresh stubs.
    url = 'https://' + source + '/wiki/' + quote(path, safe='/():,')
    return entry, url, html


async def import_wikipedia(args, progress, frontier, out, hashes, corpus_digest, archive, info):
    from .crawl import extract_documents

    state = frontier.get('kiwix')
    whole_archive = frontier.settings.get('wikipedia_all', False)
    start, stride = permutation(state['entries'], args.wikipedia_seed)
    progress.update('importing_zim', f"IMPORT Wikipedia from {info['path']}", host='wikipedia.org', url=None,
                    mode='kiwix', archive=info, archive_cursor=state['cursor'], browser_tabs=[], wikipedia_all=whole_archive)

    def checkpoint():
        out.flush()
        os.fsync(out.fileno())
        frontier.checkpoint(progress.counts(), out.tell(), corpus_digest.hexdigest(), kiwix=state)
        units = progress.state['completed']
        if whole_archive:
            units += state['cursor'] - progress.state['sources']['wikipedia.org'].get('completed', 0)
        progress.update(archive_cursor=state['cursor'], queued=frontier.pending(), completed_units=units)

    try:
        while state['cursor'] < state['entries'] and (whole_archive or progress.state['sources']['wikipedia.org'].get('attempted', 0) < args.pages_per_site):
            await asyncio.sleep(0)  # Deliver stop signals between articles, before changing the cursor.
            index = (start + stride * state['cursor']) % state['entries']
            candidate = article_entry(archive, index, info['source'])
            if candidate is None:
                state['cursor'] += 1
                if state['cursor'] % 100 == 0:
                    checkpoint()
                continue
            entry, url, html = candidate
            _, documents, _ = extract_documents(html, url, discover=False)
            saved = 0
            for document in documents:
                text = document['text']
                if len(text) < args.min_chars:
                    progress.count('too_short')
                    continue
                digest = content_hash(text)
                if digest in hashes:
                    progress.count('duplicates')
                    continue
                row = dict(document, title=entry.title, page_url=url, host='wikipedia.org',
                           sha256=hashlib.sha256(text.encode()).hexdigest(), transport='kiwix-zim',
                           extraction=EXTRACTION_VERSION, text_policy=TEXT_POLICY_VERSION,
                           archive=dict(path=info['path'], uuid=info['uuid'], checksum=info['checksum'],
                                        entry=entry.path, date=info['metadata'].get('Date'), source=info['source']),
                           fetched_at=datetime.now(timezone.utc).isoformat())
                payload = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                out.write(payload)
                corpus_digest.update(payload)
                hashes.add(digest)
                progress.count('saved')
                progress.count('characters', len(text))
                progress.count('text_bytes', len(text.encode()))
                saved += 1
            progress.count('attempted')
            progress.count('completed')
            progress.count('saved_pages' if saved else 'skipped')
            state['cursor'] += 1
            progress.update(detail=f"{'SAVE' if saved else 'SKIP'} ZIM {entry.title} ({saved} new article records)", url=url)
            checkpoint()
    finally:
        checkpoint()
    progress.update(detail=f"WIKIPEDIA DONE: {progress.state['sources']['wikipedia.org'].get('saved', 0)} unique articles; "
                    f"{state['cursor']}/{state['entries']} archive entries examined")


def download_archive(url, output, progress, stopping):
    name = download_name(url)
    output = Path(output)
    if output.suffix.lower() != '.zim':
        raise ValueError('Download output must end in .zim')
    output.parent.mkdir(parents=True, exist_ok=True)
    part, receipt = Path(str(output) + '.part'), Path(str(output) + '.download.json')
    def save_receipt(data):
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(receipt)
    with corpus_lock(output):
        progress.update('checking_download', 'Reading Kiwix archive checksum and size', url=url, mode='kiwix_download')
        with urlopen(url + '.sha256', timeout=30) as response:
            checksum_line = response.read(1024).decode('ascii').strip().split()
        if len(checksum_line) != 2 or not re.fullmatch(r'[0-9a-fA-F]{64}', checksum_line[0]) or checksum_line[1].lstrip('*') != name:
            raise ValueError('Kiwix checksum response did not match the selected archive')
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            size = int(response.headers['Content-Length'])
        expected = dict(url=url, sha256=checksum_line[0].lower(), size=size)
        if size <= 0:
            raise ValueError('Kiwix returned an invalid archive size')
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        if previous is not None and {key: previous.get(key) for key in expected} != expected:
            raise ValueError('This download path belongs to a different archive; choose a fresh file')
        if part.exists() and not receipt.exists():
            raise ValueError('An unrecognized partial file exists; choose a fresh download path')
        offset = output.stat().st_size if output.exists() else part.stat().st_size if part.exists() else 0
        if offset > size or (output.exists() and offset != size):
            raise ValueError('Existing archive size differs; choose a fresh download path')
        if not output.exists():
            if shutil.disk_usage(output.parent).free < size - offset + 100 * 1024**2:
                raise ValueError(f'Insufficient free disk space: download needs {(size-offset)/1024**3:.2f} more GiB')
            save_receipt(expected)
            if offset < size and not stopping.is_set():
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with urlopen(Request(url, headers=headers), timeout=30) as response:
                    if offset and (response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{size-1}/{size}'):
                        raise ValueError('Server did not honor the download resume range; partial file retained')
                    with part.open('ab') as dest:
                        last_update = 0.0
                        progress.update('downloading', f'DOWNLOAD {name}', downloaded_bytes=offset, total_bytes=size)
                        while not stopping.is_set():
                            block = response.read(min(1024 * 1024, size - offset + 1))
                            if not block:
                                break
                            if offset + len(block) > size:
                                raise ValueError('Download exceeds its advertised size')
                            dest.write(block)
                            offset += len(block)
                            if time.monotonic() - last_update >= 1:
                                progress.update(downloaded_bytes=offset)
                                last_update = time.monotonic()
                        dest.flush()
                        os.fsync(dest.fileno())
            if stopping.is_set():
                progress.update('stopped', 'Download stopped; use Download again to resume', downloaded_bytes=offset, total_bytes=size)
                return
            if offset != size:
                raise ValueError('Download ended early; use Download again to resume')
        progress.update('verifying_download', 'Verifying the complete archive SHA-256', downloaded_bytes=size, total_bytes=size, verified_bytes=0)
        digest, checked, last_update = hashlib.sha256(), 0, 0.0
        with (output if output.exists() else part).open('rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                if stopping.is_set():
                    progress.update('stopped', 'Verification stopped; use Download again to finish verification')
                    return
                digest.update(block)
                checked += len(block)
                if time.monotonic() - last_update >= 1:
                    progress.update(verified_bytes=checked)
                    last_update = time.monotonic()
        if digest.hexdigest() != expected['sha256']:
            raise ValueError('Archive checksum mismatch; choose a fresh download path')
        _, info = open_archive(output if output.exists() else part, partial=not output.exists())
        if not output.exists():
            # Exclusive publication never overwrites an unrelated file created during download.
            os.link(part, output)
            part.unlink()
        save_receipt(dict(expected, verified=True, uuid=info['uuid']))
        progress.update('completed', f'DONE verified Kiwix archive: {output}', downloaded_bytes=size, verified_bytes=size, output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=BASE_URL + DEFAULT_FILE)
    parser.add_argument('--output', default=DEFAULT_PATH)
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    progress, stopping = CrawlProgress(args.progress_file, 0), Event()
    def stop(*_):
        stopping.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        download_archive(args.url, args.output, progress, stopping)
    except Exception as exc:
        progress.update('failed', f'FAILED {exc}')
        raise


if __name__ == '__main__':
    main()
