"""Resumable text collection: Camoufox websites and offline Kiwix Wikipedia archives."""
import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from camoufox.async_api import AsyncCamoufox
from curl_cffi import requests as curl_requests

from .progress import CrawlProgress
from .text_policy import EXTRACTION_VERSION, attachment_url, html_headers, page_url
from .corpus import clean, content_hash
from .frontier import Frontier, corpus_lock, read_state
from .links import TEXT_POLICY_VERSION
from .sources import HOSTS, WIKI_HOSTS, allowed_host, wikipedia_article
from .kiwix import DEFAULT_PATH, open_archive, import_wikipedia

AGENT = "ScratchLLMCorpusBot"


def canonical(url, host):
    p = urlsplit(url)
    if p.scheme not in {"http", "https"} or not allowed_host(p.hostname, host):
        return None
    if p.username or p.password or p.port not in {None, 80, 443}:
        return None
    if not page_url(url):
        return None
    if host == 'wikipedia.org':
        title = wikipedia_article(url)
        if title is None:
            return None
        return 'https://en.wikipedia.org/wiki/' + quote(title.replace(' ', '_'), safe='/():,')
    # Keep queries: wiki article URLs can use ?title=... . Fragments aren't pages.
    return urlunsplit((p.scheme, p.netloc.lower(), p.path or "/", p.query, ""))


def robots(origin, ignore=False):
    url = origin + "/robots.txt"
    parser = RobotFileParser(url)
    if ignore:
        parser.parse([])
        return parser
    try:
        with urlopen(Request(url, headers={"User-Agent": AGENT}), timeout=20) as response:
            parser.parse(response.read(1_000_000).decode("utf-8", errors="replace").splitlines())
    except HTTPError as exc:
        if exc.code in (404, 410):
            parser.parse([])
        else:
            raise RuntimeError(f"robots.txt returned HTTP {exc.code}; skipping origin") from exc
    return parser


POST_SELECTOR = ".post-message, .post .body, .postMessage, .post-body, .post_body, blockquote.body, .body[id^='m']"
WIKI_SELECTOR = "#mw-content-text, .mw-parser-output"
BOARD_SELECTOR = POST_SELECTOR + ", .boardlist, .boardList, .board-list"
CRAWL_SETTINGS = ("hosts", "pages_per_site", "depth", "delay", "headed", "ignore_robots", "challenge_wait",
                  "render_wait_ms", "min_chars", "min_post_chars", "profile", "wikipedia_zim", "wikipedia_seed", "wikipedia_all")


def post_identity(root, url):
    ancestors = [root, *root.parents]
    thread_id = post_id = None
    for node in ancestors:
        identifier = node.get('id', '')
        thread = re.fullmatch(r'(?:thread[_-]?|t)(\d+)', identifier)
        post = re.fullmatch(r'(?:post|reply|op|p|m)[_-]?(\d+)', identifier)
        if thread and thread_id is None:
            thread_id = thread[1]
        if post and post_id is None:
            post_id = post[1]
    parsed = urlsplit(url or '')
    # Cross-board indexes need the post's own permalink, not the index URL, to
    # identify its board/thread. Only inspect surrounding post metadata links.
    for node in ancestors[1:]:
        if not ({'post', 'op', 'reply', 'post-container', 'postContainer'} & set(node.get('class', []))):
            continue
        for anchor in node.select('a[href]'):
            if root in anchor.parents:
                continue
            candidate = urlsplit(urljoin(url or '', anchor['href']))
            if candidate.scheme not in {'http', 'https'} or (candidate.hostname or '').removeprefix('www.') != (parsed.hostname or '').removeprefix('www.'):
                continue
            match = re.search(r'/([^/]+)/(?:res|thread)/(\d+)(?:\.html)?/?$', candidate.path)
            if match:
                post_id = post_id or (candidate.fragment if candidate.fragment.isdigit() else None)
                group = f"{(parsed.hostname or '').removeprefix('www.')}/{match[1]}/thread/{match[2]}"
                permalink = urlunsplit((candidate.scheme, candidate.netloc, candidate.path, candidate.query, post_id or ''))
                return dict(url=permalink, group=group, post_id=post_id)
        break
    match = re.search(r'/(?:res|thread)/(\d+)', parsed.path)
    thread_id = thread_id or (match[1] if match else None)
    board = parsed.path.strip('/').split('/')[0]
    group = f"{(parsed.hostname or '').removeprefix('www.')}/{board}/thread/{thread_id}" if thread_id else url
    permalink = (url + '#' + post_id) if url and post_id else url
    return dict(url=permalink, group=group, post_id=post_id)


def _extract(html, url=None, discover=True):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    links = []
    for anchor in soup.select('a[href]:not([download])') if discover else []:
        try:
            if page_url(urljoin(url or '', anchor['href'])):
                links.append(anchor['href'])
        except ValueError:
            continue
    identities = {id(root): dict(post_identity(root, url), truncated=bool(root.select_one('.viewfulltext, .abbreviated, .toolong, .too-long')))
                  for root in soup.select(POST_SELECTOR)}
    for node in soup.select("script, style, nav, header, footer, input, button, textarea, select, option, label, legend, noscript, aside, "
                            "audio, video, source, track, img, picture, svg, canvas, iframe, embed, object, "
                            ".mw-editsection, .toc, .navbox, .infobox, .metadata, .catlinks, .printfooter, "
                            ".reflist, .references, sup.reference, figure, .thumb, .postInfo, .postinfo, "
                            ".noexcerpt, .sidebar, .navbar, .hatnote, .ambox, .page-issue, .gallery, "
                            "[role='navigation'], [role='button'], [aria-hidden='true'], [hidden], .mw-collapsible-toggle, .mw-editsection-like, "
                            ".post-info, .post-quoters, .post-links, .post-file, .post-files, .files, .file, .fileinfo, .file-info, .fileText, .file-text, "
                            ".post-controls, .post-actions, .post-footer, .post-header, .post-edited, .edited, .edit-info, "
                            ".abbreviated, .toolong, .too-long, .post-expand, .expand-post, .expand-button, "
                            "div.cb:has(> a.viewfulltext), .viewfulltext, .post-message .noselect, a.dummy-link, "
                            ".embed, .embed-container, .media-player, .video-container, .audio-container, a[download], a[href^='javascript:']"):
        node.decompose()
    wiki = soup.select_one("#mw-content-text .mw-parser-output") or soup.select_one("#mw-content-text") or soup.select_one(".mw-parser-output")
    if wiki is not None:
        classes = soup.body.get("class", []) if soup.body else []
        namespaces = [c for c in classes if re.fullmatch(r"ns-\d+|ns--\d+", c)]
        if (namespaces and "ns-0" not in namespaces) or "page-Main_Page" in classes:
            return title, "", links, []  # Utility pages/homepage are for discovery only.
        for node in wiki.select("table:not(.wikitable)"):
            node.decompose()
        # Positive selection: prose, headings, lists, quotations, code, and article data tables.
        # Arbitrary divs can contain widgets/errors; never flatten the entire parser output.
        roots = wiki.select("p, h1, h2, h3, h4, h5, h6, ul, ol, dl, blockquote, pre, table.wikitable")
    else:
        roots = soup.select(POST_SELECTOR)
    # Unknown layouts supply discovery links, never presumed training text.
    root_ids = {id(root) for root in roots}
    roots = [root for root in roots if not any(id(parent) in root_ids for parent in root.parents)]
    parts, documents, skip_level = [], [], None
    for root in roots:
        if wiki is not None and re.fullmatch(r"h[1-6]", root.name):
            level = int(root.name[1])
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if root.get_text(" ", strip=True).casefold() in {"see also", "notes", "references", "citations", "snopes", "external links", "gallery", "galleries"}:
                skip_level = level if skip_level is None else min(skip_level, level)
        if skip_level is not None:
            continue
        for br in root.select("br"):
            br.replace_with("\n")
        for block in root.select("p, li, dt, dd, tr"):
            block.insert_before("\n")
        for cell in root.select("th, td"):
            cell.insert_before(" ")
        text = clean(root.get_text())
        if text:
            parts.append(text)
            if wiki is None:
                identity = identities[id(root)]
                if not identity['truncated']:
                    documents.append(dict(text=text, kind='post', **{key: value for key, value in identity.items() if key != 'truncated'}))
    text = "\n\n".join(parts)
    if wiki is not None and text:
        documents = [dict(text=text, kind='article', url=url, group=url)]
    return title, text, links, documents


def extract(html):
    title, text, links, _ = _extract(html)
    return title, text, links


def extract_documents(html, url, discover=True):
    title, _, links, documents = _extract(html, url, discover)
    return title, documents, links


def challenge_document(soup):
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    return (title in {"just a moment...", "just a moment…", "attention required! | cloudflare", "checking your browser", "security verification"}
            or soup.select_one("#challenge-form, #cf-challenge-running") is not None
            # A posting-form Turnstile widget is not a whole-page challenge.
            or (soup.select_one('.cf-turnstile') is not None
                and soup.select_one(WIKI_SELECTOR + ', ' + BOARD_SELECTOR) is None))


def is_challenge(html):
    return challenge_document(BeautifulSoup(html, "html.parser"))


def has_wiki_document(html):
    # Discovery pages may intentionally have no trainable article text.
    return BeautifulSoup(html, "html.parser").select_one(WIKI_SELECTOR) is not None


def wiki_ready(html):
    return site_ready(html, 'soyjakwiki.org')


def site_ready(html, host):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True).lower() if soup.title else ""
    return (not challenge_document(soup)
            and title not in {'403 forbidden', '404 not found', 'access denied'}
            and soup.select_one(WIKI_SELECTOR if host in WIKI_HOSTS else BOARD_SELECTOR) is not None)


async def inspect_wiki_tabs(preferred):
    return await inspect_site_tabs(preferred, 'soyjakwiki.org')


async def inspect_site_tabs(preferred, host):
    """Verification may finish in another tab, but only the requested site's HTML counts."""
    seen, details = set(), []
    for candidate in [preferred, *preferred.context.pages]:
        if id(candidate) in seen or candidate.is_closed():
            continue
        seen.add(id(candidate))
        url = candidate.url
        try:
            eligible = canonical(url, host) is not None
        except ValueError:
            eligible = False
        entry = {"url": url, "eligible": eligible}
        details.append(entry)
        if not entry["eligible"]:
            continue
        try:
            html = await asyncio.wait_for(candidate.content(), timeout=2)
            soup = BeautifulSoup(html, "html.parser")
            entry.update(title=soup.title.get_text(" ", strip=True) if soup.title else "",
                         site_ready=site_ready(html, host), challenge=challenge_document(soup))
            if host in WIKI_HOSTS:
                entry['wiki_ready'] = entry['site_ready']
            if entry["site_ready"]:
                return candidate, html, details
        except Exception as exc:
            entry["error"] = str(exc).splitlines()[0][:200]
    return None, None, details


async def browser_snapshot(page, progress):
    if progress is None or progress.path is None or page.is_closed():
        return
    path = progress.path.with_suffix(".browser.png")
    try:
        image = await page.screenshot(timeout=2500, animations="disabled")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(image)
        tmp.replace(path)
        progress.update(screenshot_at=time.time())
    except Exception as exc:
        progress.update(screenshot_error=str(exc).splitlines()[0][:200])


async def wait_for_wiki(preferred, seconds, progress=None):
    return await wait_for_site(preferred, 'soyjakwiki.org', seconds, progress)


async def wait_for_site(preferred, host, seconds, progress=None):
    deadline = time.monotonic() + seconds
    label = 'wiki document' if host in WIKI_HOSTS else 'board page'
    last_details, last_snapshot = None, 0
    while True:
        selected, html, details = await inspect_site_tabs(preferred, host)
        if progress and details != last_details:
            descriptions = [f"{row.get('title', '(unavailable)')} — {row['url']}" for row in details]
            progress.update(detail="BROWSER " + " | ".join(descriptions), browser_tabs=details)
            last_details = details
        if selected is not None:
            await selected.bring_to_front()
            if progress:
                progress.update("verified", f"READY {label} detected in {selected.url}", url=selected.url)
                await browser_snapshot(selected, progress)
            return selected, html
        now = time.monotonic()
        if progress and now - last_snapshot >= 15:
            await browser_snapshot(preferred, progress)
            last_snapshot = now
        if now >= deadline:
            raise RuntimeError(f"No verified {label} for {host} after {seconds}s; "
                               + " | ".join(f"{row.get('title', '?')} ({row['url']})" for row in details))
        await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))


def curl_fetch(url):
    with curl_requests.Session(impersonate="chrome") as session:
        for _ in range(6):
            if canonical(url, "soyjakwiki.org") is None:
                raise NonTextResource(f"Not an in-scope HTML page: {url}")
            # Streaming exposes headers before the body. Do not download a media
            # response or follow a redirect to an attachment, even without a suffix.
            response = session.get(url, timeout=45, allow_redirects=False, stream=True)
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise RuntimeError("Redirect has no Location header")
                    url = urljoin(response.url, location)
                    continue
                if not html_headers(response.headers):
                    raise NonTextResource(f"Non-HTML response: {response.headers.get('content-type', '(missing)')}")
                body = bytearray()
                for chunk in response.iter_content():
                    body.extend(chunk)
                    if len(body) > 10_000_000:
                        raise RuntimeError("HTML response exceeds 10 MB")
                return response.status_code, response.url, body.decode(response.encoding or "utf-8", errors="replace")
            finally:
                response.close()
        raise RuntimeError("Too many redirects")


class NonTextResource(RuntimeError):
    pass


class BrowserClosed(RuntimeError):
    pass


class CrawlFailed(RuntimeError):
    pass


async def fetch_page(page, url, host, render_wait_ms, challenge_wait=180, progress=None, select_page=None):
    if canonical(url, host) is None:
        raise NonTextResource(f"Not an in-scope HTML page: {url}")
    initial_page = page
    status = 0
    def track_navigation(response):
        nonlocal status
        if response.request.is_navigation_request() and response.request.frame == page.main_frame:
            status = response.status
    page.on("response", track_navigation)
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        if response is not None and not html_headers(response.headers):
            raise NonTextResource(f"Non-HTML response: {response.headers.get('content-type', '(missing)')}")
        if not status:
            status = response.status if response else 0
        await page.wait_for_timeout(render_wait_ms)
        html = await page.content()
        if site_ready(html, host) and (200 <= status < 400 or status in {0, 403, 503}):
            # Cloudflare can replace its initial 403 document after verification without
            # producing another navigation response visible to this listener.
            return page.url, html, "camoufox"
        if challenge_wait and (host in WIKI_HOSTS or is_challenge(html) or status in {403, 503}):
            message = f"WAIT {url}: checking {host} tabs for verified content; waiting up to {challenge_wait}s. Complete verification in the browser if prompted."
            if progress:
                progress.update("waiting_verification", message)
            else:
                print(message, flush=True)
            page, html = await wait_for_site(page, host, challenge_wait, progress)
            if select_page:
                select_page(page)
            return page.url, html, "camoufox"
        if 200 <= status < 400 and not is_challenge(html):
            if host not in WIKI_HOSTS or has_wiki_document(html):
                return page.url, html, "camoufox"
        reason = f"HTTP {status}, challenge or missing article text"
    except NonTextResource:
        raise
    except Exception as exc:
        if page.is_closed():
            raise BrowserClosed("Browser tab was closed") from exc
        reason = str(exc)
    finally:
        initial_page.remove_listener("response", track_navigation)
    if progress:
        _, _, details = await inspect_site_tabs(page, host)
        progress.update(browser_tabs=details)
        await browser_snapshot(page, progress)
    if host != "soyjakwiki.org":
        raise RuntimeError(reason)
    message = f"FALLBACK {url}: Camoufox {reason}; trying curl_cffi"
    if progress:
        progress.update("http_fallback", message)
    else:
        print(message, flush=True)
    status, final, html = await asyncio.to_thread(curl_fetch, url)
    if not 200 <= status < 400 or is_challenge(html) or not has_wiki_document(html):
        raise RuntimeError(f"curl_cffi HTTP {status}; no usable article text")
    return final, html, "curl_cffi"


async def text_only_request(route, progress=None):
    request = route.request
    blocked = request.resource_type in {"image", "media", "font", "texttrack"} or attachment_url(request.url)
    if request.is_navigation_request():
        blocked = blocked or not page_url(request.url)
        # Embedded players are documents too. Keep only verification subframes.
        if request.frame.parent_frame is not None:
            blocked = blocked or urlsplit(request.url).hostname != "challenges.cloudflare.com"
    if blocked:
        if progress is not None:
            progress.count("blocked_resources")
        await route.abort()
    else:
        await route.fallback()


@asynccontextmanager
async def browser_session(args, progress):
    environment = browser_environment(args.headed)
    if args.headed:
        display = environment.get('DISPLAY') or environment.get('WAYLAND_DISPLAY')
        progress.update(browser_display=display)
        if display and display != (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            progress.update(detail=f'DISPLAY using current desktop session {display} for visible Camoufox')
    manager = AsyncCamoufox(headless=not args.headed, humanize=args.headed,
                           env=environment,
                           persistent_context=True, user_data_dir=str(Path(args.profile).resolve()),
                           accept_downloads=False, service_workers="block",
                           firefox_user_prefs={"media.autoplay.default": 5, "media.preload.default": 0,
                                               "media.play-stand-alone": False, "browser.startup.page": 0,
                                               "browser.sessionstore.resume_from_crash": False,
                                               "browser.sessionstore.resume_session_once": False})
    context = await manager.__aenter__()
    try:
        # Applies to visible/headless browsers and every newly opened tab.
        await context.route("**/*", lambda route: text_only_request(route, progress))
        yield context
    finally:
        progress.update("closing_browser", "CLOSE closing Camoufox and saving the browser profile")
        # The job supervisor terminates remaining owned processes if browser shutdown stalls.
        await asyncio.wait_for(manager.__aexit__(None, None, None), timeout=15)


def browser_environment(headed, environ=None):
    """Recover desktop settings for services that started before graphical login."""
    environment = dict(os.environ if environ is None else environ)
    if not headed or sys.platform != 'linux':
        return environment
    if not (environment.get('DISPLAY') or environment.get('WAYLAND_DISPLAY')):
        keys = {'DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR', 'XDG_SESSION_TYPE', 'MOZ_ENABLE_WAYLAND'}
        try:
            result = subprocess.run(['systemctl', '--user', 'show-environment'], env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True, timeout=3)
            desktop = {}
            for line in result.stdout.splitlines():
                key, _, value = line.partition('=')
                if key in keys:
                    values = shlex.split(value)
                    if len(values) == 1:
                        desktop[key] = values[0]
            if desktop.get('DISPLAY') or desktop.get('WAYLAND_DISPLAY'):
                # The display and its authorization must come from the same session.
                environment.pop('XAUTHORITY', None)
                environment.update(desktop)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if not (environment.get('DISPLAY') or environment.get('WAYLAND_DISPLAY')):
        raise CrawlFailed('Visible browser requested, but no desktop display is available. '
                          'Log into the server desktop, or uncheck "Visible browser for Cloudflare verification" '
                          'for a headless collection (CLI: omit --headed).')
    if not environment.get('DISPLAY') and environment.get('WAYLAND_DISPLAY'):
        environment['MOZ_ENABLE_WAYLAND'] = '1'
    return environment


async def crawl(args, progress, cancel):
    with corpus_lock(args.output):
        await _crawl(args, progress, cancel)


async def _crawl(args, progress, cancel):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes, seen, corpus_digest = set(), set(), hashlib.sha256()
    if output.exists():
        with output.open('rb') as source:
            for line in source:
                row = json.loads(line)
                if row.get("extraction") != EXTRACTION_VERSION:
                    raise ValueError("Output contains older extraction results. Use a fresh output file for strict article/post text.")
                if not line.endswith(b'\n'):
                    raise ValueError("Corpus has an incomplete final line; repair it or choose a fresh file before continuing")
                hashes.add(content_hash(row['text']))
                seen.add(row.get('page_url') or row['url'].split('#', 1)[0])
                corpus_digest.update(line)
    settings = {key: getattr(args, key) for key in CRAWL_SETTINGS}
    archive = info = None
    if 'wikipedia.org' in args.hosts:
        if args.resume and not read_state(output)['settings'].get('wikipedia_archive'):
            raise ValueError('This is a legacy Wikipedia web crawl. Choose a fresh corpus for Kiwix collection.')
        archive, info = open_archive(args.wikipedia_zim)
        settings['wikipedia_archive'] = info
    frontier = Frontier(output, settings, resume=args.resume)
    progress.state.update(frontier.stats, resumed=args.resume, frontier_path=str(frontier.path), queued=frontier.pending())
    for host in args.hosts:
        progress.state['sources'].setdefault(host, {})
    progress.state['budget'] = args.pages_per_site * len(args.hosts)
    if info and args.wikipedia_all:
        progress.state['budget'] += info['entries'] - args.pages_per_site
    try:
        if archive is not None:
            with output.open('ab') as out:
                await import_wikipedia(args, progress, frontier, out, hashes, corpus_digest, archive, info)
        web_hosts = [host for host in args.hosts if host != 'wikipedia.org']
        if not web_hosts:
            return
        profile = Path(args.profile).resolve()
        profile.mkdir(parents=True, exist_ok=True)
        progress.update("opening_browser", f"START {'visible' if args.headed else 'headless'} browser; "
                        f"{len(hashes)} text records already saved; {'resuming saved queue; ' if args.resume else ''}"
                        f"total limit {args.pages_per_site} navigation attempts/site", headed=args.headed, mode='web')
        async with browser_session(args, progress) as context, _corpus_writer(output) as out:
            restored = list(context.pages)
            page = await context.new_page()
            for extra in restored:
                await extra.close()
            await page.bring_to_front()
            def select_page(selected):
                nonlocal page
                page = selected
            def closed(*_):
                if progress.state["phase"] != "closing_browser":
                    cancel("Browser window/tab closed by user")
            context.on("close", closed)

            def checkpoint(finished=None, release=None):
                out.flush()
                os.fsync(out.fileno())
                frontier.checkpoint(progress.counts(), out.tell(), corpus_digest.hexdigest(), finished, release)
                units = progress.state['completed']
                if info and args.wikipedia_all:
                    units += frontier.get('kiwix')['cursor'] - progress.state['sources']['wikipedia.org'].get('completed', 0)
                progress.update(queued=frontier.pending(), completed_units=units)

            for index, host in enumerate(web_hosts):
                if index:
                    # A failed Cloudflare tab can still be refreshing itself.
                    # Do not let its navigation interrupt the next source's goto.
                    previous_tabs = list(context.pages)
                    page = await context.new_page()
                    for previous_page in previous_tabs:
                        await previous_page.close()
                    await page.bring_to_front()
                progress.update(host=host, browser_tabs=[])
                policies, last_request = {}, 0.0
                while progress.state['sources'].get(host, {}).get('attempted', 0) < args.pages_per_site:
                    pending = frontier.next(host)
                    if pending is None:
                        break
                    url, depth = pending
                    completed, attempted_page = True, False
                    try:
                        origin = "{}://{}".format(urlsplit(url).scheme, urlsplit(url).netloc)
                        progress.update("checking_policy", url=url, host=host, queued=frontier.pending())
                        if origin not in policies:
                            try:
                                policies[origin] = await asyncio.to_thread(robots, origin, args.ignore_robots)
                            except Exception as exc:
                                progress.update(detail=f"SKIP {origin}: {exc}")
                                policies[origin] = None
                        policy = policies[origin]
                        if policy is None or not policy.can_fetch(AGENT, url):
                            progress.count('skipped')
                            progress.update(detail=f"SKIP {url}: robots policy unavailable or disallowed")
                            continue
                        delay = max(args.delay, policy.crawl_delay(AGENT) or policy.crawl_delay('*') or 0)
                        rate = policy.request_rate(AGENT) or policy.request_rate('*')
                        if rate and rate.requests:
                            delay = max(delay, rate.seconds / rate.requests)
                        progress.update('rate_limit')
                        await asyncio.sleep(max(0, last_request + delay - time.monotonic()))
                        progress.count('attempted')
                        attempted_page = True
                        checkpoint()
                        attempted = progress.state['sources'][host]['attempted']
                        progress.update('fetching', f'FETCH [{attempted}/{args.pages_per_site}] {url}')
                        last_request = time.monotonic()
                        final_url, html, transport = await fetch_page(page, url, host, args.render_wait_ms, args.challenge_wait, progress, select_page)
                        final = canonical(final_url, host)
                        if not final:
                            progress.count("skipped")
                            progress.update(detail=f"SKIP {url}: redirect outside crawl scope")
                            continue
                        final_origin = "{}://{}".format(urlsplit(final).scheme, urlsplit(final).netloc)
                        if final_origin not in policies:
                            policies[final_origin] = await asyncio.to_thread(robots, final_origin, args.ignore_robots)
                        if not policies[final_origin] or not policies[final_origin].can_fetch(AGENT, final):
                            progress.count("skipped")
                            progress.update(detail=f"SKIP {final}: redirect disallowed by robots policy")
                            continue
                        progress.update("extracting", url=final)
                        title, documents, links = extract_documents(html, final)
                        if depth < args.depth:
                            discovered = []
                            for href in links:
                                try:
                                    link = canonical(urljoin(final, href), host)
                                except ValueError:
                                    continue
                                if link:
                                    discovered.append((link, depth + 1))
                                else:
                                    progress.count("filtered_links")
                            frontier.discover(host, discovered)
                        saved = 0
                        for document in documents:
                            text = document['text']
                            minimum = args.min_post_chars if document['kind'] == 'post' else args.min_chars
                            if len(text) < minimum:
                                progress.count('too_short')
                                continue
                            digest = content_hash(text)
                            if digest in hashes:
                                progress.count('duplicates')
                                continue
                            row = dict(document, page_url=final, host=host, title=title, sha256=hashlib.sha256(text.encode()).hexdigest(),
                                       transport=transport, extraction=EXTRACTION_VERSION, text_policy=TEXT_POLICY_VERSION,
                                       fetched_at=datetime.now(timezone.utc).isoformat())
                            payload = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                            out.write(payload)
                            corpus_digest.update(payload)
                            hashes.add(digest)
                            saved += 1
                            progress.count('saved')
                            progress.count('characters', len(text))
                            progress.count('text_bytes', len(text.encode()))
                        if saved:
                            progress.count('saved_pages')
                            seen.add(final)
                            progress.update(detail=f'SAVE {final} ({saved} new article/post records)')
                        else:
                            reason = ('URL already saved; discovered links only' if final in seen else
                                      'no article/post text (discovery only)' if not documents else
                                      'duplicate text or too short; discovered links only')
                            progress.count('skipped')
                            progress.update(detail=f'SKIP {final}: {reason}')
                    except (BrowserClosed, asyncio.CancelledError):
                        completed = False
                        raise
                    except NonTextResource as exc:
                        progress.count("skipped")
                        progress.update(detail=f"SKIP {url}: {exc}")
                    except Exception as exc:
                        progress.count("failed")
                        progress.update(detail=f"ERROR {url}: {exc}")
                    finally:
                        if completed and attempted_page:
                            progress.count('completed')
                        checkpoint(finished=url if completed else None, release=None if completed else url)
                source = progress.state['sources'].get(host, {})
                progress.update(detail=f"SITE DONE {host}: {source.get('attempted', 0)} attempted, {source.get('saved', 0)} unique text records, "
                                f"{source.get('text_bytes', 0)} text bytes; {source.get('duplicates', 0)} duplicates removed")
    finally:
        frontier.close()


@asynccontextmanager
async def _corpus_writer(path):
    with path.open('ab') as stream:
        yield stream


async def run(args):
    progress = CrawlProgress(args.progress_file, args.pages_per_site * len(args.hosts))
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    stop_reason = None
    def cancel(reason="Stop requested"):
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
            progress.update("stopping", f"STOP {reason}; saving collection state")
            task.cancel()
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {sig: signal.getsignal(sig) for sig in signals}
    for sig in signals:
        loop.add_signal_handler(sig, cancel)
    progress.update("starting", "START initializing crawler")
    heartbeat = asyncio.create_task(progress.heartbeat())
    try:
        await crawl(args, progress, cancel)
        if progress.state["completed"] and progress.state["failed"] == progress.state["completed"]:
            raise CrawlFailed("Every attempted page failed; no usable pages were fetched. See the browser diagnostics above.")
        progress.update("completed", f"DONE {progress.state['completed']} pages checked; {progress.state['saved']} saved, "
                        f"{progress.state['skipped']} skipped, {progress.state['failed']} failed. Browser closed.")
    except (asyncio.CancelledError, BrowserClosed) as exc:
        progress.update("stopped", f"STOPPED {stop_reason or str(exc)}. Browser closed.")
    except Exception as exc:
        progress.update("failed", f"FAILED {exc}")
        raise
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        for sig in signals:
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous[sig])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosts", nargs="+", choices=HOSTS, default=list(HOSTS))
    p.add_argument("--output", default="data/raw-text-v4.jsonl")
    p.add_argument("--pages-per-site", type=int, default=100)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--delay", type=float, default=3.0)
    p.add_argument("--render-wait-ms", type=int, default=1500)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--min-post-chars", type=int, default=40)
    p.add_argument('--wikipedia-zim', default=DEFAULT_PATH, help='Local Kiwix English Wikipedia .zim archive')
    p.add_argument('--wikipedia-seed', type=int, default=42, help='Repeatable spread-out archive order')
    p.add_argument('--wikipedia-all', action='store_true', help='Process the entire ZIM; the page limit applies only to website sources')
    p.add_argument("--resume", action="store_true", help="Continue the saved queue; a larger page limit extends its total budget")
    p.add_argument("--headed", action="store_true", help="Show Camoufox so you can complete browser verification")
    p.add_argument("--profile", default="data/browser-profile", help="Persistent browser profile; do not share between simultaneous crawlers")
    p.add_argument("--challenge-wait", type=int, default=180, help="Seconds to wait for verification on any source; wiki then tries curl_cffi fallback")
    p.add_argument("--progress-file", help="Atomic JSON progress snapshot for the job monitor")
    p.add_argument("--ignore-robots", action="store_true", help="Skip robots.txt fetching and rules; retain --delay and crawl limits")
    args = p.parse_args()
    if args.resume:
        saved = read_state(args.output)['settings']
        for key, value in saved.items():
            if key == 'wikipedia_all':
                args.wikipedia_all = args.wikipedia_all or value
            else:
                setattr(args, key, max(args.pages_per_site, value) if key == 'pages_per_site' else value)
    if args.pages_per_site < 1 or args.depth < 0 or args.delay < 0 or args.render_wait_ms < 0 or args.challenge_wait < 0 or min(args.min_chars, args.min_post_chars) < 1:
        p.error("invalid crawl limits")
    try:
        asyncio.run(run(args))
    except CrawlFailed as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
