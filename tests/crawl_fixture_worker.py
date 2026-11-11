"""Real-browser test worker: all requests are intercepted; no external corpus access."""
import asyncio
import json
import sys
from pathlib import Path

from camoufox.async_api import AsyncCamoufox
from scratchllm import crawl

mode, *arguments = sys.argv[1:]
BASE = "https://soyjakwiki.org/"
TEXT = "An actual article paragraph with enough written text for extraction. " * 8
ROOT = f'<body class="ns-0"><div id="mw-content-text"><div class="mw-parser-output"><p>{TEXT}</p><a href="/empty">empty</a><a href="/new">new</a><a href="/same">same</a></div></div></body>'


class FixtureBrowser:
    def __init__(self, **kwargs):
        if mode == 'text-only':
            # Exercise both --headed settings with an offscreen browser in tests.
            # Production still receives the selected headed flag and installs its guard.
            kwargs['headless'] = True
        self.manager = AsyncCamoufox(**kwargs)

    async def __aenter__(self):
        context = await self.manager.__aenter__()
        # Reproduce persistent-profile extra tabs. The crawler must reduce this to one.
        await context.new_page()
        await context.new_page()

        verification_tab_opened = False
        async def route(request):
            nonlocal verification_tab_opened
            status = 200
            if mode in {'text-only', 'resume', 'board-verify'}:
                log = Path(arguments[arguments.index('--output') + 1]).with_suffix('.requests.jsonl')
                with log.open('a') as stream:
                    stream.write(json.dumps({'url': request.request.url, 'type': request.request.resource_type,
                                             'page': id(request.request.frame.page)}) + '\n')
            if mode == 'board-verify':
                if request.request.url == 'https://soyjak.st/':
                    status = 403
                    html = '''<title>Just a moment...</title><form id="challenge-form">Verification UI</form>
                              <script>setTimeout(() => location.href = '/verified', 2800)</script>'''
                else:
                    text = TEXT if 'soyjak.st' in request.request.url else 'Another site has actual written post text. ' * 15
                    html = f'''<title>Board</title><div class="boardlist"><a href="/b/">Board</a></div>
                               <div class="post" id="op_1"><div class="post-message">{text}</div></div>
                               <form><div class="cf-turnstile">Posting verification UI</div></form>
                               <audio src="/sound.mp3" preload="auto">Player UI</audio>'''
            elif mode == 'text-only':
                if request.request.url.endswith('/new'):
                    html = ROOT.replace(TEXT, 'New written article text. ' * 25)
                    html = html.replace('<a href="/empty">empty</a><a href="/new">new</a><a href="/same">same</a>', '')
                else:
                    html = f'''<div class="post"><div class="post-message">{TEXT}
                      <span class="noselect">[<a class="dummy-link">Expand</a>]</span>
                      <small class="edited">Last edited today by admin</small>
                      <audio controls preload="auto" src="/song.mp3">Player fallback label</audio>
                      <video preload="auto" src="/video.webm">Video controls</video>
                      <img src="/image-without-extension"><iframe src="/player"></iframe>
                      <div class="file">File (hide): track.mp3 <a href="/file/post/opaque">Download</a></div>
                      </div></div><div class="post-message">A truncated preview
                      <div class="cb mt-5 ml-5">Message too long. <a class="viewfulltext" href="/new">View the full text</a></div></div>
                      <a href="/SONG.MP3">Audio</a><a href="/file/post/opaque">Attachment</a>
                      <a href="/download?filename=track.ogg">Download</a><a href="/new">Thread</a>
                      <script>fetch('/file/post/opaque').catch(() => {{}}); fetch('/song.mp3').catch(() => {{}});</script>'''
            elif mode == "other-tab" and request.request.url.endswith('/verified'):
                html = ROOT
            elif mode in {"wait", "close-window", "other-tab"}:
                html = '<title>Just a moment...</title><form id="challenge-form">Manual verification</form>'
                if mode == "close-window":
                    async def close_later():
                        await asyncio.sleep(2)
                        await context.close()
                    asyncio.create_task(close_later())
                if mode == "other-tab" and not verification_tab_opened:
                    verification_tab_opened = True
                    async def verified_tab():
                        await asyncio.sleep(1)
                        verified = await context.new_page()
                        await verified.goto(BASE + 'verified')
                    asyncio.create_task(verified_tab())
            elif request.request.url.endswith('/empty'):
                html = '<div id="mw-content-text"><div class="mw-parser-output"></div></div>'
            elif request.request.url.endswith('/new'):
                html = ROOT.replace(TEXT, 'New article content. ' * 30)
            else:
                html = ROOT
            await request.fulfill(status=status, content_type="text/html", body=html)
        # The production guard is registered after this handler. Allowed requests
        # fall through to these offline fixtures; blocked requests never get here.
        await context.route("**/*", route)
        return context

    async def __aexit__(self, *args):
        return await self.manager.__aexit__(*args)


# Interception runs inside the ordinary CLI and lifecycle, including its real signal handlers.
crawl.AsyncCamoufox = FixtureBrowser
sys.argv = ["scratchllm.crawl", *arguments]
crawl.main()
