"""Text-only collection regressions: no model training, no external network."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scratchllm import prepare
from scratchllm.crawl import (EXTRACTION_VERSION, NonTextResource, canonical, curl_fetch,
                              extract, fetch_page)
from scratchllm.jobs import CrawlRequest, Jobs
import test_crawl_lifecycle


class TextPolicyTests(unittest.TestCase):
    def test_media_urls_never_enter_page_queue(self):
        blocked = ['/file/post/track.mp3', '/SONG.MP3?x=1', '/song%2Emp3', '/song%252emp3',
                   '/song.mp3;download=1', '/file/post/opaque', '/uploads/opaque', '/audio/stream',
                   '/track.flac', '/film.mkv', '/playlist.m3u8', '/book.pdf', '/picture.avif',
                   '/file.php?id=3', '/download?filename=track.mp3', '/download?download=1',
                   '/view?url=https%3A%2F%2Fexample.test%2Ftrack.ogg',
                   '/index.php?title=File%3ASong.mp3', '/Special:FilePath/opaque',
                   '/index.php?title=Article&action=edit', '/index.php?title=Article&action=history']
        for path in blocked:
            with self.subTest(path=path):
                self.assertIsNone(canonical('https://swedishwin.com' + path, 'swedishwin.com'))
        for path in ['/g/', '/g/index.html', '/g/2.html', '/g/thread/123.html#123', '/arch/res/456.html']:
            self.assertIsNotNone(canonical('https://swedishwin.com' + path, 'swedishwin.com'))
        self.assertEqual(canonical('https://soyjakwiki.org/index.php?title=Article', 'soyjakwiki.org'),
                         'https://soyjakwiki.org/index.php?title=Article')

    def test_post_ui_and_player_text_removed_without_keyword_stripping(self):
        html = '''<nav>Site menu</nav><form><div class="post"><header>Anonymous Reply</header>
          <div class="post-message"><h2>My heading</h2>User-written text.<br>Expand this idea and play a song.
          <span class="noselect bold">[<a class="dummy-link">Expand</a>]</span>
          <small class="edited">Last edited yesterday by admin</small>
          <div class="cb mt-5 ml-5">Message too long. <a class="viewfulltext" href="/g/thread/123.html">View the full text</a></div>
          <div class="file">File (hide): (100KB, track.mp3)<a download href="/file/post/opaque">Download</a></div>
          <audio controls src="/song.mp3">Audio fallback</audio><video>Play Pause Volume</video>
          <iframe src="/player">Embedded player</iframe><object>Download plugin</object>
          <label>Delete this post</label><button>Report</button><span role="button">Quote</span>
          <div class="embed-container">Watch on video site</div></div></div></form>
          <a href="/song.mp3">Listen</a><a href="/file/post/opaque">File</a>'''
        _, text, links = extract(html)
        self.assertIn('User-written text.', text)
        self.assertIn('Expand this idea and play a song.', text)
        self.assertEqual(links, ['/g/thread/123.html'])
        for label in ['Site menu', 'Anonymous', '[Expand]', 'Last edited', 'Message too long', 'View the full text',
                      'File (hide)', 'track.mp3', 'Download', 'fallback', 'Volume', 'player', 'Delete', 'Report', 'Watch on']:
            self.assertNotIn(label, text)
        self.assertEqual(extract('<article><h1>Settings</h1><p>Unknown site UI</p></article>')[1], '')

    def test_browser_rejects_direct_media_and_non_html_before_reading_dom(self):
        page = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()
        page.is_closed = MagicMock(return_value=False)
        with patch('scratchllm.crawl.curl_fetch') as fallback:
            with self.assertRaises(NonTextResource):
                asyncio.run(fetch_page(page, 'https://swedishwin.com/file/post/song.mp3', 'swedishwin.com', 0))
            page.goto.assert_not_awaited()
            page.goto.return_value = MagicMock(headers={'content-type': 'audio/mpeg'})
            with self.assertRaises(NonTextResource):
                asyncio.run(fetch_page(page, 'https://soyjakwiki.org/opaque', 'soyjakwiki.org', 0))
            page.content.assert_not_awaited()
            fallback.assert_not_called()

    def test_curl_checks_headers_and_redirects_before_consuming_body(self):
        for headers, status in [({'content-type': 'audio/mpeg'}, 200),
                                ({'content-type': 'application/octet-stream'}, 200),
                                ({'content-type': 'text/html', 'content-disposition': 'attachment; filename=x.html'}, 200),
                                ({'location': '/song.mp3'}, 302)]:
            with self.subTest(headers=headers), patch('scratchllm.crawl.curl_requests.Session') as session:
                response = MagicMock(status_code=status, url='https://soyjakwiki.org/Article', headers=headers)
                session.return_value.__enter__.return_value.get.return_value = response
                with self.assertRaises(NonTextResource):
                    curl_fetch('https://soyjakwiki.org/Article')
                response.iter_content.assert_not_called()
                response.close.assert_called_once()
                session.return_value.__enter__.return_value.get.assert_called_once()
        with patch('scratchllm.crawl.curl_requests.Session') as session:
            response = MagicMock(status_code=200, url='https://soyjakwiki.org/Article',
                                 headers={'content-type': 'text/html; charset=utf-8'}, encoding='utf-8')
            response.iter_content.return_value = [b'<p>Actual ', b'article</p>']
            session.return_value.__enter__.return_value.get.return_value = response
            self.assertEqual(curl_fetch(response.url), (200, response.url, '<p>Actual article</p>'))
            session.return_value.__enter__.return_value.get.assert_called_once_with(response.url, timeout=45, allow_redirects=False, stream=True)

    def test_preparation_rejects_old_crawl_before_writing_or_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'old.jsonl', Path(directory)/'prepared'
            source.write_text(json.dumps(dict(text='Old UI text', extraction='article-post-text-v3')) + '\n')
            with patch('sys.argv', ['prepare', '--input', str(source), '--output', str(output)]):
                with self.assertRaises(SystemExit):
                    prepare.main()
            self.assertFalse(output.exists())
            source.write_text(json.dumps(dict(text='Actual written text', host='swedishwin.com', extraction=EXTRACTION_VERSION)) + '\n')
            prepare.validate_corpus(source)


class TextOnlyBrowserTests(unittest.TestCase):
    def test_visible_and_headless_settings_block_media_and_only_save_written_text(self):
        helper = test_crawl_lifecycle.CrawlLifecycleTests()
        for headed in (False, True):
            with self.subTest(headed=headed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = Jobs(root, 'cpu')
                try:
                    helper.browser_job(jobs, 'text-only', CrawlRequest(hosts=['swedishwin.com'], output='corpus.jsonl',
                        pages_per_site=10, depth=1, delay=0.5, headed=headed, challenge_wait=0))
                    result = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                    self.assertEqual(result['status'], 'succeeded', result['log'])
                    self.assertEqual(result['progress']['completed'], 2)
                    self.assertEqual(result['progress']['saved'], 2)
                    self.assertGreaterEqual(result['progress']['blocked_resources'], 4)
                    requests = [json.loads(line) for line in (root/'corpus.requests.jsonl').read_text().splitlines()]
                    self.assertEqual([row['url'] for row in requests], ['https://swedishwin.com/', 'https://swedishwin.com/new'])
                    rows = [json.loads(line) for line in (root/'corpus.jsonl').read_text().splitlines()]
                    self.assertTrue(all(row['extraction'] == EXTRACTION_VERSION for row in rows))
                    for row in rows:
                        for label in ['Expand', 'Last edited', 'Message too long', 'File (hide)', 'Player', 'Download']:
                            self.assertNotIn(label, row['text'])
                    self.assertNotIn('FETCH [3/', result['log'])
                    helper.assert_closed(jobs)
                finally:
                    jobs.close()


if __name__ == '__main__':
    unittest.main()
