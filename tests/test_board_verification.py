"""Board challenge handling and source isolation, using offline browser fixtures only."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scratchllm.crawl import fetch_page, is_challenge, site_ready, wait_for_site
from scratchllm.jobs import CrawlRequest, Jobs
import test_crawl_lifecycle


class BoardVerificationTests(unittest.TestCase):
    def test_challenge_settles_then_each_source_uses_a_fresh_tab(self):
        helper = test_crawl_lifecycle.CrawlLifecycleTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            try:
                helper.browser_job(jobs, 'board-verify', CrawlRequest(hosts=['soyjak.st', 'swedishwin.com'],
                    output='corpus.jsonl', pages_per_site=1, headed=False, challenge_wait=10))
                result = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertIn('WAIT https://soyjak.st/', result['log'])
                self.assertIn('READY board page detected', result['log'])
                self.assertEqual(result['progress']['saved'], 2)
                self.assertTrue(result['browser_snapshot'])
                rows = [json.loads(line) for line in (root/'corpus.jsonl').read_text().splitlines()]
                self.assertEqual({row['host'] for row in rows}, {'soyjak.st', 'swedishwin.com'})
                for row in rows:
                    self.assertNotIn('UI', row['text'])
                visits = [json.loads(line) for line in (root/'corpus.requests.jsonl').read_text().splitlines()]
                self.assertEqual([visit['url'] for visit in visits], ['https://soyjak.st/', 'https://soyjak.st/verified', 'https://swedishwin.com/'])
                self.assertEqual(visits[0]['page'], visits[1]['page'])
                self.assertNotEqual(visits[1]['page'], visits[2]['page'])
                helper.assert_closed(jobs)
            finally:
                jobs.close()

    def test_board_timeout_preserves_diagnostics_and_never_saves_challenge_text(self):
        helper = test_crawl_lifecycle.CrawlLifecycleTests()
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                helper.browser_job(jobs, 'wait', CrawlRequest(hosts=['soyjak.st'], output='corpus.jsonl',
                    pages_per_site=1, headed=False, challenge_wait=1))
                result = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'failed', result['log'])
                self.assertIn('No verified board page for soyjak.st after 1s', result['log'])
                self.assertTrue(result['browser_snapshot'])
                self.assertTrue(result['progress']['browser_tabs'][0]['challenge'])
                self.assertEqual(Path(directory, 'corpus.jsonl').read_text(), '')
                helper.assert_closed(jobs)
            finally:
                jobs.close()

    def test_posting_widget_is_not_a_whole_page_challenge(self):
        html = '<title>Board</title><div class="post-message">User-written post</div><form><div class="cf-turnstile"></div></form>'
        self.assertFalse(is_challenge(html))
        self.assertTrue(site_ready(html, 'soyjak.st'))
        self.assertTrue(site_ready('<div class="boardlist"><a href="/soy/">Board</a></div>', 'soyjak.st'))
        self.assertFalse(site_ready('<title>Just a moment...</title><div class="boardlist"></div>', 'soyjak.st'))

    def test_board_adopts_only_verified_same_site_html_tabs_without_http_fallback(self):
        blocked, unrelated, media, verified = [AsyncMock() for _ in range(4)]
        for page in (blocked, unrelated, media, verified):
            page.is_closed = MagicMock(return_value=False)
            page.on = MagicMock()
            page.remove_listener = MagicMock()
        blocked.url = 'https://soyjak.st/'
        blocked.goto.return_value = MagicMock(status=403, headers={'content-type': 'text/html'})
        blocked.content.return_value = '<title>Just a moment...</title><form id="challenge-form"></form>'
        unrelated.url = 'https://swedishwin.com/b/'
        media.url = 'https://soyjak.st/sound.mp3'
        verified.url = 'https://www.soyjak.st/soy/'
        verified.content.return_value = '<title>Board</title><div class="post-message">Actual post text</div>'
        blocked.context.pages = [blocked, unrelated, media, verified]
        select = MagicMock()
        with patch('scratchllm.crawl.curl_fetch') as fallback:
            result = asyncio.run(fetch_page(blocked, blocked.url, 'soyjak.st', 0, 1, select_page=select))
        self.assertEqual(result, (verified.url, verified.content.return_value, 'camoufox'))
        unrelated.content.assert_not_awaited()
        media.content.assert_not_awaited()
        select.assert_called_once_with(verified)
        fallback.assert_not_called()
        blocked.remove_listener.assert_called_once()
        blocked.context.pages = [unrelated, media]
        with self.assertRaisesRegex(RuntimeError, 'No verified board page'):
            asyncio.run(wait_for_site(blocked, 'soyjak.st', 0))


if __name__ == '__main__':
    unittest.main()
