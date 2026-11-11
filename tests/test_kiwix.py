"""Real offline ZIM import, resume, download integrity, and job wiring; no model training."""
import asyncio
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scratchllm import crawl
from scratchllm.corpus import corpus_plan
from scratchllm.frontier import Frontier, read_state
from scratchllm.jobs import CrawlRequest, Jobs, KiwixRequest, PipelineRequest
from scratchllm.kiwix import BASE_URL, DEFAULT_FILE, DEFAULT_PATH, archive_info, download_archive, open_archive, permutation, article_entry
from scratchllm.links import TEXT_POLICY_VERSION
from scratchllm.progress import CrawlProgress
from scratchllm.serve import create_app
from kiwix_fixture import make_archive
from test_serving_selection import FakeGenerator
import test_crawl_lifecycle


class KiwixTests(unittest.TestCase):
    def args(self, root, **values):
        request = CrawlRequest(hosts=['wikipedia.org'], output=str(root/'corpus.jsonl'), wikipedia_zim=str(root/'wiki.zim'), **values)
        return SimpleNamespace(**request.model_dump(), render_wait_ms=0, profile=str(root/'profile'), progress_file=None)

    def collect(self, args):
        with patch('scratchllm.crawl.browser_session', side_effect=AssertionError('Wikipedia opened a browser')), \
                patch('scratchllm.crawl.urlopen', side_effect=AssertionError('Wikipedia used the network')):
            asyncio.run(crawl.run(args))

    def rows(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def test_archive_import_text_only_provenance_dedupe_resume_and_grouped_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/'wiki.zim')
            args = self.args(root, pages_per_site=2)
            self.collect(args)
            first = self.rows(args.output)
            initial = read_state(args.output)
            self.assertEqual(initial['stats']['completed'], 2)
            self.assertGreater(initial['pending'], 0)
            self.assertFalse((root/'profile').exists())
            args.resume, args.pages_per_site = True, 100
            self.collect(args)
            rows = self.rows(args.output)
            saved = read_state(args.output)
            self.assertEqual(saved['pending'], 0)
            self.assertEqual(saved['kiwix']['cursor'], archive_info(root/'wiki.zim')['entries'])
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[:len(first)], first)
            self.assertEqual(saved['stats']['duplicates'], 1)
            self.assertEqual(saved['stats']['completed'], 11)  # Redirects/resources do not consume the article budget.
            for row in rows:
                self.assertEqual(row['transport'], 'kiwix-zim')
                self.assertEqual(row['host'], 'wikipedia.org')
                self.assertEqual(row['text_policy'], TEXT_POLICY_VERSION)
                self.assertEqual(row['group'], row['url'])
                self.assertEqual(row['archive']['uuid'], saved['settings']['wikipedia_archive']['uuid'])
                self.assertIn('written word', row['text'])
                self.assertIn('Article conclusion', row['text'])
                for unwanted in ('UI', 'MEDIA', 'REF', 'https://', 'HOME', 'CATEGORY', 'FILE', 'TALK', 'LANDING', 'UNRECOGNIZED'):
                    self.assertNotIn(unwanted, row['text'])
            _, report = corpus_plan(Path(args.output))
            self.assertTrue(report['ready'])
            self.assertEqual(report['sources']['wikipedia.org']['accepted_records'], 8)
            second = copy.copy(args)
            second.output, second.resume = str(root/'all.jsonl'), False
            self.collect(second)
            self.assertEqual([row['text'] for row in rows], [row['text'] for row in self.rows(second.output)])
            before = Path(args.output).read_bytes()
            stat = (root/'wiki.zim').stat()
            os.utime(root/'wiki.zim', ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
            with self.assertRaisesRegex(ValueError, 'saved collection settings'):
                self.collect(args)
            self.assertEqual(Path(args.output).read_bytes(), before)

    def test_stop_commits_cursor_and_resume_keeps_original_archive_and_seed(self):
        class StopAfterOne(CrawlProgress):
            stopped = False
            def update(self, *args, **kwargs):
                super().update(*args, **kwargs)
                if self.state['saved'] and not self.stopped:
                    self.stopped = True
                    asyncio.current_task().cancel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/'wiki.zim')
            args = self.args(root, pages_per_site=100)
            with patch('scratchllm.crawl.CrawlProgress', StopAfterOne):
                self.collect(args)
            saved = read_state(args.output)
            self.assertEqual(saved['stats']['saved'], 1)
            self.assertGreater(saved['kiwix']['cursor'], 0)
            jobs = Jobs(root, 'cpu')
            restored = jobs._crawl_request(CrawlRequest(output='corpus.jsonl', resume=True, wikipedia_zim='missing.zim', wikipedia_seed=5))
            self.assertEqual(restored.wikipedia_seed, 42)
            self.assertEqual(restored.wikipedia_zim, str(root/'wiki.zim'))
            args.resume = True
            self.collect(args)
            self.assertEqual(len(self.rows(args.output)), 8)

    def test_missing_foreign_and_legacy_web_archives_fail_before_workers(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.jobs.subprocess.Popen') as spawn:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            request = CrawlRequest(hosts=['wikipedia.org'], wikipedia_zim='wiki.zim')
            with self.assertRaisesRegex(ValueError, 'download it first'):
                jobs.start('crawl', request)
            with self.assertRaises(ValueError):
                jobs.start('pipeline', PipelineRequest(crawl=request))
            make_archive(root/'wiki.zim', source='fr.wikipedia.org')
            with self.assertRaisesRegex(ValueError, 'English'):
                jobs.start('crawl', request)
            old = Frontier(root/'old.jsonl', dict(hosts=['wikipedia.org'], pages_per_site=10))
            old.close()
            with self.assertRaisesRegex(ValueError, 'legacy Wikipedia web crawl'):
                jobs.start('crawl', CrawlRequest(output='old.jsonl', resume=True))
            spawn.assert_not_called()

    def test_spread_order_covers_every_entry_and_media_body_is_never_read(self):
        for length in (1, 8, 99, 1024, 930735):
            start, stride = permutation(length, 42)
            self.assertEqual(len({(start + stride * n) % length for n in range(length)}), length)
        class Media:
            mimetype = 'image/png'
            @property
            def content(self):
                raise AssertionError('Read media bytes')
        entry = SimpleNamespace(is_redirect=False, path='image.png', get_item=lambda: Media())
        archive = SimpleNamespace(has_new_namespace_scheme=True, _get_entry_by_id=lambda index: entry)
        self.assertIsNone(article_entry(archive, 0, 'en.wikipedia.org'))

    def test_supervised_import_and_api_archive_discovery(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator):
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            (root/'old.pt').write_text('fake')
            with TestClient(create_app(root/'old.pt', 'cpu', workspace=root)) as client:
                rows = client.get('/wikipedia/archives').json()['archives']
                self.assertEqual(rows[0]['path'], DEFAULT_PATH)
                response = client.post('/jobs/crawl', json=dict(hosts=['wikipedia.org'], output='corpus.jsonl', pages_per_site=100))
                self.assertEqual(response.status_code, 200, response.text)
                helper = test_crawl_lifecycle.CrawlLifecycleTests()
                # Real worker; no browser, HTTP requests, or model optimization.
                class Status:
                    def status(self):
                        return client.get('/jobs').json()
                result = helper.wait_for(Status(), lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertFalse(result['browser_snapshot'])
                self.assertEqual(result['progress']['mode'], 'kiwix')
                self.assertEqual(len(self.rows(root/'corpus.jsonl')), 8)
                self.assertFalse(client.get('/health').json()['training'])
                self.assertEqual(client.get('/health').json()['checkpoint'], str(root/'old.pt'))

    def test_mixed_collection_only_browses_other_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            jobs = Jobs(root, 'cpu')
            helper = test_crawl_lifecycle.CrawlLifecycleTests()
            try:
                helper.browser_job(jobs, 'resume', CrawlRequest(hosts=['wikipedia.org', 'soyjakwiki.org'], output='corpus.jsonl',
                    pages_per_site=2, depth=1, delay=0.5, headed=False, wikipedia_all=True))
                result = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                visits = [json.loads(line)['url'] for line in (root/'corpus.requests.jsonl').read_text().splitlines()]
                self.assertEqual(visits, ['https://soyjakwiki.org/', 'https://soyjakwiki.org/empty'])
                self.assertEqual(result['progress']['sources']['wikipedia.org']['saved'], 8)
                self.assertEqual(result['progress']['completed_units'], result['progress']['budget'])
                self.assertEqual(result['progress']['sources']['soyjakwiki.org']['saved'], 1)
                helper.assert_closed(jobs)
            finally:
                jobs.close()


class DownloadTests(unittest.TestCase):
    def test_range_resume_checksum_and_archive_verification_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = make_archive(root/'fixture.zim').read_bytes()
            checksum = hashlib.sha256(payload).hexdigest()
            url, output = BASE_URL + DEFAULT_FILE, root/'download.zim'
            part = Path(str(output) + '.part')
            offset = len(payload) // 2
            part.write_bytes(payload[:offset])
            Path(str(output) + '.download.json').write_text(json.dumps(dict(url=url, sha256=checksum, size=len(payload))))
            calls = []
            class Response(io.BytesIO):
                def __init__(self, data=b'', status=200, **headers):
                    super().__init__(data)
                    self.status, self.headers = status, headers
            def fetch(request, **kwargs):
                if isinstance(request, str):
                    self.assertEqual(request, url + '.sha256')
                    return Response(f'{checksum}  {DEFAULT_FILE}'.encode())
                if request.get_method() == 'HEAD':
                    return Response(**{'Content-Length': str(len(payload))})
                calls.append(request)
                self.assertEqual(request.get_header('Range'), f'bytes={offset}-')
                return Response(payload[offset:], 206, **{'Content-Range': f'bytes {offset}-{len(payload)-1}/{len(payload)}'})
            progress = CrawlProgress(None, 0)
            with patch('scratchllm.kiwix.urlopen', side_effect=fetch):
                download_archive(url, output, progress, Event())
            self.assertEqual(output.read_bytes(), payload)
            self.assertFalse(part.exists())
            self.assertEqual(len(calls), 1)
            self.assertEqual(progress.state['phase'], 'completed')
            self.assertTrue(json.loads(Path(str(output) + '.download.json').read_text())['verified'])

            # A fully downloaded file with the wrong digest must remain unpublished.
            output.unlink()
            part.write_bytes(b'x' * len(payload))
            with patch('scratchllm.kiwix.urlopen', side_effect=fetch):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    download_archive(url, output, CrawlProgress(None, 0), Event())
            self.assertFalse(output.exists())
            self.assertTrue(part.exists())

    def test_download_request_limits_to_official_english_archives(self):
        self.assertEqual(KiwixRequest().url, BASE_URL + DEFAULT_FILE)
        for url in ('http://download.kiwix.org/zim/wikipedia/' + DEFAULT_FILE,
                    BASE_URL.replace('download.kiwix.org', 'evil.test') + DEFAULT_FILE,
                    BASE_URL + 'wikipedia_fr_all_nopic_2026-01.zim', BASE_URL + DEFAULT_FILE + '?other=1'):
            with self.assertRaises(ValueError):
                KiwixRequest(url=url)


if __name__ == '__main__':
    unittest.main()
