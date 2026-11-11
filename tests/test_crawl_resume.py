"""Durable queue recovery and real-browser offline resume; never trains a model."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scratchllm.frontier import Frontier, corpus_lock, prefix_digest, read_state
from scratchllm.jobs import CrawlRequest, Jobs
import test_crawl_lifecycle


class FrontierTests(unittest.TestCase):
    def test_recovery_retries_only_inflight_and_retains_pending_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'corpus.jsonl'
            settings = dict(hosts=['soyjakwiki.org'], pages_per_site=2, depth=2)
            frontier = Frontier(output, copy.deepcopy(settings))
            try:
                root, depth = frontier.next('soyjakwiki.org')
                self.assertEqual(depth, 0)
                frontier.discover('soyjakwiki.org', [('https://soyjakwiki.org/one', 1), ('https://soyjakwiki.org/two', 1)])
                frontier.checkpoint(dict(attempted=1), 0, hashlib.sha256(b'').hexdigest(), finished=root)
                self.assertEqual(frontier.next('soyjakwiki.org')[0], 'https://soyjakwiki.org/one')
                # Simulate a crash with the second page still active.
            finally:
                frontier.close()
            resumed = Frontier(output, dict(settings, pages_per_site=5), resume=True)
            try:
                self.assertEqual(resumed.next('soyjakwiki.org'), ('https://soyjakwiki.org/one', 1))
                resumed.checkpoint(dict(attempted=2), 0, hashlib.sha256(b'').hexdigest(), finished='https://soyjakwiki.org/one')
                self.assertEqual(resumed.next('soyjakwiki.org'), ('https://soyjakwiki.org/two', 1))
                self.assertEqual(read_state(output)['settings']['pages_per_site'], 5)
                self.assertEqual(read_state(output)['pending'], 1)
            finally:
                resumed.close()

    def test_changed_prefix_or_settings_is_rejected_without_modifying_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'corpus.jsonl'
            output.write_bytes(b'original\n')
            settings = dict(hosts=['soyjakwiki.org'], pages_per_site=2, depth=1)
            frontier = Frontier(output, copy.deepcopy(settings))
            frontier.close()
            with self.assertRaisesRegex(ValueError, 'saved collection settings'):
                Frontier(output, dict(settings, depth=3), resume=True)
            output.write_bytes(b'modified\n')
            with self.assertRaisesRegex(ValueError, 'Corpus changed'):
                Frontier(output, copy.deepcopy(settings), resume=True)
            self.assertEqual(output.read_bytes(), b'modified\n')

    def test_complete_crash_tail_is_preserved_and_lock_excludes_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'corpus.jsonl'
            output.write_bytes(b'committed\n')
            settings = dict(hosts=['soyjakwiki.org'], pages_per_site=2)
            frontier = Frontier(output, copy.deepcopy(settings))
            frontier.close()
            with output.open('ab') as stream:
                stream.write(b'flushed before queue commit\n')
            resumed = Frontier(output, copy.deepcopy(settings), resume=True)
            resumed.close()
            self.assertEqual(prefix_digest(output, len(b'committed\n')), read_state(output)['corpus_digest'])
            self.assertTrue(output.read_bytes().endswith(b'flushed before queue commit\n'))
            with corpus_lock(output):
                with self.assertRaisesRegex(ValueError, 'another worker'):
                    with corpus_lock(output):
                        self.fail('Second writer acquired the corpus')
            with corpus_lock(output):
                pass


class CrawlResumeTests(unittest.TestCase):
    def test_real_worker_extends_saved_budget_without_refetching_completed_pages(self):
        helper = test_crawl_lifecycle.CrawlLifecycleTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            try:
                helper.browser_job(jobs, 'resume', CrawlRequest(output='corpus.jsonl', pages_per_site=1, depth=1, delay=0.5, headed=False))
                first = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(first['status'], 'succeeded', first['log'])
                self.assertEqual(first['resumable_crawl']['pending'], 3)
                self.assertEqual(first['progress']['saved'], 1)
                helper.assert_closed(jobs)
                helper.browser_job(jobs, 'resume', CrawlRequest(output='corpus.jsonl', resume=True, pages_per_site=4, hosts=['swedishwin.com']))
                result = helper.wait_for(jobs, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertEqual(result['settings']['hosts'], ['soyjakwiki.org'])
                self.assertFalse(result['settings']['headed'])
                self.assertEqual(result['progress']['completed'], 4)
                self.assertEqual(result['progress']['saved'], 2)
                self.assertTrue(result['progress']['resumed'])
                self.assertEqual(result['resumable_crawl']['pending'], 0)
                visits = [json.loads(line)['url'] for line in (root/'corpus.requests.jsonl').read_text().splitlines()]
                self.assertEqual(visits, ['https://soyjakwiki.org/', 'https://soyjakwiki.org/empty', 'https://soyjakwiki.org/new', 'https://soyjakwiki.org/same'])
                helper.assert_closed(jobs)
            finally:
                jobs.close()


if __name__ == '__main__':
    unittest.main()
