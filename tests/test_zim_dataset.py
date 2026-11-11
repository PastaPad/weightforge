"""Whole-archive import and real two-stage dataset preparation; no model training."""
import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scratchllm import crawl
from scratchllm.frontier import read_state, state_path
from scratchllm.jobs import Jobs, CrawlRequest, PrepareRequest, DatasetRequest
from scratchllm.kiwix import DEFAULT_PATH, archive_info
from scratchllm.serve import create_app
from kiwix_fixture import make_archive
import test_crawl_lifecycle
from test_serving_selection import FakeGenerator


class WholeZimTests(unittest.TestCase):
    def driver(self, mode='complete'):
        real_popen = subprocess.Popen
        fixture = str(Path(__file__).with_name('dataset_fixture_worker.py'))
        def spawn(command, **kwargs):
            self.assertEqual(command[3], 'scratchllm.pipeline')
            self.assertIn('--dataset-only', command)
            return real_popen([sys.executable, '-u', fixture, 'driver', mode, *command[4:]], **kwargs)
        return patch('scratchllm.jobs.subprocess.Popen', side_effect=spawn)

    def request(self):
        return DatasetRequest(crawl=CrawlRequest(hosts=['wikipedia.org'], wikipedia_all=True, pages_per_site=1),
                              prepare=PrepareRequest(vocab_size=300, validation_fraction=0.2))

    def finish(self, jobs):
        self.assertTrue(jobs.finished.wait(40), jobs.status())
        result = jobs.status()['job']
        self.assertEqual(result.get('remaining_pids'), [], result['log'])
        return result

    def test_real_api_builds_complete_dataset_with_a_one_article_limit_and_no_training(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator):
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            (root/'model.pt').write_text('fake')
            with TestClient(create_app(root/'model.pt', 'cpu', workspace=root)) as client:
                with self.driver():
                    response = client.post('/jobs/zim-dataset', json=self.request().model_dump())
                self.assertEqual(response.status_code, 200, response.text)
                self.assertFalse(client.get('/health').json()['training'])
                class Status:
                    def status(self):
                        return client.get('/jobs').json()
                result = test_crawl_lifecycle.CrawlLifecycleTests().wait_for(Status(), lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertEqual([stage['kind'] for stage in result['pipeline']['stages']], ['crawl', 'prepare'])
                self.assertNotIn('checkpoint', result)
                self.assertNotIn('run', result['pipeline']['outputs'])
                corpus = root/result['settings']['crawl']['output']
                saved = read_state(corpus)
                self.assertEqual(saved['pending'], 0)
                self.assertEqual(saved['kiwix']['cursor'], archive_info(root/DEFAULT_PATH)['entries'])
                self.assertEqual(saved['stats']['saved'], 8)
                self.assertGreater(saved['stats']['completed'], 1)
                dataset = root/result['output']
                meta = json.loads((dataset/'meta.json').read_text())
                self.assertEqual(sum(meta['documents'].values()), 8)
                self.assertTrue(all(meta['tokens'].values()))
                self.assertEqual(client.get('/datasets').json()['latest'], result['output'])
                self.assertFalse(list(dataset.parent.glob('.prepare-*')))
                events = [json.loads(line) for line in (root/'dataset-events.jsonl').read_text().splitlines()]
                self.assertEqual([row['module'] for row in events], ['scratchllm.crawl', 'scratchllm.prepare'])
                self.assertIn('--wikipedia-all', events[0]['arguments'])
                self.assertIn('--atomic-output', events[1]['arguments'])

    def test_cancel_and_resume_during_import_or_preparation_reuses_complete_corpus(self):
        for mode in ('slow-crawl', 'wait-prepare'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_archive(root/DEFAULT_PATH)
                jobs = Jobs(root, 'cpu')
                try:
                    with self.driver(mode):
                        jobs.start('dataset', self.request())
                    if mode == 'slow-crawl':
                        predicate = lambda job: (job.get('progress') or {}).get('saved', 0) > 0
                    else:
                        predicate = lambda job: 'FIXTURE READY atomic preparation' in job['log']
                    test_crawl_lifecycle.CrawlLifecycleTests().wait_for(jobs, predicate)
                    jobs.stop()
                    stopped = self.finish(jobs)
                    self.assertEqual(stopped['status'], 'stopped', stopped['log'])
                    self.assertFalse(stopped['forced_stop'])
                    self.assertTrue(stopped['resumable_dataset'])
                    self.assertFalse((root/stopped['output']).exists())
                    self.assertFalse(list((root/'data').glob('.prepare-*')))
                    corpus = root/stopped['settings']['crawl']['output']
                    prefix = corpus.read_bytes()
                    resumed = DatasetRequest.model_validate(stopped['settings'])
                    resumed.auto_paths, resumed.crawl.resume = False, True
                    with self.driver():
                        jobs.start('dataset', resumed)
                    result = self.finish(jobs)
                    self.assertEqual(result['status'], 'succeeded', result['log'])
                    self.assertTrue(corpus.read_bytes().startswith(prefix))
                    self.assertEqual(len(corpus.read_text().splitlines()), 8)
                    self.assertEqual(result['output'], stopped['output'])
                    self.assertEqual(read_state(corpus)['pending'], 0)
                finally:
                    jobs.close()

    def test_old_capped_import_can_be_extended_to_the_whole_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            request = CrawlRequest(hosts=['wikipedia.org'], output=str(root/'raw.jsonl'), wikipedia_zim=str(root/DEFAULT_PATH), pages_per_site=1)
            args = SimpleNamespace(**request.model_dump(), render_wait_ms=0, profile=str(root/'profile'), progress_file=str(root/'progress.json'))
            with patch('scratchllm.crawl.browser_session', side_effect=AssertionError('Unexpected browser')):
                asyncio.run(crawl.run(args))
                original = read_state(args.output)
                settings = original['settings']
                settings.pop('wikipedia_all')  # Existing pre-whole-ZIM saved state.
                with closing(sqlite3.connect(state_path(args.output))) as db, db:
                    db.execute('UPDATE metadata SET value=? WHERE key=?', (json.dumps(settings), 'settings'))
                restored = Jobs(root, 'cpu')._crawl_request(CrawlRequest(output='raw.jsonl', resume=True, wikipedia_all=True, pages_per_site=1))
                self.assertTrue(restored.wikipedia_all)
                args.resume, args.wikipedia_all = True, True
                asyncio.run(crawl.run(args))
            self.assertEqual(read_state(args.output)['pending'], 0)
            self.assertEqual(len(Path(args.output).read_text().splitlines()), 8)
            progress = json.loads((root/'progress.json').read_text())
            self.assertTrue(progress['wikipedia_all'])
            self.assertEqual(progress['completed_units'], progress['budget'])
            self.assertEqual(progress['budget'], progress['archive']['entries'])

    def test_data_only_requests_and_output_paths_are_validated(self):
        for request in (dict(crawl={'hosts': ['soyjakwiki.org'], 'wikipedia_all': True}),
                        dict(crawl={'hosts': ['wikipedia.org'], 'wikipedia_all': False})):
            with self.assertRaises(ValueError):
                DatasetRequest.model_validate(request)
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.jobs.subprocess.Popen') as spawn:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            jobs = Jobs(root, 'cpu')
            request = self.request()
            request.auto_paths = False
            request.crawl.output = DEFAULT_PATH
            with self.assertRaises(ValueError):
                jobs.start('dataset', request)
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
