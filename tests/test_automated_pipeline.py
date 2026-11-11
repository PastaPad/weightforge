"""Real orchestration, isolated file-only stage workers: no collection or model training."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from scratchllm.jobs import CrawlRequest, Jobs, PipelineRequest, PrepareRequest, TrainRequest, DialogueRequest
from scratchllm.serve import create_app
from test_serving_selection import FakeGenerator


class AutomatedPipelineTests(unittest.TestCase):
    def spawn_fixture(self, mode):
        real_popen = subprocess.Popen
        fixture = str(Path(__file__).with_name('pipeline_fixture_worker.py'))
        def spawn(command, **kwargs):
            self.assertEqual(command[3], 'scratchllm.pipeline')
            return real_popen([sys.executable, '-u', fixture, 'driver', mode, *command[4:]], **kwargs)
        return patch('scratchllm.jobs.subprocess.Popen', side_effect=spawn)

    def request(self, **kwargs):
        return PipelineRequest(crawl=CrawlRequest(hosts=['soyjak.st', 'swedishwin.com']),
                               train=TrainRequest(budget_mode='passes', max_epochs=20, steps=987), **kwargs)

    def wait_for(self, status, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = status()['job']
            if predicate(job):
                return job
            time.sleep(0.05)
        self.fail('Pipeline did not reach expected state: ' + json.dumps(status()))

    def finished(self, jobs):
        self.assertTrue(jobs.finished.wait(timeout=25), jobs.status())
        result = jobs.status()['job']
        self.assertFalse(jobs.active())
        self.assertEqual(jobs.job.get('remaining_pids'), [], result['log'])
        self.assertFalse(any(jobs._live_process(row) for row in jobs.job['processes']))
        return result

    def events(self, root):
        return [json.loads(line) for line in (root/'fixture-events.jsonl').read_text().splitlines()]

    def test_server_runs_all_stages_without_status_polling_and_persists_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            try:
                with self.spawn_fixture('complete'):
                    initial = jobs.start('pipeline', self.request())['job']
                self.assertTrue(jobs.training())
                with self.assertRaisesRegex(ValueError, 'already running'):
                    jobs.start('crawl', CrawlRequest())
                # The server's supervisor drives every transition; no status endpoint or UI is needed.
                result = self.finished(jobs)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertEqual(result['pipeline']['phase'], 'completed')
                self.assertEqual([row['status'] for row in result['pipeline']['stages']], ['succeeded'] * 3)
                events = self.events(root)
                self.assertEqual([row['kind'] for row in events], ['crawl', 'prepare', 'train'])
                paths = result['pipeline']['outputs']
                self.assertIn(str(root/paths['corpus']), events[1]['arguments'])
                self.assertIn(str(root/paths['dataset']), events[2]['arguments'])
                self.assertIn('soyjak.st', events[0]['arguments'])
                config = json.loads((root/paths['run']/'web-config.json').read_text())
                self.assertEqual(config['steps'], 3876)  # Uses the newly prepared data, not stale 987.
                self.assertEqual(result['metrics'][-1]['stop_reason'], 'early_stopping')
                self.assertEqual(result['checkpoint'], paths['checkpoint'])
                self.assertEqual(result['pipeline']['corpus_report']['duplicates_removed'], 7)
                self.assertEqual(result['pipeline']['corpus_report']['sources']['soyjak.st']['tokens']['train'], 1587273)
                self.assertEqual(json.loads(jobs.state_path.read_text())['id'], initial['id'])
                self.assertEqual(Jobs(root, 'cpu').status()['job']['pipeline'], result['pipeline'])
                self.assertEqual(jobs.prepared_datasets()['latest'], paths['dataset'])
                self.assertFalse(jobs.training())
                for kind in ['crawl', 'prepare', 'train']:
                    self.assertIn(f'FIXTURE DONE {kind}', result['log'])

                (root/'old.pt').write_text('old fixture')
                with patch('scratchllm.serve.Generator', FakeGenerator), TestClient(create_app(root/'old.pt', 'cpu', workspace=root)) as client:
                    self.assertEqual(client.get('/jobs').json()['job']['checkpoint'], paths['checkpoint'])
                    self.assertEqual(client.get('/health').json()['checkpoint'], str(root/'old.pt'))
                    self.assertEqual(client.get('/jobs/browser', params={'job_id': result['id']}).content, b'fixture-screenshot')
                    self.assertEqual(client.get('/jobs/browser', params={'job_id': 'other'}).status_code, 404)
            finally:
                jobs.close()

    def test_failed_stages_and_invalid_dataset_cancel_downstream_work(self):
        for mode, expected in [('fail-crawl', ['crawl']), ('fail-prepare', ['crawl', 'prepare']),
                               ('small-data', ['crawl', 'prepare']), ('fail-train', ['crawl', 'prepare', 'train']),
                               ('missing-best', ['crawl', 'prepare', 'train'])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = Jobs(root, 'cpu')
                try:
                    with self.spawn_fixture(mode):
                        jobs.start('pipeline', self.request())
                    result = self.finished(jobs)
                    self.assertEqual(result['status'], 'failed', result['log'])
                    self.assertEqual(result['pipeline']['phase'], 'failed')
                    self.assertEqual([event['kind'] for event in self.events(root)], expected)
                    self.assertNotIn('pending', [stage['status'] for stage in result['pipeline']['stages']])
                    self.assertNotIn('checkpoint', result)
                finally:
                    jobs.close()

    def test_stop_at_each_stage_cancels_rest_and_training_saves_before_exit(self):
        for index, kind in enumerate(['crawl', 'prepare', 'train']):
            with self.subTest(stage=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = Jobs(root, 'cpu')
                try:
                    with self.spawn_fixture(f'wait-{kind}'):
                        jobs.start('pipeline', self.request())
                    self.wait_for(jobs.status, lambda job: f'FIXTURE READY {kind}' in job['log'])
                    jobs.stop()
                    result = self.finished(jobs)
                    self.assertEqual(result['status'], 'stopped', result['log'])
                    self.assertFalse(result['forced_stop'])
                    self.assertEqual(result['pipeline']['phase'], 'stopped')
                    self.assertEqual([event['kind'] for event in self.events(root)], ['crawl', 'prepare', 'train'][:index + 1])
                    if kind == 'train':
                        self.assertTrue((root/result['output']/'last.pt').is_file())
                        self.assertEqual(result['metrics'][-1]['stop_reason'], 'user_stop')
                finally:
                    jobs.close()

    def test_browser_window_closure_does_not_advance_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                with self.spawn_fixture('browser-closed'):
                    jobs.start('pipeline', self.request())
                result = self.finished(jobs)
                self.assertEqual(result['status'], 'stopped', result['log'])
                self.assertEqual([event['kind'] for event in self.events(Path(directory))], ['crawl'])
            finally:
                jobs.close()

    def test_api_reserves_gpu_through_pipeline_and_stop_is_responsive(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator):
            root = Path(directory)
            (root/'old.pt').write_text('old fixture')
            with TestClient(create_app(root/'old.pt', 'cpu', workspace=root)) as client:
                with self.spawn_fixture('wait-crawl'):
                    response = client.post('/jobs/pipeline', json=self.request().model_dump())
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(client.get('/health').json()['training'])
                self.assertEqual(client.post('/complete', json={'prompt': 'test'}).status_code, 409)
                self.assertEqual(client.post('/model/load', json={'checkpoint': 'old.pt'}).status_code, 409)
                status = lambda: client.get('/jobs').json()
                self.wait_for(status, lambda job: 'FIXTURE READY crawl' in job['log'])
                self.assertEqual(client.post('/jobs/stop').status_code, 200)
                job = self.wait_for(status, lambda job: job['status'] not in Jobs.LIVE)
                self.assertEqual(job['status'], 'stopped', job['log'])
                self.assertFalse(client.get('/health').json()['training'])

    def test_coordinator_crash_cleans_stage_process_and_reports_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                with self.spawn_fixture('wait-prepare'):
                    jobs.start('pipeline', self.request())
                self.wait_for(jobs.status, lambda job: 'FIXTURE READY prepare' in job['log'])
                jobs.process.kill()
                result = self.finished(jobs)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['pipeline']['phase'], 'failed')
                self.assertEqual([stage['status'] for stage in result['pipeline']['stages']], ['succeeded', 'failed', 'cancelled'])
                self.assertEqual([event['kind'] for event in self.events(Path(directory))], ['crawl', 'prepare'])
            finally:
                jobs.close()

    def test_pipeline_paths_are_linked_fresh_and_validated_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.jobs.subprocess.Popen') as spawn:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            (root/'occupied').mkdir()
            (root/'occupied/user-data').write_text('keep')
            request = self.request(auto_paths=False, prepare=PrepareRequest(output='occupied'))
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                jobs.start('pipeline', request)
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                jobs.start('pipeline', self.request(auto_paths=False, prepare=PrepareRequest(output='data')))
            with self.assertRaisesRegex(ValueError, 'inside the workspace'):
                jobs.start('pipeline', self.request(auto_paths=False, prepare=PrepareRequest(output='../outside')))
            resolved = jobs._pipeline_request(self.request(), 'example')
            self.assertEqual(resolved.crawl.output, 'data/raw-pipeline-example.jsonl')
            self.assertEqual(resolved.prepare.input, resolved.crawl.output)
            self.assertEqual(resolved.train.data, resolved.prepare.output)
            self.assertEqual((root/'occupied/user-data').read_text(), 'keep')
            with self.assertRaises(ValueError):
                PipelineRequest(train=TrainRequest(resume=True))
            spawn.assert_not_called()

    def test_dialogue_pipeline_keeps_base_and_returns_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = Jobs(root, 'cpu')
            try:
                with self.spawn_fixture('complete'):
                    jobs.start('pipeline', self.request(dialogue=DialogueRequest()))
                result = self.finished(jobs)
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertEqual([event['kind'] for event in self.events(root)], ['crawl', 'prepare', 'train', 'dialogue', 'finetune'])
                self.assertEqual([stage['status'] for stage in result['pipeline']['stages']], ['succeeded'] * 5)
                outputs = result['pipeline']['outputs']
                self.assertEqual(result['checkpoint'], outputs['checkpoint'])
                self.assertNotEqual(outputs['run'], outputs['base_run'])
                self.assertTrue((root/outputs['base_run']/'best.pt').is_file())
                config = json.loads((root/outputs['run']/'web-config.json').read_text())
                self.assertEqual(config['steps'], 6)  # 32 examples * 3 passes / 16 per update.
                self.assertEqual(config['learning_rate'], 0.00005)
                self.assertEqual(result['pipeline']['dialogue_report']['groups'], 8)
                self.assertEqual(jobs.prepared_datasets()['latest'], outputs['dataset'])
                self.assertEqual(jobs.prepared_datasets()['dialogue_datasets'][0]['path'], outputs['dialogue_dataset'])
            finally:
                jobs.close()

    def test_dialogue_failure_and_stop_prevent_finetuning(self):
        for mode in ('fail-dialogue', 'wait-dialogue', 'wait-finetune'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jobs = Jobs(root, 'cpu')
                try:
                    with self.spawn_fixture(mode):
                        jobs.start('pipeline', self.request(dialogue=DialogueRequest()))
                    if mode.startswith('wait-'):
                        kind = mode.removeprefix('wait-')
                        self.wait_for(jobs.status, lambda job: f'FIXTURE READY {kind}' in job['log'])
                        self.assertTrue(jobs.training())
                        jobs.stop()
                    result = self.finished(jobs)
                    self.assertEqual(result['status'], 'failed' if mode.startswith('fail') else 'stopped', result['log'])
                    self.assertTrue((root/result['pipeline']['outputs']['base_run']/'best.pt').is_file())
                    if mode != 'wait-finetune':
                        self.assertNotIn('finetune', [event['kind'] for event in self.events(root)])
                        self.assertNotIn('checkpoint', result)
                    else:
                        self.assertTrue((root/result['output']/'last.pt').is_file())
                finally:
                    jobs.close()


if __name__ == '__main__':
    unittest.main()
