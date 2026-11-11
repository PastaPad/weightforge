"""Selection/persistence checks with a fake model; no training or generation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from fastapi.testclient import TestClient
from scratchllm.jobs import Jobs
from scratchllm.serve import create_app


class FakeModel:
    def to(self, device):
        return self

    def config_dict(self):
        return {"vocab_size": 1, "context": 1, "width": 1, "layers": 1, "heads": 1}


class FakeGenerator:
    def __init__(self, checkpoint, device):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.model = FakeModel()
        self.device = device


class SelectionTests(unittest.TestCase):
    def test_missing_checkpoint_keeps_tools_available_and_allows_explicit_loading(self):
        for saved_selection in (False, True):
            with self.subTest(saved_selection=saved_selection), tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory, \
                    patch('scratchllm.serve.Generator', FakeGenerator), patch('scratchllm.serve.torch.cuda.is_available', return_value=False):
                root = Path(directory)
                (root/'available.pt').write_text('fake')
                state = root/'runs/serving.json'
                if saved_selection:
                    state.parent.mkdir()
                    state.write_text(json.dumps({'checkpoint': 'missing.pt'}))
                fallback = root/('available.pt' if saved_selection else 'missing.pt')
                with TestClient(create_app(fallback, 'cpu', workspace=root, state_path='runs/serving.json')) as client:
                    health = client.get('/health').json()
                    self.assertEqual(health['status'], 'ok')
                    self.assertIsNone(health['model'])
                    self.assertIsNone(health['checkpoint'])
                    self.assertIn('missing.pt', health['checkpoint_error'])
                    self.assertEqual(client.get('/').status_code, 200)
                    self.assertEqual(client.get('/huggingface/assets').status_code, 200)
                    self.assertEqual(client.get('/jobs').status_code, 200)
                    for endpoint in ('/complete', '/chat', '/workspace/chat'):
                        response = client.post(endpoint, json={'prompt': 'test'})
                        self.assertEqual(response.status_code, 409)
                        self.assertIn('No native checkpoint', response.json()['detail'])
                    with patch.object(Jobs, 'start', return_value={'job': None}) as start:
                        self.assertEqual(client.post('/jobs/hf-finetune', json={}).status_code, 200)
                        self.assertEqual(start.call_args.args[0], 'hf-finetune')
                    self.assertEqual(state.exists(), saved_selection)
                    if saved_selection:
                        self.assertEqual(json.loads(state.read_text())['checkpoint'], 'missing.pt')
                    self.assertEqual(client.post('/model/load', json={'checkpoint': 'missing.pt'}).status_code, 400)
                    self.assertIsNone(client.get('/health').json()['model'])
                    self.assertEqual(client.post('/model/load', json={'checkpoint': 'available.pt'}).status_code, 200)
                    health = client.get('/health').json()
                    self.assertEqual(health['checkpoint'], 'available.pt')
                    self.assertIsNotNone(health['model'])
                    self.assertIsNone(health['checkpoint_error'])
                    self.assertEqual(json.loads(state.read_text())['checkpoint'], 'available.pt')

    def test_browser_snapshot_is_scoped_to_current_job(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator):
            root = Path(directory)
            (root / 'fake.pt').write_text('fake')
            logs = root / 'runs/jobs'
            logs.mkdir(parents=True)
            (logs / 'test.log').write_text('WAIT verification')
            (logs / 'test.progress.browser.png').write_bytes(b'png-test-data')
            (logs / 'latest.json').write_text(json.dumps(dict(id='test', kind='crawl', status='stopped',
                log_path='runs/jobs/test.log', progress_path='runs/jobs/test.progress.json')))
            with TestClient(create_app(root/'fake.pt', 'cpu', workspace=root)) as client:
                self.assertTrue(client.get('/jobs').json()['job']['browser_snapshot'])
                response = client.get('/jobs/browser?job_id=test')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b'png-test-data')
                self.assertEqual(response.headers['content-type'], 'image/png')
                self.assertEqual(client.get('/jobs/browser?job_id=other').status_code, 404)

    def test_status_and_stop_are_not_blocked_by_inference(self):
        entered, release = Event(), Event()
        class BlockingGenerator(FakeGenerator):
            def complete(self, **kwargs):
                entered.set()
                release.wait(timeout=10)
                return {"text": "fake response"}
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', BlockingGenerator):
            root = Path(directory)
            checkpoint = root / 'fake.pt'
            checkpoint.write_text('fake')
            with TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client, ThreadPoolExecutor(max_workers=2) as pool:
                inference = pool.submit(client.post, '/complete', json={'prompt': 'test'})
                try:
                    self.assertTrue(entered.wait(timeout=2))
                    status = pool.submit(client.get, '/jobs').result(timeout=2)
                    self.assertEqual(status.status_code, 200)
                    stop = pool.submit(client.post, '/jobs/stop').result(timeout=2)
                    self.assertEqual(stop.status_code, 200)
                finally:
                    release.set()
                    inference.result(timeout=2)

    def test_load_persists_and_restart_uses_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch('scratchllm.serve.torch.cuda.is_available', return_value=False):
            root = Path(directory)
            for name in ('old', 'wiki'):
                (root / name).mkdir()
                (root / name / 'best.pt').write_text('fake')
                (root / name / 'web-job.json').write_text(json.dumps({'data': f'data/{name}'}))
            def app():
                return create_app(root / 'old/best.pt', 'cpu', workspace=root, state_path='runs/serving.json')
            with TestClient(app()) as client:
                result = client.post('/model/load', json={'checkpoint': 'wiki/best.pt'})
                self.assertEqual(result.status_code, 200, result.text)
                health = client.get('/health').json()
                self.assertEqual(health['checkpoint'], 'wiki/best.pt')
                self.assertEqual(health['dataset'], 'data/wiki')
                self.assertTrue(health['selection_persistent'])
                self.assertEqual(client.post('/model/load', json={'checkpoint': 'missing.pt'}).status_code, 400)
            self.assertEqual(json.loads((root / 'runs/serving.json').read_text())['checkpoint'], 'wiki/best.pt')
            with TestClient(app()) as client:
                health = client.get('/health').json()
                self.assertEqual(Path(health['checkpoint']), root / 'wiki/best.pt')
                self.assertEqual(health['dataset'], 'data/wiki')


if __name__ == '__main__':
    unittest.main()
