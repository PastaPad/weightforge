"""Atomic HF GGUF publication, cancellation and API supervision; no optimizer steps."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import psutil
from fastapi.testclient import TestClient

from scratchllm.hf_export import export_model, run_converter
from scratchllm.hf_requests import HFExportRequest
from scratchllm.huggingface import Stopped, sha256
from scratchllm.jobs import Jobs
from scratchllm.progress import CrawlProgress
from scratchllm.serve import create_app
from test_huggingface import fixture


class HFExportTests(unittest.TestCase):
    def test_export_preserves_source_checks_tokenizer_and_publishes_without_overwriting(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            model, _, _, _, _ = fixture(root)
            original = {str(p.relative_to(model)): sha256(p) for p in model.iterdir() if p.is_file()}
            request = HFExportRequest(model=str(model), output=str(root/'exports/ready.gguf'), precision='f32')
            def convert(converter, staged, output, precision, progress, cancel):
                config = json.loads((staged/'tokenizer_config.json').read_text())
                self.assertTrue(config['add_bos_token'])
                self.assertFalse(config['add_eos_token'])
                self.assertTrue(next(staged.glob('*.safetensors')).is_symlink())
                shutil.copyfile(root/'model.gguf', output)
            with patch('scratchllm.hf_export.run_converter', side_effect=convert) as converter:
                result = export_model(request)
                self.assertEqual(result['sha256'], sha256(request.output))
                self.assertGreater(result['tensors'], 0)
                self.assertEqual(result['architecture'], 'llama')
                with self.assertRaisesRegex(ValueError, 'fresh GGUF'):
                    export_model(request)
                self.assertEqual(converter.call_count, 1)
            self.assertEqual(original, {str(p.relative_to(model)): sha256(p) for p in model.iterdir() if p.is_file()})
            self.assertFalse(list((root/'exports').glob('.hf-export-*')))
            request.output = str(root/'exports/cancelled.gguf')
            cancelled = Event()
            cancelled.set()
            with self.assertRaises(Stopped):
                export_model(request, cancel=cancelled)
            self.assertFalse(Path(request.output).exists())
            # A converter failure must leave no apparently complete output or staging directory.
            with patch('scratchllm.hf_export.run_converter', side_effect=ValueError('conversion failed')):
                with self.assertRaisesRegex(ValueError, 'conversion failed'):
                    export_model(request)
            self.assertFalse(Path(request.output).exists())
            self.assertFalse(list((root/'exports').glob('.hf-export-*')))
            source = next(model.glob('*.safetensors'))
            with source.open('ab') as stream:
                stream.write(b'changed')
            with patch('scratchllm.hf_export.run_converter', side_effect=AssertionError('Changed source reached converter')):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    export_model(request)

    def test_stop_terminates_converter_child(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            (root/'convert_hf_to_gguf.py').write_text(
                'import os, sys, time\nfrom pathlib import Path\n'
                'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
                'assert os.environ["HF_HUB_OFFLINE"] == "1"\n'
                'Path(sys.argv[sys.argv.index("--outfile") + 1]).write_text(str(os.getpid()))\n'
                'time.sleep(60)\n')
            cancel = Event()
            class StopOnProgress:
                def update(self, **state):
                    if state['bytes_written']:
                        cancel.set()
            output = root/'pending.gguf'
            with self.assertRaises(Stopped):
                run_converter(root, root, output, 'f16', StopOnProgress(), cancel)
            self.assertFalse(psutil.pid_exists(int(output.read_text())))

    def test_api_launch_is_cpu_scoped_and_download_requires_complete_export(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            fixture(root)
            with TestClient(create_app(root/'missing.pt', 'cpu', workspace=root)) as client:
                with patch('scratchllm.jobs.subprocess.Popen', side_effect=RuntimeError('intercept export')) as spawn:
                    with self.assertRaisesRegex(RuntimeError, 'intercept export'):
                        client.post('/jobs/hf-export', json={'model': 'models/base', 'output': 'exports/result.gguf'})
                    self.assertIn('scratchllm.hf_export', spawn.call_args.args[0])
                    self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(client.post('/jobs/hf-export', json={'output': '../outside.gguf'}).status_code, 400)
                self.assertEqual(client.post('/jobs/hf-export', json={'output': 'exports/wrong.pt'}).status_code, 422)
                self.assertEqual(client.get('/huggingface/gguf?job_id=missing').status_code, 404)
                output = root/'exports/result.gguf'
                output.parent.mkdir()
                shutil.copyfile(root/'model.gguf', output)
                job = dict(id='test', kind='hf-export', status='running', output='exports/result.gguf', export_ready=True)
                with patch.object(Jobs, 'status', return_value={'job': job}):
                    self.assertEqual(client.get('/huggingface/gguf?job_id=test').status_code, 404)
                    job['status'] = 'succeeded'
                    response = client.get('/huggingface/gguf?job_id=test')
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, output.read_bytes())
                    self.assertEqual(client.get('/huggingface/gguf?job_id=other').status_code, 404)


if __name__ == '__main__':
    unittest.main()
