"""GGUF metadata, atomic snapshots, cancellation and concurrent CPU supervision; no training."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import gguf
import numpy as np
import torch
from fastapi.testclient import TestClient

from scratchllm.gguf_export import CHAT_TEMPLATE, ExportCancelled, export_gguf, load_snapshot
from scratchllm.jobs import Jobs
from scratchllm.progress import CrawlProgress
from scratchllm.serve import create_app
from test_serving_selection import FakeGenerator
from gguf_fixture import checkpoint


def field(reader, name):
    return reader.fields[name].contents()


class GGUFExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_modern_and_legacy_f32_f16_preserve_tensors_and_token_ids(self):
        for architecture in ('modern', 'gpt'):
            for precision in ('f32', 'f16'):
                with self.subTest(architecture=architecture, precision=precision), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, model, tokenizer = checkpoint(root, architecture)
                    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                    out = root/'model.gguf'
                    report = export_gguf(path, out, precision)
                    reader = gguf.GGUFReader(out)
                    self.assertEqual(field(reader, 'general.architecture'), 'llama' if architecture == 'modern' else 'gpt2')
                    self.assertEqual(field(reader, 'tokenizer.ggml.tokens'), [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())])
                    self.assertEqual(field(reader, 'tokenizer.ggml.bos_token_id'), 1)
                    self.assertEqual(field(reader, 'tokenizer.ggml.eos_token_id'), 2)
                    self.assertEqual(field(reader, 'tokenizer.chat_template'), CHAT_TEMPLATE)
                    self.assertEqual(field(reader, 'scratchllm.checkpoint_sha256'), original_hash)
                    self.assertEqual(field(reader, 'scratchllm.checkpoint_step'), 17)
                    tensors = {tensor.name: tensor.data for tensor in reader.tensors}
                    np.testing.assert_allclose(tensors['token_embd.weight'], model.token.weight.detach().numpy(), rtol=0.001 if precision == 'f16' else 0, atol=1e-7 if precision == 'f16' else 0)
                    self.assertNotIn('output.weight', tensors)
                    self.assertEqual(tensors['output_norm.weight'].dtype, np.float32)
                    self.assertEqual(tensors['token_embd.weight'].dtype, np.float16 if precision == 'f16' else np.float32)
                    if architecture == 'modern':
                        key, value = model.blocks[0].attn.key_value.weight.detach().chunk(2, dim=0)
                        for name, expected in [('attn_k', key), ('attn_v', value), ('attn_q', model.blocks[0].attn.query.weight.detach())]:
                            np.testing.assert_allclose(tensors[f'blk.0.{name}.weight'], expected.numpy(), rtol=0.001, atol=1e-7)
                    else:
                        self.assertFalse(tensors['blk.0.attn_qkv.bias'].any())
                        self.assertFalse(tensors['blk.0.attn_output.bias'].any())
                    self.assertEqual(report['sha256'], hashlib.sha256(out.read_bytes()).hexdigest())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
                    with self.assertRaisesRegex(ValueError, 'fresh'):
                        export_gguf(path, out)
                    self.assertFalse(list(root.glob('*.part')))

    def test_cancel_failed_tensor_and_unsupported_tokenizer_never_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = checkpoint(root)
            event = Event()
            class StoppingProgress(CrawlProgress):
                def update(self, *args, **kwargs):
                    super().update(*args, **kwargs)
                    if kwargs.get('completed') == 1:
                        event.set()
            with self.assertRaises(ExportCancelled):
                export_gguf(path, root/'cancelled.gguf', progress=StoppingProgress(None, 0), cancel=event)
            self.assertFalse((root/'cancelled.gguf').exists())
            self.assertFalse(list(root.glob('.*.part')))
            with patch.object(gguf.GGUFWriter, 'write_tensor_data', side_effect=OSError('fixture disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    export_gguf(path, root/'failed.gguf')
            self.assertFalse((root/'failed.gguf').exists())
            self.assertFalse(list(root.glob('.*.part')))
            state = torch.load(path, weights_only=True)
            tokenizer = json.loads(state['tokenizer_json'])
            tokenizer['pre_tokenizer']['add_prefix_space'] = True
            state['tokenizer_json'] = json.dumps(tokenizer)
            torch.save(state, root/'foreign.pt')
            with self.assertRaisesRegex(ValueError, 'ByteLevel'):
                export_gguf(root/'foreign.pt', root/'foreign.gguf')
            with self.assertRaisesRegex(ValueError, 'last.pt'):
                export_gguf(root/'last.pt', root/'resume.gguf')
            self.assertFalse(list(root.glob('*.gguf')))

    def test_atomic_training_replacement_keeps_one_consistent_checkpoint_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = checkpoint(root)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            original = torch.load
            replacement = original(path, weights_only=True)
            replacement['step'] = 18
            torch.save(replacement, root/'replacement.pt')
            def swap(stream, **kwargs):
                os.replace(root/'replacement.pt', path)
                return original(stream, **kwargs)
            with patch('scratchllm.gguf_export.torch.load', side_effect=swap):
                state, sha256 = load_snapshot(path)
            self.assertEqual(state['step'], 17)
            self.assertEqual(sha256, before)
            self.assertEqual(original(path, weights_only=True)['step'], 18)

    def test_export_and_cancel_are_independent_of_active_training_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = checkpoint(root)
            instances = []
            class BusyTrainingJobs(Jobs):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if kwargs.get('state_path') is None:
                        self.job = dict(id='fixture-training', kind='train', status='running', output='runs/active',
                                        log_path='runs/active.log', settings={}, processes=[], progress_path=None)
                        self.close = Mock()  # A logical training reservation; no training process exists.
                    instances.append(self)
            real_popen = subprocess.Popen
            def slow_export(command, **kwargs):
                self.assertEqual(command[3], 'scratchllm.gguf_export')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                return real_popen([sys.executable, '-u', str(Path(__file__).with_name('gguf_fixture.py')), *command[4:]], **kwargs)
            with patch('scratchllm.serve.Jobs', BusyTrainingJobs), patch('scratchllm.serve.Generator', FakeGenerator), \
                    TestClient(create_app(path, 'cpu', workspace=root)) as client:
                before = client.get('/jobs').json()
                self.assertTrue(client.get('/health').json()['training'])
                with patch('scratchllm.jobs.subprocess.Popen', side_effect=slow_export):
                    response = client.post('/exports', json=dict(checkpoint=path.name, output='exports/stop.gguf'))
                self.assertEqual(response.status_code, 200, response.text)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    job = client.get('/exports').json()['job']
                    if (job.get('progress') or {}).get('completed', 0):
                        break
                    time.sleep(0.05)
                else:
                    self.fail(str(job))
                client.post('/exports/stop')
                self.assertTrue(instances[1].finished.wait(15))
                self.assertEqual(client.get('/exports').json()['job']['status'], 'stopped')
                self.assertEqual(client.get('/jobs').json(), before)
                self.assertFalse((root/'exports/stop.gguf').exists())
                response = client.post('/exports', json=dict(checkpoint=path.name, output='exports/model.gguf', precision='f32'))
                self.assertEqual(response.status_code, 200, response.text)
                job_id = response.json()['job']['id']
                self.assertTrue(instances[1].finished.wait(15))
                job = client.get('/exports').json()['job']
                self.assertEqual(job['status'], 'succeeded', job['log'])
                self.assertTrue(job['export_ready'])
                downloaded = client.get('/exports/file', params=dict(job_id=job_id))
                self.assertEqual(downloaded.content[:4], b'GGUF')
                self.assertIn('attachment', downloaded.headers['content-disposition'])
                self.assertEqual(client.get('/exports/file?job_id=other').status_code, 404)
                self.assertEqual(client.get('/jobs').json(), before)
                self.assertEqual(client.get('/health').json()['checkpoint'], str(path))
                self.assertEqual(client.post('/exports', json=dict(checkpoint=path.name, output='../outside.gguf')).status_code, 400)
                self.assertEqual(client.post('/exports', json=dict(checkpoint=path.name, output='exports/model.gguf')).status_code, 400)
            restored = Jobs(root, 'cpu', state_path='runs/exports/latest.json')
            self.assertTrue(restored.status()['job']['export_ready'])


if __name__ == '__main__':
    unittest.main()
