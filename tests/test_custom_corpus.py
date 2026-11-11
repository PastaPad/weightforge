"""Custom-text uploads, frozen preparation, initialization and resume; no model fitting."""
import asyncio
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

from scratchllm.custom_corpus import PreparationStopped, prepare, save_upload
from scratchllm.dialogue import CHAT_FORMAT, TEXT_FORMAT, fingerprint, message_text
from scratchllm.dialogue_training import finetuning_provenance, initialize_from
from scratchllm.generate import Generator
from scratchllm.jobs import CustomCorpusRequest, FineTuneRequest, Jobs, TrainRequest
from scratchllm.model import GPT, ModelConfig, inherit_architecture
from scratchllm.progress import CrawlProgress
from scratchllm.serve import create_app
from scratchllm.train import main as training_main, save_checkpoints
from scratchllm.training import EarlyStopping, ShuffledStream, evaluate, evaluation_plan, make_batch, training_plan
from test_dialogue import fixture_base
from test_serving_selection import FakeGenerator


def fixture_corpus(root, chat=False):
    checkpoint, tokenizer, _ = fixture_base(root, context=256)
    model = GPT(ModelConfig(vocab_size=tokenizer.get_vocab_size(), context=256, width=32, layers=2, heads=4,
                           architecture='modern', kv_heads=2, ffn_width=64, dropout=0))
    state = dict(model=model.state_dict(), model_config=model.config_dict(), tokenizer_json=tokenizer.to_str())
    if chat:
        state['chat_format'] = CHAT_FORMAT
    torch.save(state, checkpoint)
    source = root/'my-corpus.txt'
    passages = [f'Human passage {index} discusses a different subject: {topic}. Läs mer. https://example.org/{index}'
                for index, topic in enumerate(('astronomy', 'botany', 'cooking', 'music', 'history', 'mathematics'))]
    source.write_bytes(('\ufeff' + '\r\n\r\n'.join([*passages, passages[0], 'https://empty.example/'])).encode())
    return checkpoint, source, tokenizer, model


class CustomCorpusTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_upload_validation_unicode_streaming_and_original_provenance(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, _, _, _ = fixture_corpus(root)
            with patch('scratchllm.serve.Generator', FakeGenerator), \
                    patch('scratchllm.jobs.subprocess.Popen', side_effect=AssertionError('Upload must not start any job')), \
                    TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                raw = '\ufeffFirst paragraph.\r\n\r\n日本語, שלום, Räksmörgås.'.encode()
                result = client.post('/corpus/upload?filename=my%20corpus.txt', content=raw)
                self.assertEqual(result.status_code, 200, result.text)
                saved = result.json()
                self.assertEqual(saved['filename'], 'my corpus.txt')
                self.assertEqual(saved['sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual((root/saved['path']).read_bytes(), raw)
                self.assertIsNone(client.get('/jobs').json()['job'])
                second = client.post('/corpus/upload?filename=my%20corpus.txt', content=raw).json()
                self.assertNotEqual(saved['path'], second['path'])
                for filename, content in (('../outside.txt', raw), ('wrong.json', raw), ('bad.txt', b'\xff'),
                                           ('null.txt', b'abc\x00def'), ('empty.txt', b' \r\n\t')):
                    with self.subTest(filename=filename):
                        response = client.post('/corpus/upload', params=dict(filename=filename), content=content)
                        self.assertEqual(response.status_code, 400, response.text)
                with patch('scratchllm.custom_corpus.MAX_BYTES', 4):
                    self.assertEqual(client.post('/corpus/upload?filename=large.txt', content=b'12345').status_code, 413)
                files = client.get('/corpus/uploads').json()['files']
                self.assertEqual({row['path'] for row in files}, {saved['path'], second['path']})
                self.assertFalse(list((root/'data/custom-uploads').glob('*.part')))
                self.assertEqual(len(list((root/'data/custom-uploads').glob('*.json'))), 2)
                self.assertFalse((root.parent/'outside.txt').exists())
            async def stream():
                for byte in raw:
                    yield bytes([byte])
            saved = asyncio.run(save_upload(root, 'streamed.txt', stream()))
            self.assertEqual((root/saved['path']).read_bytes(), raw)

    def test_frozen_tokens_grouped_split_cleaning_and_short_stream_evaluation(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, source, tokenizer, model = fixture_corpus(root)
            hashes = fingerprint(checkpoint), fingerprint(source)
            data = root/'data/custom'
            meta = prepare(checkpoint, source, data, 0.2)
            self.assertEqual(meta['format'], TEXT_FORMAT)
            self.assertIsNone(meta['chat_format'])
            self.assertEqual(meta['custom_report']['duplicates_removed'], 1)
            self.assertEqual(meta['custom_report']['empty_removed'], 1)
            self.assertEqual(meta['custom_report']['groups'], 6)
            self.assertEqual(meta['custom_report']['source']['sha256'], hashes[1])
            self.assertEqual((data/'tokenizer.json').read_text(), tokenizer.to_str())
            groups, streams = {}, {}
            for split in ('train', 'val'):
                rows = [json.loads(line) for line in (data/f'{split}.jsonl').read_text().splitlines()]
                groups[split] = {row['group'] for row in rows}
                streams[split] = np.fromfile(data/f'{split}.bin', dtype='<u4')
                expected = []
                for row in rows:
                    self.assertNotIn('https://', row['text'])
                    self.assertEqual(row['source_sha256'], hashes[1])
                    expected.extend([1, *tokenizer.encode(row['text']).ids, 2])
                self.assertEqual(streams[split].tolist(), expected)
                self.assertEqual(len(expected), meta['tokens'][split])
            self.assertFalse(groups['train'] & groups['val'])
            self.assertLess(len(streams['val']), model.cfg.context)
            jobs = Jobs(root, 'cpu')
            request = FineTuneRequest(checkpoint='base.pt', data='data/custom', width=512, context=512)
            preview = jobs.preview_training(request)
            self.assertEqual(preview['model_config'], model.config_dict())
            self.assertEqual(preview['epoch_unit'], 'tokens')
            self.assertEqual(preview['steps'], math.ceil(3 * (len(streams['train']) - 1) / (16 * 256)))
            cfg = inherit_architecture(request.model_dump(), meta['base_model_config'])
            evaluation = evaluation_plan(streams, cfg)
            self.assertEqual(evaluation['protocol'], 'fixed-token-nll-v1')
            self.assertEqual(evaluation['coverage']['val'], 1)
            self.assertTrue(math.isfinite(evaluate(model, streams['val'], evaluation['windows']['val'], 2, 'cpu')))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            sampler = ShuffledStream(len(streams['val']), cfg['context'], 42)
            windows = sampler.take(2)
            x, y = make_batch(streams['val'], windows, 'cpu')
            self.assertGreater(int((y != -100).sum()), 0)
            self.assertLessEqual(x.shape[1], cfg['context'])
            restored = ShuffledStream(len(streams['val']), cfg['context'], 0)
            restored.load_state_dict(copy.deepcopy(sampler.state_dict()))
            self.assertEqual(sampler.take(10), restored.take(10))
            listed = jobs.prepared_datasets()
            self.assertEqual(listed['datasets'], [])
            self.assertEqual(listed['dialogue_datasets'], [])
            self.assertEqual(listed['finetuning_datasets'][0]['path'], 'data/custom')
            self.assertEqual((fingerprint(checkpoint), fingerprint(source)), hashes)
            with self.assertRaisesRegex(ValueError, 'fine-tuning'):
                jobs.preview_training(TrainRequest(data='data/custom'))

    def test_single_block_uses_disjoint_chunks_and_cancel_never_publishes_dataset(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, source, tokenizer, _ = fixture_corpus(root)
            source.write_text('A single block of text about stars, planets, gardens, food and languages. ' * 3)
            data = root/'single'
            meta = prepare(checkpoint, source, data)
            self.assertEqual(meta['custom_report']['split_unit'], 'nonoverlapping token chunks')
            expected = tokenizer.encode(message_text(source.read_text())).ids
            chunks = []
            for split in ('train', 'val'):
                stream = np.fromfile(data/f'{split}.bin', dtype='<u4')
                offset = 0
                for row in map(json.loads, (data/f'{split}.jsonl').read_text().splitlines()):
                    start, end = row['token_range']
                    self.assertEqual(stream[offset + 1:offset + 1 + end - start].tolist(), expected[start:end])
                    chunks.append((start, end))
                    offset += end - start + 2
            chunks.sort()
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], len(expected))
            self.assertTrue(all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])))
            event = Event()
            class StopAtPublication(CrawlProgress):
                def update(self, *args, **kwargs):
                    super().update(*args, **kwargs)
                    if self.state['phase'] == 'writing' and kwargs.get('completed'):
                        event.set()
            with self.assertRaises(PreparationStopped):
                prepare(checkpoint, source, root/'cancelled', progress=StopAtPublication(None, 0), cancel=event)
            self.assertFalse((root/'cancelled').exists())
            self.assertFalse(list(root.glob('.prepare-*')))
            with self.assertRaisesRegex(ValueError, 'empty'):
                prepare(checkpoint, source, data)
            source.write_text('https://example.org/only-a-link')
            with self.assertRaisesRegex(ValueError, 'distinct'):
                prepare(checkpoint, source, root/'empty')
            self.assertFalse((root/'empty').exists())

    def test_preparation_worker_and_finetuning_launch_resume_inherit_selected_architecture(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, source, _, model = fixture_corpus(root, chat=True)
            jobs = Jobs(root, 'cpu')
            try:
                result = jobs.start('custom-corpus', CustomCorpusRequest(input=source.name, checkpoint=checkpoint.name, output='data/custom'))
                self.assertEqual(result['job']['kind'], 'custom-corpus')
                self.assertTrue(jobs.finished.wait(20))
                result = jobs.status()['job']
                self.assertEqual(result['status'], 'succeeded', result['log'])
                self.assertTrue(result['dataset_ready'])
                self.assertEqual(result['progress']['phase'], 'completed')
                request = FineTuneRequest(checkpoint='base.pt', data='data/custom', out='runs/custom')
                preview = jobs.preview_training(request)
                with patch('scratchllm.jobs.subprocess.Popen', side_effect=RuntimeError('intercept training')) as spawn:
                    with self.assertRaisesRegex(ValueError, 'fine-tuning'):
                        jobs.start('train', TrainRequest(data=request.data))
                    with self.assertRaisesRegex(RuntimeError, 'intercept training'):
                        jobs.start('finetune', request)
                    self.assertIn('--init-from', spawn.call_args.args[0])
                    config = json.loads((root/'runs/custom/web-config.json').read_text())
                    for key, value in model.config_dict().items():
                        if key != 'vocab_size':
                            self.assertEqual(config[key], value)
                    self.assertEqual(config['steps'], preview['steps'])
                    (root/'runs/custom/last.pt').write_text('resume marker only')
                    with self.assertRaisesRegex(RuntimeError, 'intercept training'):
                        jobs.start('finetune', FineTuneRequest(out='runs/custom', resume=True, data='absent', checkpoint='absent'))
                    self.assertIn('--resume', spawn.call_args.args[0])
                    self.assertNotIn('--init-from', spawn.call_args.args[0])
                    self.assertIn(str(root/'data/custom'), spawn.call_args.args[0])
                    self.assertEqual(json.loads((root/'runs/custom/web-config.json').read_text()), config)
            finally:
                jobs.close()

    def test_actual_trainer_initializes_exact_weights_before_optimizer_creation(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, source, tokenizer, model = fixture_corpus(root, chat=True)
            data = root/'data'
            prepare(checkpoint, source, data)
            cfg = FineTuneRequest(gradient_checkpointing=True).model_dump(exclude={'checkpoint', 'data', 'out', 'resume'})
            config = root/'config.json'
            config.write_text(json.dumps(cfg))
            arguments = ['train', '--init-from', str(checkpoint), '--data', str(data), '--out', str(root/'run'),
                         '--config', str(config), '--device', 'cpu', '--precision', 'fp32']
            with patch.object(sys, 'argv', arguments), patch('scratchllm.train.signal.signal'), \
                    patch('scratchllm.train.initialize_from', wraps=initialize_from) as initialize, \
                    patch('scratchllm.train.torch.optim.AdamW', side_effect=RuntimeError('stop before optimizer creation')):
                with self.assertRaisesRegex(RuntimeError, 'stop before optimizer'):
                    training_main()
            initialized = initialize.call_args.args[0]
            self.assertEqual(initialized.config_dict(), model.config_dict())
            self.assertTrue(initialized.gradient_checkpointing)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(initialized.state_dict()[name], value, atol=0, rtol=0)
            self.assertFalse(list((root/'run').glob('*.pt')))
            self.assertEqual(initialize.call_args.args[3], tokenizer.to_str())

    def test_saved_chat_format_and_completed_resume_with_metadata_fingerprint(self):
        for chat in (False, True):
            with self.subTest(chat=chat), tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
                root = Path(directory)
                checkpoint, source, tokenizer, model = fixture_corpus(root, chat=chat)
                data, out = root/'data', root/'run'
                meta = prepare(checkpoint, source, data)
                initialized = GPT(model.cfg)
                initialize_from(initialized, checkpoint, meta, tokenizer.to_str())
                cfg = inherit_architecture(FineTuneRequest().model_dump(exclude={'data', 'out', 'resume', 'checkpoint'}), model.config_dict())
                streams = {split: np.fromfile(data/f'{split}.bin', dtype='<u4') for split in ('train', 'val')}
                plan, evaluation = training_plan(cfg, meta), evaluation_plan(streams, cfg)
                groups = [dict(params=[p for p in initialized.parameters() if p.ndim >= 2], weight_decay=cfg['weight_decay']),
                          dict(params=[p for p in initialized.parameters() if p.ndim < 2], weight_decay=0.0)]
                optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=(0.9, 0.95))
                state = dict(model=initialized.state_dict(), model_config=initialized.config_dict(), training_config=cfg,
                             optimizer=optimizer.state_dict(), tokenizer_json=tokenizer.to_str(), step=plan['steps'], best_val=1.0,
                             data_hashes={**{split: fingerprint(data/f'{split}.bin') for split in streams}, 'meta': fingerprint(data/'meta.json')},
                             evaluation=evaluation, plan=plan, precision='fp32', stop_reason='pass_budget',
                             scaler=torch.amp.GradScaler('cuda', enabled=False).state_dict(), early_stopping=EarlyStopping().state_dict(),
                             rng=np.random.default_rng(1).bit_generator.state, eval_rng=np.random.default_rng(2).bit_generator.state,
                             torch_rng=torch.get_rng_state(), cuda_rng=None,
                             sampler=ShuffledStream(len(streams['train']), cfg['context'], cfg['seed']).state_dict(),
                             **finetuning_provenance(meta))
                out.mkdir()
                save_checkpoints(out, state, initialized, None, 'raw', True)
                inference = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
                self.assertEqual(inference['finetuning_format'], TEXT_FORMAT)
                self.assertEqual(inference['custom_corpus']['source']['sha256'], fingerprint(source))
                self.assertEqual(Generator(out/'best.pt', 'cpu').chat_format, CHAT_FORMAT if chat else None)
                config = root/'config.json'
                config.write_text(json.dumps(cfg))
                arguments = ['train', '--resume', '--device', 'cpu', '--precision', 'fp32', '--data', str(data), '--out', str(out), '--config', str(config)]
                with patch.object(sys, 'argv', arguments), patch('scratchllm.train.signal.signal'), \
                        patch.object(GPT, 'forward', side_effect=AssertionError('No model fitting')), \
                        patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No model fitting')) as step:
                    training_main()
                    changed = copy.deepcopy(meta)
                    changed['custom_report']['source']['sha256'] = 'changed'
                    (data/'meta.json').write_text(json.dumps(changed))
                    with self.assertRaisesRegex(SystemExit, 'mismatch'):
                        training_main()
                    step.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'changed'):
                    initialize_from(initialized, checkpoint, dict(meta, base_checkpoint_sha256='changed'), tokenizer.to_str())


if __name__ == '__main__':
    unittest.main()
