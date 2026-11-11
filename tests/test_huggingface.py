"""GGUF round trips, scraper preparation and adapter lifecycle; no optimizer steps."""
import copy
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from gguf_fixture import checkpoint
from test_serving_selection import FakeGenerator
from scratchllm.corpus import corpus_plan, records
from scratchllm.gguf_export import export_gguf
from scratchllm.hf_requests import HFImportRequest, HFPrepareRequest, HFTrainRequest, HFMergeRequest, HFRepositoryRequest
from scratchllm.huggingface import (Stopped, assets, convert_gguf, field, import_model, inspect_gguf,
                                  load_import, merge_shards, prepare_corpus, repository_files, sha256)
from scratchllm.hf_train import (adapter_archive, batch_loss, build_model, checkpoint_path, evaluate,
                                load_data, merge_model, run_training, save_snapshot, training_preview, windows)
from scratchllm.jobs import Jobs
from scratchllm.serve import create_app
from scratchllm.text_policy import EXTRACTION_VERSION


def fixture(root, architecture='modern', precision='f32'):
    path, original, tokenizer = checkpoint(root, architecture)
    file = root/'model.gguf'
    export_gguf(path, file, 'f32')
    model = root/'models/base'
    imported = convert_gguf(file, model, dict(repo='fixture/test', revision='a' * 40, files=[]), precision)
    source = root/'data/raw.jsonl'
    source.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(text=f'Astronomy record {i}: the planets have water, clouds, dust and rings. https://example.org/{i}',
                 url=f'https://soyjak.st/thread/{i//2}', host='soyjak.st', extraction=EXTRACTION_VERSION,
                 kind='post', group=f'thread-{i//2}') for i in range(8)]
    source.write_text('\n'.join(json.dumps(row) for row in [*rows, rows[0]]) + '\n')
    return model, imported, source, original, tokenizer


def prepared(root):
    model, imported, source, original, tokenizer = fixture(root)
    data = root/'data/prepared'
    meta = prepare_corpus(HFPrepareRequest(model=str(model), input=str(source), output=str(data),
                                          context=32, min_chars=1, min_post_chars=1))
    request = HFTrainRequest(model=str(model), data=str(data), out=str(root/'runs/adapter'),
                             mode='lora', device='cpu', batch_size=2, accumulation=1, rank=2, alpha=4)
    return request, imported, meta, original, tokenizer


def legacy_splits(source, output):
    """Existing pre-URL-policy prepared text, with extraction provenance in metadata."""
    output.mkdir(parents=True)
    validation, report = corpus_plan(source, validation_fraction=0.25, min_chars=1, min_post_chars=1)
    rows = {'train': [], 'val': []}
    for row in records(source, min_chars=1, min_post_chars=1):
        split = 'val' if row['group'] in validation else 'train'
        row.pop('extraction')
        row.pop('text_policy')
        row['text'] += ' https://old.example/path >>123456'
        rows[split].append(row)
    for split in rows:
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows[split]))
    report.pop('text_policy')
    (output/'meta.json').write_text(json.dumps(dict(documents=report['documents'], corpus_report=report)))
    (output/'report.json').write_text(json.dumps(report))
    return rows


def rewrite_gguf(source, output, part=None, quantize=False, overrides=None):
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(output, field(reader, 'general.architecture'))
    try:
        for name, item in reader.fields.items():
            if name.startswith(('GGUF.', 'split.')) or name == 'general.architecture':
                continue
            writer.add_key_value(name, (overrides or {}).get(name, item.contents()), item.types[0], item.types[-1] if len(item.types) > 1 else None)
        if part is not None:
            writer.add_uint16('split.no', part)
            writer.add_uint16('split.count', 2)
            writer.add_int32('split.tensors.count', len(reader.tensors))
        for index, tensor in enumerate(reader.tensors):
            if part is not None and index % 2 != part:
                continue
            data, dtype = tensor.data.copy(), tensor.tensor_type
            if quantize and data.ndim == 2:
                dtype = gguf.GGMLQuantizationType.Q8_0
                data = gguf.quantize(data, dtype)
            writer.add_tensor(tensor.name, data, raw_dtype=dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.no_updates = patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Training is not authorized'))
        self.no_updates.start()
        self.addCleanup(self.no_updates.stop)

    def test_conversion_preserves_tokens_chat_and_logits(self):
        for architecture in ('modern', 'gpt'):
            for precision in ('f32', 'f16'):
                with self.subTest(architecture=architecture, precision=precision), tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
                    root = Path(directory)
                    model_path, imported, _, original, tokenizer = fixture(root, architecture, precision)
                    actual_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).eval()
                    self.assertEqual(len(actual_tokenizer), tokenizer.get_vocab_size())
                    for text in ('Hello there!', 'Räksmörgås 日本語 שלום', '  spaces\t\r\n'):
                        ids = tokenizer.encode(text).ids
                        self.assertEqual(actual_tokenizer.encode(text, add_special_tokens=False), ids)
                        inputs = torch.tensor([[1, *ids]])
                        with torch.inference_mode():
                            torch.testing.assert_close(model(inputs).logits[:, -1:], original(inputs)[0], atol=0.0005, rtol=0.001)
                    self.assertIn('Assistant:', actual_tokenizer.apply_chat_template([dict(role='user', content='Hello')], tokenize=False, add_generation_prompt=True))
                    self.assertEqual(load_import(model_path, verify=True), imported)
                    self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))
                    with self.assertRaisesRegex(ValueError, 'fresh'):
                        convert_gguf(root/'model.gguf', model_path, {}, precision)
                    (model_path/'tokenizer.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        load_import(model_path, verify=True)

    def test_packed_shards_are_joined_and_dequantized(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            path, original, tokenizer = checkpoint(root)
            source = root/'source.gguf'
            export_gguf(path, source, 'f32')
            parts = [root/f'model-{index + 1:05d}-of-00002.gguf' for index in range(2)]
            for index, part in enumerate(parts):
                rewrite_gguf(source, part, index, quantize=True)
            joined = root/'joined.gguf'
            merge_shards(parts, joined)
            info = inspect_gguf(joined)
            self.assertIn('Q8_0', info['quantizations'])
            self.assertNotIn('split.count', gguf.GGUFReader(joined).fields)
            for part in parts:
                for tensor in gguf.GGUFReader(part).tensors:
                    actual = next(row for row in gguf.GGUFReader(joined).tensors if row.name == tensor.name)
                    np.testing.assert_array_equal(actual.data, tensor.data)
            convert_gguf(joined, root/'imported', {}, 'f32')
            actual = AutoModelForCausalLM.from_pretrained(root/'imported', local_files_only=True).eval()
            ids = torch.tensor([[1, *tokenizer.encode('Hello there!').ids]])
            with torch.inference_mode():
                torch.testing.assert_close(actual(ids).logits[:, -1:], original(ids)[0], atol=0.005, rtol=0.02)
            with self.assertRaisesRegex(ValueError, 'numbering'):
                merge_shards(parts[::-1], root/'wrong.gguf')

    def test_missing_padding_sentinel_and_bos_policy_are_normalized(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            path, _, tokenizer = checkpoint(root)
            export_gguf(path, root/'source.gguf', 'f32')
            rewrite_gguf(root/'source.gguf', root/'sentinel.gguf', overrides={
                'tokenizer.ggml.padding_token_id': 2**32 - 1, 'tokenizer.ggml.add_bos_token': False})
            convert_gguf(root/'sentinel.gguf', root/'imported', {}, 'f32')
            actual = AutoTokenizer.from_pretrained(root/'imported', local_files_only=True)
            self.assertEqual(actual.pad_token_id, actual.eos_token_id)
            self.assertEqual(len(actual), tokenizer.get_vocab_size())
            self.assertEqual(actual.encode('Hello!'), tokenizer.encode('Hello!').ids)
            model = AutoModelForCausalLM.from_pretrained(root/'imported', local_files_only=True)
            self.assertEqual(model.config.pad_token_id, actual.eos_token_id)

    def test_hub_revision_checksum_and_interrupted_download_identity(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            path, _, _ = checkpoint(root)
            file = root/'model.gguf'
            export_gguf(path, file, 'f32')
            part = dict(filename='model.gguf', bytes=file.stat().st_size, sha256=sha256(file))
            catalog = dict(repo='fixture/repo', revision='a' * 40, license='mit', files=[dict(filename='model.gguf', bytes=part['bytes'], parts=[part])])
            request = HFImportRequest(repo='fixture/repo', filename='model.gguf', output=str(root/'models/imported'))
            with patch('scratchllm.huggingface.repository_files', return_value=catalog) as lookup, \
                    patch('huggingface_hub.try_to_load_from_cache', return_value=str(file)), \
                    patch('huggingface_hub.hf_hub_download', side_effect=Stopped()):
                with self.assertRaises(Stopped):
                    import_model(request, root/'cache')
                self.assertEqual(lookup.call_count, 1)
                self.assertFalse(Path(request.output).exists())
            with patch('scratchllm.huggingface.repository_files', side_effect=AssertionError('Must reuse pinned revision')), \
                    patch('huggingface_hub.try_to_load_from_cache', return_value=str(file)), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(file)) as download:
                manifest = import_model(request, root/'cache')
                self.assertEqual(download.call_args.kwargs['revision'], 'a' * 40)
                self.assertEqual(manifest['source']['files'][0]['sha256'], part['sha256'])
            altered = request.model_copy(update={'output': str(root/'other')})
            broken = copy.deepcopy(catalog)
            broken['files'][0]['parts'][0]['sha256'] = '0' * 64
            with patch('scratchllm.huggingface.repository_files', return_value=broken), \
                    patch('huggingface_hub.try_to_load_from_cache', return_value=str(file)), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(file)):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    import_model(altered, root/'cache')
                self.assertFalse(Path(altered.output).exists())

    def test_repository_groups_and_request_validation(self):
        rows = [SimpleNamespace(rfilename=name, size=100, lfs={'sha256': 'a' * 64}) for name in (
            'model-00001-of-00002.gguf', 'model-00002-of-00002.gguf', 'solo.gguf', 'incomplete-00001-of-00002.gguf', 'README.md')]
        with patch('huggingface_hub.HfApi') as api:
            api.return_value.model_info.return_value = SimpleNamespace(sha='a' * 40, siblings=rows, card_data={'license': 'mit'}, gated=False)
            catalog = repository_files(HFRepositoryRequest(repo='https://huggingface.co/test/model/'))
        self.assertEqual(catalog['repo'], 'test/model')
        self.assertEqual(len(catalog['files']), 2)
        self.assertEqual(catalog['files'][0]['bytes'], 200)
        for filename in ('../model.gguf', '/model.gguf', 'other.txt', 'dir\\model.gguf'):
            with self.assertRaises(ValueError):
                HFImportRequest(repo='test/model', filename=filename)
        with self.assertRaises(ValueError):
            HFTrainRequest(device='cpu', mode='qlora')

    def test_scraper_group_splits_exact_tokenizer_and_changed_data_rejection(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            request, imported, meta, _, tokenizer = prepared(root)
            groups = {}
            for split in ('train', 'val'):
                rows = list(map(json.loads, (Path(request.data)/f'{split}.jsonl').read_text().splitlines()))
                groups[split] = {row['group'] for row in rows}
                expected = []
                for row in rows:
                    self.assertNotIn('http', row['text'])
                    self.assertIn('https://soyjak.st/', row['url'])
                    expected.extend([1, *tokenizer.encode(row['text']).ids, 2])
                self.assertEqual(np.fromfile(Path(request.data)/f'{split}.bin', dtype='<u4').tolist(), expected)
            self.assertFalse(groups['train'] & groups['val'])
            self.assertEqual(meta['corpus_report']['duplicates_removed'], 1)
            self.assertEqual(meta['artifact_id'], imported['artifact_id'])
            plan = training_preview(request)
            self.assertEqual(plan['steps'], math.ceil(3 * plan['windows'] / 2))
            self.assertTrue(plan['fits_estimate'])
            self.assertEqual(len(assets(root)['datasets']), 1)
            self.assertEqual(Jobs(root, 'cpu').prepared_datasets()['datasets'], [])
            with (Path(request.data)/'train.bin').open('r+b') as stream:
                stream.write(b'\x00\x00\x00\x00')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_data(request, verify=True)

    def test_existing_native_train_jsonl_is_listed_and_retokenized_without_its_old_bins(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            model, _, raw, _, tokenizer = fixture(root)
            original = legacy_splits(raw, root/'data/previous')
            source = root/'data/previous/train.jsonl'
            before = source.read_bytes()
            (source.parent/'train.bin').write_bytes(b'old incompatible tokenizer IDs must not be reused')
            listed = {row['path']: row for row in assets(root)['corpora']}
            self.assertEqual(listed['data/previous/train.jsonl']['kind'], 'prepared_split')
            self.assertEqual(listed['data/previous/val.jsonl']['split'], 'val')
            output = root/'data/retokenized'
            meta = prepare_corpus(HFPrepareRequest(model=str(model), input=str(source), output=str(output),
                                                   context=32, min_chars=1, min_post_chars=1))
            groups = set()
            for split in ('train', 'val'):
                rows = list(map(json.loads, (output/f'{split}.jsonl').read_text().splitlines()))
                tokens = []
                for row in rows:
                    self.assertNotIn('https://', row['text'])
                    self.assertNotIn('>>123456', row['text'])
                    self.assertEqual(row['extraction'], EXTRACTION_VERSION)
                    groups.add(row['group'])
                    tokens.extend([1, *tokenizer.encode(row['text']).ids, 2])
                self.assertEqual(np.fromfile(output/f'{split}.bin', dtype='<u4').tolist(), tokens)
            self.assertEqual(groups, {row['group'] for row in original['train']})
            self.assertFalse(groups & {row['group'] for row in original['val']})
            self.assertEqual(sum(meta['documents'].values()), len(original['train']))
            self.assertEqual(meta['source'], str(source))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual((source.parent/'train.bin').read_bytes(), b'old incompatible tokenizer IDs must not be reused')

    def test_plain_text_keeps_provenance_and_uses_imported_tokenizer_ids(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            model, _, _, _, tokenizer = fixture(root)
            source, output = root/'data/short.txt', root/'data/text-prepared'
            raw = 'First sentence.\nAnother short text.\nRäksmörgås i solen.\nFirst sentence.\n'.encode('utf-8')
            source.write_bytes(raw)
            meta = prepare_corpus(HFPrepareRequest(model=str(model), input=str(source), output=str(output),
                                                   context=32, min_chars=1, min_post_chars=1))
            groups, texts = {}, set()
            for split in ('train', 'val'):
                rows = list(map(json.loads, (output/f'{split}.jsonl').read_text().splitlines()))
                self.assertTrue(rows)
                groups[split] = {row['group'] for row in rows}
                ids = []
                for row in rows:
                    texts.add(row['text'])
                    self.assertEqual(row['source']['path'], str(source))
                    ids.extend([1, *tokenizer.encode(row['text']).ids, 2])
                self.assertEqual(np.fromfile(output/f'{split}.bin', dtype='<u4').tolist(), ids)
            self.assertEqual(texts, {'First sentence.', 'Another short text.', 'Räksmörgås i solen.'})
            self.assertFalse(groups['train'] & groups['val'])
            self.assertEqual(meta['corpus_report']['duplicates_removed'], 1)
            self.assertEqual(meta['corpus_report']['source']['sha256'], sha256(source))
            self.assertEqual(source.read_bytes(), raw)

    def test_real_lora_gradients_shift_and_frozen_base_without_updates(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            request, imported, meta, _, _ = prepared(Path(directory))
            model = build_model(request, imported)
            frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
            stream = np.fromfile(Path(request.data)/'val.bin', dtype='<u4')
            rows = windows(len(stream), meta['context'])
            model.train()
            loss = batch_loss(model, stream, rows[:2], 'cpu')
            loss.backward()  # Gradient-connectivity diagnostic only: no optimizer/update.
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.named_parameters() if 'lora_B' in name))
            for name, parameter in model.named_parameters():
                if name in frozen:
                    self.assertIsNone(parameter.grad)
                    torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
            model.eval()
            inputs = torch.tensor(stream[:8].astype(np.int64))[None]
            with torch.inference_mode():
                expected = model(inputs, labels=inputs, use_cache=False).loss
                actual = batch_loss(model, stream, [(0, 7)], 'cpu')
            torch.testing.assert_close(expected, actual)
            self.assertTrue(math.isfinite(evaluate(model, stream, rows, 2, 'cpu')))

    def test_stopped_zero_update_snapshot_resume_merge_and_download(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            request, imported, _, _, tokenizer = prepared(root)
            stopped = Event()
            stopped.set()
            run_training(request, cancel=stopped)
            run = Path(request.out)
            first = checkpoint_path(run)
            state = torch.load(first/'state.pt', weights_only=True)
            self.assertEqual(state['step'], 0)
            self.assertEqual(state['stop_reason'], 'user_stop')
            run_training(HFTrainRequest(out=str(run), resume=True), cancel=stopped)
            latest = torch.load(checkpoint_path(run)/'state.pt', weights_only=True)
            self.assertEqual(latest['settings'], state['settings'])
            self.assertEqual(latest['sampler'], state['sampler'])
            for key in state['adapter']:
                torch.testing.assert_close(state['adapter'][key], latest['adapter'][key], atol=0, rtol=0)
            merged = root/'models/merged'
            merge_model(HFMergeRequest(run=str(run), output=str(merged), precision='f32'))
            base = AutoModelForCausalLM.from_pretrained(request.model, local_files_only=True)
            actual = AutoModelForCausalLM.from_pretrained(merged, local_files_only=True)
            for key, value in base.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key], value, atol=0, rtol=0)
            archive = root/'adapter.zip'
            adapter_archive(run, archive)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('adapter/adapter_model.safetensors', z.namelist())
                self.assertIn('tokenizer/tokenizer.json', z.namelist())
            # A changed dataset must be rejected before constructing the model on resume.
            meta = Path(request.data)/'hf-meta.json'
            meta.write_text(meta.read_text() + ' ')
            with patch('scratchllm.hf_train.build_model', side_effect=AssertionError('Changed inputs reached model loading')):
                with self.assertRaisesRegex(ValueError, 'Resume requires'):
                    run_training(HFTrainRequest(out=str(run), resume=True), cancel=stopped)

    def test_preparation_supervision_and_api_training_launch_are_separate(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            model, _, source, _, _ = fixture(root)
            jobs = Jobs(root, 'cpu')
            try:
                result = jobs.start('hf-prepare', HFPrepareRequest(model='models/base', input='data/raw.jsonl',
                    output='data/worker', min_chars=1, min_post_chars=1, context=32))
                self.assertEqual(result['job']['kind'], 'hf-prepare')
                self.assertTrue(jobs.finished.wait(30))
                finished = jobs.status()['job']
                self.assertEqual(finished['status'], 'succeeded', finished['log'])
                self.assertTrue(finished['hf_ready'])
                with patch('scratchllm.jobs.subprocess.Popen', side_effect=RuntimeError('intercept user training launch')) as spawn:
                    with self.assertRaisesRegex(RuntimeError, 'intercept user'):
                        jobs.start('hf-finetune', HFTrainRequest(model='models/base', data='data/worker', out='runs/new', device='cpu', mode='lora'))
                    args = spawn.call_args.args[0]
                    self.assertIn('scratchllm.hf_train', args)
                    self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            finally:
                jobs.close()
            with patch('scratchllm.serve.Generator', FakeGenerator), TestClient(create_app(root/'modern-best.pt', 'cpu', workspace=root)) as client:
                self.assertTrue(client.get('/health').json()['features']['huggingface'])
                self.assertEqual(client.get('/huggingface/assets').json()['models'][0]['path'], 'models/base')
                response = client.post('/huggingface/plan', json=dict(model='models/base', data='data/worker', device='cpu', mode='lora'))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(client.post('/jobs/hf-import', json=dict(repo='test/repo', filename='model.gguf', output='../outside')).status_code, 400)
                self.assertEqual(client.post('/jobs/hf-import', json=dict(repo='test/repo', filename='../model.gguf')).status_code, 422)


if __name__ == '__main__':
    unittest.main()
