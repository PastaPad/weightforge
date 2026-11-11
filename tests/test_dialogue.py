"""Data/initialization/evaluation/chat invariants; no optimizer steps or model fitting."""
import copy
import gzip
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from scratchllm.dialogue import (CHAT_FORMAT, SOURCES, download, encode_example, examples, fingerprint,
                                prepare, prompt_tokens)
from scratchllm.dialogue_training import ShuffledExamples, initialize_from, load_dialogue
from scratchllm.generate import Generator
from scratchllm.jobs import DialogueRequest, FineTuneRequest, Jobs, PipelineRequest, TrainRequest
from scratchllm.model import GPT, ModelConfig
from scratchllm.serve import ChatRequest
from scratchllm.train import save_checkpoints, main as training_main
from scratchllm.training import EarlyStopping, evaluate, evaluation_plan, make_batch, training_plan


def fixture_base(root, context=256):
    # Explicit byte vocabulary, not a fitted tokenizer or a pretrained model.
    vocab = {value: index for index, value in enumerate(['<|pad|>', '<|bos|>', '<|eos|>', *sorted(pre_tokenizers.ByteLevel.alphabet())])}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(['<|pad|>', '<|bos|>', '<|eos|>'])
    model = GPT(ModelConfig(vocab_size=len(vocab), context=context, width=16, layers=1, heads=2, dropout=0))
    checkpoint = root/'base.pt'
    torch.save(dict(model=model.state_dict(), model_config=model.config_dict(), tokenizer_json=tokenizer.to_str()), checkpoint)
    return checkpoint, tokenizer, model


def fixture_dolly(root):
    path = root/'dolly.jsonl'
    rows = [dict(instruction=f'Explain subject {i}.', context=f'Reference passage {i // 2}.',
                 response=f'This is human answer {i}. Läs mer. https://example.org/{i}') for i in range(20)]
    rows += rows[:1]
    rows.append(dict(instruction='Long?', context='', response='Very long answer. ' * 100))
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


class DialogueTests(unittest.TestCase):
    def test_custom_conversations_expand_replies_preserve_groups_and_mask_prompts(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, tokenizer, _ = fixture_base(root, context=512)
            source = root/'conversations.jsonl'
            original = [dict(group=f'family-{i}', synthetic=True, messages=[
                dict(role='user', content=f'Hva e meet ze fono? {i}'),
                dict(role='assistant', content=f'Zea keel e meet ze fono. {i}'),
                dict(role='user', content='Meetnu.'),
                dict(role='assistant', content=f'Zea keel e meet ze fono. {i}')]) for i in range(6)]
            source.write_text(''.join(json.dumps(row) + '\n' for row in original))
            before = source.read_bytes()
            output = root/'prepared'
            with patch('scratchllm.dialogue.download', side_effect=AssertionError('Custom-only must stay offline')):
                meta = prepare(checkpoint, output, sources=[], input=source)
            self.assertEqual(sum(meta['documents'].values()), 12)
            groups = {}
            for split in ('train', 'val'):
                rows = list(map(json.loads, (output/f'{split}.jsonl').read_text().splitlines()))
                groups[split] = {row['group'] for row in rows}
                ids = np.fromfile(output/f'{split}.bin', dtype='<u4')
                masks = np.fromfile(output/f'{split}.mask.bin', dtype='u1')
                windows = json.loads((output/f'{split}.windows.json').read_text())
                for row, (start, length) in zip(rows, windows):
                    answer = tokenizer.encode(row['messages'][-1]['content']).ids + [2]
                    chunk = ids[start:start+length+1]
                    mask = masks[start:start+length+1]
                    self.assertEqual(chunk[mask.astype(bool)].tolist(), answer)
                    self.assertTrue(row['provenance']['synthetic'])
                    self.assertEqual(mask.tolist(), [0]*(len(chunk)-len(answer)) + [1]*len(answer))
            self.assertFalse(groups['train'] & groups['val'])
            self.assertEqual(source.read_bytes(), before)

    def setUp(self):
        torch.set_num_threads(2)

    def prepared(self, root):
        checkpoint, tokenizer, model = fixture_base(root)
        raw = fixture_dolly(root)
        with patch('scratchllm.dialogue.download', return_value=raw):
            meta = prepare(checkpoint, root/'data/dialogue', ['dolly15k'], validation_fraction=0.25)
        return checkpoint, tokenizer, model, meta

    def test_human_tree_selection_keeps_best_human_branch_and_rejects_synthetic_ancestors(self):
        def node(key, text, role, **kwargs):
            return dict(message_id=key, text=text, role=role, lang='en', synthetic=False, deleted=False,
                        review_result=True, rank=0, replies=[], **kwargs)
        root = node('q', 'A question?', 'prompter')
        answer = node('a', 'An answer.', 'assistant')
        followup = node('q2', 'And then?', 'prompter')
        followup['replies'] = [node('a2', 'Then continue.', 'assistant')]
        answer['replies'] = [followup]
        synthetic = node('bot', 'Synthetic response', 'assistant')
        synthetic.update(synthetic=True, replies=[followup])
        low = node('low', 'Lower ranked.', 'assistant')
        low['rank'] = 1
        root['replies'] = [synthetic, low, answer]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trees.jsonl.gz'
            row = dict(message_tree_id='tree', tree_state='ready_for_export', prompt=root)
            with gzip.open(path, 'wt') as stream:
                stream.write(json.dumps(row) + '\n')
            rows = list(examples('oasst1', path))
            self.assertEqual([row['id'] for row in rows], ['a', 'a2'])
            self.assertEqual([len(row['messages']) for row in rows], [2, 4])
            self.assertEqual(len({row['group'] for row in rows}), 1)
            root['synthetic'] = True
            with gzip.open(path, 'wt') as stream:
                stream.write(json.dumps(row) + '\n')
            self.assertEqual(list(examples('oasst1', path)), [])

    def test_frozen_vocabulary_group_split_masks_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, tokenizer, _, meta = self.prepared(root)
            data = root/'data/dialogue'
            self.assertEqual((data/'tokenizer.json').read_text(), tokenizer.to_str())
            self.assertEqual(meta['tokenizer_sha256'], hashlib.sha256(tokenizer.to_str().encode()).hexdigest())
            self.assertEqual(meta['base_checkpoint_sha256'], fingerprint(checkpoint))
            self.assertEqual(meta['dialogue_report']['duplicates_removed'], 1)
            self.assertEqual(meta['dialogue_report']['over_context_removed'], 1)
            streams = {split: np.memmap(data/f'{split}.bin', dtype='<u4', mode='r') for split in ('train', 'val')}
            masks, windows = load_dialogue(data, meta, streams, 256)
            groups = {}
            for split in streams:
                rows = [json.loads(line) for line in (data/f'{split}.jsonl').read_text().splitlines()]
                groups[split] = {row['group'] for row in rows}
                for row, (start, size) in zip(rows, windows[split]):
                    x, y = make_batch(streams[split], [(start, size)], 'cpu', masks[split])
                    prompt = prompt_tokens(tokenizer, row['messages'][:-1])
                    self.assertTrue(torch.all(y[0, :len(prompt) - 1] == -100))
                    answer = tokenizer.decode(y[y != -100].tolist())
                    self.assertIn('human answer', answer)
                    self.assertNotIn('https://', answer)
                    self.assertEqual(y[0, -1].item(), tokenizer.token_to_id('<|eos|>'))
                    self.assertEqual(x[0, :len(prompt)].tolist(), prompt)
            self.assertFalse(groups['train'] & groups['val'])
            self.assertTrue((data/'SOURCES.md').is_file())
            catalog = Jobs(root, 'cpu').prepared_datasets()
            self.assertIsNone(catalog['latest'])
            self.assertEqual(catalog['dialogue_datasets'][0]['path'], 'data/dialogue')

    def test_initialize_copies_weights_and_rejects_tokenizer_or_checkpoint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, tokenizer, original, meta = self.prepared(root)
            candidate = GPT(original.cfg)
            initialize_from(candidate, checkpoint, meta, tokenizer.to_str())
            for key, value in original.state_dict().items():
                torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
            self.assertIs(candidate.token.weight, candidate.head.weight)
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                initialize_from(candidate, checkpoint, dict(meta, tokenizer_sha256='wrong'), tokenizer.to_str())
            with self.assertRaisesRegex(ValueError, 'changed'):
                initialize_from(candidate, checkpoint, dict(meta, base_checkpoint_sha256='wrong'), tokenizer.to_str())
            state = dict(model=candidate.state_dict(), model_config=candidate.config_dict(), tokenizer_json=tokenizer.to_str(),
                         step=1, best_val=1.0, data_hashes={}, evaluation={}, chat_format=CHAT_FORMAT,
                         base_checkpoint=str(checkpoint), base_checkpoint_sha256=meta['base_checkpoint_sha256'], dialogue_sources={})
            output = root/'saved'
            output.mkdir()
            save_checkpoints(output, state, candidate, None, 'raw', True)
            generator = Generator(output/'best.pt', 'cpu')
            self.assertEqual(generator.chat_format, CHAT_FORMAT)
            self.assertEqual(generator.tokenizer.to_str(), tokenizer.to_str())

    def test_deduplication_after_history_shortening_prevents_identical_split_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = fixture_base(root, context=128)
            rows = [dict(source='dolly15k', id=str(index), group=str(index), messages=[
                dict(role='user', content=('Old history ' + str(index) + ' ') * 20),
                dict(role='assistant', content='Old answer. ' * 20),
                dict(role='user', content='Same short question?'), dict(role='assistant', content='Same final reply.')]) for index in range(2)]
            rows.append(dict(source='dolly15k', id='third', group='third', messages=[
                dict(role='user', content='A different question?'), dict(role='assistant', content='A different reply.')]))
            with patch('scratchllm.dialogue.download', return_value=root/'unused'), patch('scratchllm.dialogue.examples', return_value=iter(rows)):
                meta = prepare(checkpoint, root/'prepared', ['dolly15k'])
            self.assertEqual(meta['dialogue_report']['duplicates_removed'], 1)
            self.assertEqual(sum(meta['documents'].values()), 2)

    def test_masked_evaluation_is_assistant_weighted_and_sampler_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, model, meta = self.prepared(root)
            data = root/'data/dialogue'
            streams = {split: np.memmap(data/f'{split}.bin', dtype='<u4', mode='r') for split in ('train', 'val')}
            masks, windows = load_dialogue(data, meta, streams, 256)
            sampler = ShuffledExamples(windows['train'], 42)
            self.assertEqual(sorted(sampler.take(len(windows['train']))), sorted(windows['train']))
            sampler.take(3)
            state = copy.deepcopy(sampler.state_dict())
            restored = ShuffledExamples(windows['train'], 8)
            restored.load_state_dict(state)
            self.assertEqual(sampler.take(51), restored.take(51))
            cfg = FineTuneRequest().model_dump()
            cfg.update({key: value for key, value in meta['base_model_config'].items() if key != 'vocab_size'})
            plan = evaluation_plan(streams, cfg, windows, masks)
            self.assertEqual(plan['protocol'], 'fixed-assistant-nll-v1')
            self.assertEqual(plan['tokens']['val'], meta['supervised_tokens']['val'])
            model.eval()
            expected = []
            with torch.inference_mode():
                for window in windows['val']:
                    x, y = make_batch(streams['val'], [window], 'cpu', masks['val'])
                    logits, _ = model(x, y)
                    expected.extend((-logits.log_softmax(-1)[y != -100].gather(-1, y[y != -100].unsqueeze(-1))).flatten().tolist())
            for batch_size in (1, 3):
                value = evaluate(model, streams['val'], windows['val'], batch_size, 'cpu', mask=masks['val'])
                self.assertAlmostEqual(value, sum(expected)/len(expected), places=5)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertEqual(training_plan(cfg, meta)['steps'], math.ceil(3 * meta['documents']['train'] / 16))
            self.assertLessEqual(training_plan(cfg, meta)['warmup_steps'], training_plan(cfg, meta)['steps'] // 10)

    def test_chat_template_matches_training_and_drops_only_complete_old_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, tokenizer, _ = fixture_base(Path(directory), context=120)
            generator = Generator(checkpoint, 'cpu')
            with self.assertRaisesRegex(ValueError, 'dialogue'):
                generator.chat('Question?')
            generator.chat_format = CHAT_FORMAT
            messages = [dict(role='user', content='Question?'), dict(role='assistant', content='Answer.')]
            ids, mask = encode_example(tokenizer, messages, 120)
            with patch.object(generator, '_generate', return_value={'completion': 'Answer.', 'generated_tokens': 7, 'prompt_tokens_used': 12}) as generate:
                result = generator.chat('Question?')
                self.assertEqual(generate.call_args.args[0], ids[:mask.index(1)])
                self.assertEqual(result['completion'], 'Answer.')
                result = generator.chat('Question?', history=[dict(role='user', content='Old ' * 20), dict(role='assistant', content='Old answer. ' * 10)])
                self.assertEqual(result['history_messages_used'], 0)
                with self.assertRaisesRegex(ValueError, 'shorten'):
                    generator.chat('Long question ' * 30)
            self.assertIsNone(encode_example(tokenizer, [messages[0], dict(role='assistant', content='Too long ' * 30)], 120))
            with self.assertRaises(ValueError):
                ChatRequest(prompt='Question', history=[dict(role='assistant', content='Bad order')])

    def test_preview_initialization_and_resume_launch_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, meta = self.prepared(root)
            jobs = Jobs(root, 'cpu')
            request = FineTuneRequest(checkpoint='base.pt', data='data/dialogue', out='runs/dialogue', width=512)
            preview = jobs.preview_training(request)
            self.assertEqual(preview['model_config']['width'], 16)
            with patch('scratchllm.jobs.subprocess.Popen', side_effect=RuntimeError('intercept worker')) as spawn:
                with self.assertRaisesRegex(ValueError, 'dialogue'):
                    jobs.start('train', TrainRequest(data=request.data))
                with self.assertRaisesRegex(RuntimeError, 'intercept worker'):
                    jobs.start('finetune', request)
                command = spawn.call_args.args[0]
                self.assertIn('--init-from', command)
                config = json.loads((root/'runs/dialogue/web-config.json').read_text())
                self.assertEqual(config['width'], 16)
                self.assertEqual(config['steps'], preview['steps'])
                (root/'runs/dialogue/last.pt').write_text('fixture resume marker')
                with self.assertRaisesRegex(RuntimeError, 'intercept worker'):
                    jobs.start('finetune', FineTuneRequest(out='runs/dialogue', resume=True, checkpoint='missing.pt', data='missing'))
                command = spawn.call_args.args[0]
                self.assertIn('--resume', command)
                self.assertNotIn('--init-from', command)
                self.assertIn(str(root/'data/dialogue'), command)
                self.assertEqual(json.loads((root/'runs/dialogue/web-config.json').read_text()), config)

    def test_cached_export_checksum_and_fresh_output_protect_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/SOURCES['dolly15k']['file']
            path.write_text('user data')
            with patch('scratchllm.dialogue.urlopen') as fetch:
                with self.assertRaisesRegex(ValueError, 'differs'):
                    download('dolly15k', root)
            fetch.assert_not_called()
            self.assertEqual(path.read_text(), 'user data')
            with self.assertRaisesRegex(ValueError, 'empty'):
                prepare('missing.pt', root)
            self.assertEqual(path.read_text(), 'user data')

    def test_completed_resume_restores_without_steps_and_detects_changed_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, tokenizer, model, meta = self.prepared(root)
            data, out = root/'data/dialogue', root/'resume'
            out.mkdir()
            cfg = FineTuneRequest().model_dump(exclude={'data', 'out', 'resume', 'checkpoint'})
            cfg.update({key: value for key, value in meta['base_model_config'].items() if key != 'vocab_size'})
            config = root/'config.json'
            config.write_text(json.dumps(cfg))
            streams = {split: np.memmap(data/f'{split}.bin', dtype='<u4', mode='r') for split in ('train', 'val')}
            masks, windows = load_dialogue(data, meta, streams, cfg['context'])
            plan = training_plan(cfg, meta)
            evaluation = evaluation_plan(streams, cfg, windows, masks)
            hashes = {split: fingerprint(data/f'{split}.bin') for split in streams}
            hashes.update({f'{split}.{suffix}': fingerprint(data/f'{split}.{suffix}')
                           for split in streams for suffix in ('mask.bin', 'windows.json')})
            hashes['meta'] = fingerprint(data/'meta.json')
            decay, no_decay = [], []
            for parameter in model.parameters():
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
            optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']}, {'params': no_decay, 'weight_decay': 0.0}],
                                          lr=cfg['learning_rate'], betas=(0.9, 0.95))
            state = dict(training_config=cfg, model_config=model.config_dict(), model=model.state_dict(),
                         tokenizer_json=tokenizer.to_str(), data_hashes=hashes, optimizer=optimizer.state_dict(),
                         scaler=torch.amp.GradScaler('cuda', enabled=False).state_dict(), precision='fp32',
                         step=plan['steps'], stop_reason='pass_budget', early_stopping=EarlyStopping().state_dict(),
                         rng=np.random.default_rng(1).bit_generator.state, eval_rng=np.random.default_rng(2).bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=None, chat_format=CHAT_FORMAT,
                         plan=plan, evaluation=evaluation, sampler=ShuffledExamples(windows['train'], cfg['seed']).state_dict())
            torch.save(state, out/'last.pt')
            arguments = ['train', '--resume', '--device', 'cpu', '--precision', 'fp32', '--data', str(data), '--out', str(out), '--config', str(config)]
            with patch.object(sys, 'argv', arguments), patch('scratchllm.train.signal.signal'), \
                    patch.object(GPT, 'forward', side_effect=AssertionError('Training not permitted')), \
                    patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Training not permitted')) as step:
                training_main()
                step.assert_not_called()
                # Same dimensions, but different labels: the stream fingerprint must stop resume.
                changed = bytearray((data/'train.mask.bin').read_bytes())
                ones = [index for index, value in enumerate(changed) if value]
                zeros = [index for index, value in enumerate(changed) if not value]
                changed[ones[1]], changed[zeros[2]] = 0, 1
                (data/'train.mask.bin').write_bytes(changed)
                with self.assertRaisesRegex(SystemExit, 'mismatch'):
                    training_main()
                step.assert_not_called()
    def test_optional_pipeline_links_five_stages_and_rejects_path_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            request = PipelineRequest(dialogue=DialogueRequest())
            resolved = jobs._pipeline_request(request, 'example')
            stages = resolved.stages()
            self.assertEqual([kind for kind, _ in stages], ['crawl', 'prepare', 'train', 'dialogue', 'finetune'])
            self.assertEqual(stages[3][1].checkpoint, str(Path(resolved.train.out)/'best.pt'))
            self.assertEqual(stages[4][1].data, resolved.dialogue.output)
            self.assertNotEqual(resolved.train.out, resolved.finetune.out)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                jobs._pipeline_request(PipelineRequest(auto_paths=False, dialogue=DialogueRequest(output='runs/response')), 'example')


if __name__ == '__main__':
    unittest.main()
