"""Architecture, cache and gradient equivalence; no optimizer steps or model fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scratchllm.dialogue import CHAT_FORMAT, prepare
from scratchllm.dialogue_training import initialize_from
from scratchllm.generate import Generator
from scratchllm.jobs import FineTuneRequest, Jobs, TrainRequest
from scratchllm.model import GPT, ModelConfig, inherit_architecture, model_config, parameter_count
from scratchllm.presets import hardware_presets
from scratchllm.training import training_plan
from test_dialogue import fixture_base, fixture_dolly


class ModernModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def model(self, **kwargs):
        return GPT(ModelConfig(vocab_size=32, context=16, width=32, layers=2, heads=4, kv_heads=2,
                               ffn_width=64, architecture='modern', **kwargs))

    def test_exact_counts_and_legacy_checkpoint_schema(self):
        for architecture in ('gpt', 'modern'):
            cfg = ModelConfig(vocab_size=32, context=16, width=32, layers=2, heads=4, architecture=architecture)
            model = GPT(cfg)
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count(cfg))
            self.assertIs(model.token.weight, model.head.weight)
            if architecture == 'gpt':
                self.assertEqual(set(model.config_dict()), {'vocab_size', 'context', 'width', 'layers', 'heads', 'dropout'})
                self.assertIn('position.weight', model.state_dict())
                GPT(ModelConfig(**model.config_dict())).load_state_dict(model.state_dict(), strict=True)
        preset = hardware_presets()['presets'][0]
        cfg = model_config(8192, preset['settings'])
        with torch.device('meta'):
            model = GPT(cfg)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 41955840)
        self.assertEqual(preset['parameters'], 41955840)
        self.assertLess(preset['fp32_weights_gradients_adam_mib'], 650)
        request = TrainRequest(**preset['settings'])
        self.assertTrue(request.gradient_checkpointing)
        self.assertFalse(request.optimizer_foreach)
        plan = training_plan(request.model_dump(), dict(vocab_size=8192, tokens=dict(train=100001, val=10001), documents=dict(train=10, val=2)))
        self.assertEqual(plan['parameters'], preset['parameters'])
        self.assertEqual(plan['tokens_per_step'], 8192)

    def test_causality_cache_chunks_and_sliding_context_match_uncached_logits(self):
        model = self.model(dropout=0).eval()
        tokens = torch.randint(0, 32, (2, 16))
        with torch.inference_mode():
            full, _ = model(tokens, tokens)
            altered = tokens.clone()
            altered[:, 8:] = (altered[:, 8:] + 1) % 32
            after, _ = model(altered, altered)
            torch.testing.assert_close(full[:, :8], after[:, :8])
            cache, cursor = None, 0
            for length in (5, 1, 4, 1, 5):
                result, cache = model.forward_cached(tokens[:, cursor:cursor + length], cache)
                cursor += length
                torch.testing.assert_close(result[:, -1], full[:, cursor - 1], rtol=1e-4, atol=1e-6)
                self.assertEqual(cache[0][0].shape, (2, 2, cursor, 8))
            for prompt in (tokens[:, :5], tokens[:, :15]):
                cached = model.generate(prompt, 10, temperature=0, use_cache=True)
                uncached = model.generate(prompt, 10, temperature=0, use_cache=False)
                torch.testing.assert_close(cached, uncached, rtol=0, atol=0)

    def test_checkpointed_gradients_match_and_never_change_weights(self):
        model = self.model(dropout=0.2).train()
        candidate = copy.deepcopy(model)
        candidate.gradient_checkpointing = True
        x, y = torch.randint(0, 32, (1, 12)), torch.randint(0, 32, (1, 12))
        y[:, :4] = -100
        before = {key: value.detach().clone() for key, value in candidate.state_dict().items()}
        rng = torch.get_rng_state()
        loss = model(x, y)[1]
        expected = torch.autograd.grad(loss, tuple(model.parameters()))
        torch.set_rng_state(rng)
        checkpointed = candidate(x, y)[1]
        actual = torch.autograd.grad(checkpointed, tuple(candidate.parameters()))
        torch.testing.assert_close(loss, checkpointed)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)
        for key, value in candidate.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in candidate.parameters()))

    def test_invalid_dimensions_and_generation_statistics(self):
        for fields in (dict(kv_heads=0), dict(kv_heads=3), dict(ffn_width=0), dict(width=30, heads=2), dict(rope_theta=float('nan'))):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                ModelConfig(vocab_size=32, **(dict(architecture='modern', width=32, heads=4) | fields))
        model = self.model(dropout=0).eval()
        ids, stats = model.generate(torch.tensor([[1, 4, 7]]), max_new_tokens=3, temperature=0, collect_stats=True)
        self.assertEqual(ids.shape[-1], 6)
        self.assertEqual(stats['stop_reason'], 'token_limit')
        self.assertTrue(stats['cached_decoding'])
        self.assertFalse(stats['probability_is_factual_confidence'])
        self.assertGreater(stats['mean_token_entropy_nats'], 0)
        self.assertTrue(0 < stats['mean_selected_token_probability'] <= 1)

    def test_modern_dialogue_initialization_preview_and_serving_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, tokenizer, legacy = fixture_base(root)
            cfg = ModelConfig(vocab_size=259, context=256, width=32, heads=4, layers=1,
                              architecture='modern', kv_heads=2, ffn_width=64, dropout=0)
            model = GPT(cfg)
            torch.save(dict(model=model.state_dict(), model_config=model.config_dict(), tokenizer_json=tokenizer.to_str()), checkpoint)
            raw = fixture_dolly(root)
            with patch('scratchllm.dialogue.download', return_value=raw):
                meta = prepare(checkpoint, root/'data/dialogue', ['dolly15k'])
            request = FineTuneRequest(checkpoint='base.pt', data='data/dialogue', gradient_checkpointing=True)
            jobs = Jobs(root, 'cpu')
            inherited = jobs.finetune_config(request, meta)
            self.assertEqual(inherited['architecture'], 'modern')
            self.assertEqual(inherited['kv_heads'], 2)
            self.assertEqual(model_config(meta['vocab_size'], inherited).to_dict(), model.config_dict())
            self.assertEqual(jobs.preview_training(request)['parameters'], parameter_count(cfg))
            candidate = GPT(model_config(meta['vocab_size'], inherited))
            initialize_from(candidate, checkpoint, meta, tokenizer.to_str())
            state = torch.load(checkpoint, weights_only=True)
            state['chat_format'] = CHAT_FORMAT
            torch.save(state, root/'chat.pt')
            generator = Generator(root/'chat.pt', 'cpu')
            result = generator.chat('Hi', max_new_tokens=2, temperature=0, collect_stats=True)
            self.assertEqual(result['generated_tokens'], 2)
            self.assertTrue(result['diagnostics']['cached_decoding'])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
            restored = inherit_architecture(inherited, legacy.config_dict())
            self.assertEqual(model_config(259, restored).to_dict(), legacy.config_dict())


if __name__ == '__main__':
    unittest.main()
