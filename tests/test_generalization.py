"""Generalization machinery checks using fixed tensors; never optimize a model."""
import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

from scratchllm.jobs import Jobs, TrainRequest
from scratchllm.model import GPT, ModelConfig
from scratchllm.serve import create_app
from scratchllm.train import save_checkpoints
from scratchllm.training import (EarlyStopping, ShuffledStream, WeightEMA, evaluate,
                                evaluation_plan, fixed_windows, learning_rate, make_batch, training_plan)
from test_serving_selection import FakeGenerator


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def model(self):
        return GPT(ModelConfig(vocab_size=32, context=8, width=16, layers=2, heads=4, dropout=0.4))

    def test_full_evaluation_covers_targets_once_including_tail(self):
        stream = np.arange(20) % 32
        windows = fixed_windows(len(stream), 8, 100, 42)
        targets = [i for start, n in windows for i in range(start + 1, start + n + 1)]
        self.assertEqual(targets, list(range(1, len(stream))))
        x, y = make_batch(stream, windows, 'cpu')
        self.assertEqual(y[-1].tolist(), [17, 18, 19, -100, -100, -100, -100, -100])
        self.assertEqual(x[0].tolist(), list(range(8)))
        self.assertEqual(y[0].tolist(), list(range(1, 9)))

    def test_fixed_subset_is_unique_repeatable_and_rng_independent(self):
        streams = {split: np.arange(10003) % 32 for split in ('train', 'val')}
        cfg = dict(context=8, batch_size=3, eval_batches=2, seed=42)
        before = np.random.get_state()
        plan = evaluation_plan(streams, cfg)
        self.assertEqual(plan, evaluation_plan(streams, cfg))
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        for rows in plan['windows'].values():
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(set(rows)), 6)
        self.assertNotEqual(plan['windows']['train'], plan['windows']['val'])

    def test_nll_is_token_weighted_repeatable_and_does_not_mutate_model(self):
        model = self.model().train()
        stream = np.arange(20) % 32
        windows = fixed_windows(len(stream), 8, 100, 42)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        # An independent per-token calculation, including uneven final batches.
        model.eval()
        expected = []
        with torch.inference_mode():
            for row in windows:
                x, y = make_batch(stream, [row], 'cpu')
                logits, _ = model(x, y)
                expected.extend((-logits.log_softmax(-1).gather(-1, y.unsqueeze(-1))).flatten().tolist())
        model.train()
        for batch_size in (1, 2, 3):
            loss = evaluate(model, stream, windows, batch_size, 'cpu')
            self.assertAlmostEqual(loss, sum(expected) / len(expected), places=6)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_causal_predictions_and_smoothing_are_separate_from_nll(self):
        model = self.model().eval()
        x = torch.arange(8).unsqueeze(0)
        y = x + 1
        y[0, -1] = -100
        with torch.inference_mode():
            logits, nll = model(x, y)
            _, smooth = model(x, y, label_smoothing=0.2)
            expected_uniform = -logits[0, :-1].log_softmax(-1).mean()
            torch.testing.assert_close(smooth, 0.8 * nll + 0.2 * expected_uniform)
            changed = x.clone()
            changed[:, 4:] += 10
            later, _ = model(changed, y)
            torch.testing.assert_close(logits[:, :4], later[:, :4])

    def test_shuffled_passes_have_no_missing_or_repeated_targets_and_resume(self):
        sampler = ShuffledStream(54, 8, 42)
        for _ in range(3):
            first = sampler.take(1)
            rows = first + sampler.take(len(sampler.windows) - 1)
            targets = [i for start, n in rows for i in range(start + 1, start + n + 1)]
            self.assertEqual(sorted(targets), list(range(1, 54)))
        sampler.take(2)
        saved = copy.deepcopy(sampler.state_dict())
        restored = ShuffledStream(54, 8, 999)
        restored.load_state_dict(saved)
        self.assertEqual(sampler.take(100), restored.take(100))
        self.assertEqual(sampler.state_dict(), restored.state_dict())

    def test_budget_rescales_warmup_decay_and_eval_cadence(self):
        cfg = TrainRequest(budget_mode='updates', context=256).model_dump()
        cfg.update(batch_size=4, accumulation=8, warmup_steps=200, max_epochs=20)
        meta = dict(vocab_size=8192, tokens=dict(train=106508, val=16016))
        plan = training_plan(cfg, meta)
        self.assertLess(plan['steps'], cfg['steps'])
        self.assertLessEqual(plan['warmup_steps'], plan['steps'] // 10)
        self.assertLessEqual(plan['planned_epochs'], cfg['max_epochs'] + plan['tokens_per_step'] / (106508 - 1))
        self.assertEqual(plan['eval_every'], 14)
        self.assertAlmostEqual(learning_rate(plan['warmup_steps'], cfg, plan), cfg['learning_rate'])
        self.assertAlmostEqual(learning_rate(plan['steps'] - 1, cfg, plan), cfg['min_learning_rate'])
        cfg.update(max_epochs=0)
        self.assertEqual(training_plan(cfg, meta)['steps'], cfg['steps'])
        cfg.update(max_epochs=0.001)
        short = training_plan(cfg, meta)
        self.assertEqual((short['steps'], short['warmup_steps']), (1, 0))
        self.assertTrue(math.isfinite(learning_rate(0, cfg, short)))

    def test_pass_budget_grows_with_data_and_tracks_effective_batch(self):
        request = TrainRequest(budget_mode='passes', max_epochs=20, steps=987, accumulation=2)
        cfg = request.model_dump()
        small = dict(vocab_size=10949, tokens=dict(train=101000, val=15493))
        large = dict(vocab_size=16384, tokens=dict(train=1587273, val=594931))
        self.assertEqual(training_plan(cfg, small)['steps'], 987)
        self.assertEqual(training_plan(cfg, large)['steps'], 15501)
        self.assertEqual(training_plan(dict(cfg, accumulation=8), large)['steps'], 3876)
        # The stale fixed value cannot cap a pass-sized run, including its LR schedule.
        plan = training_plan(dict(cfg, steps=1), large)
        self.assertEqual(plan['steps'], 15501)
        self.assertAlmostEqual(learning_rate(15500, cfg, plan), cfg['min_learning_rate'])
        with self.assertRaises(ValueError):
            TrainRequest(budget_mode='passes', max_epochs=0)
        with self.assertRaises(ValueError):
            training_plan(dict(cfg, max_epochs=0), large)

    def test_fixed_and_legacy_budgets_stay_explicit(self):
        cfg = TrainRequest(budget_mode='updates', steps=987, max_epochs=0, accumulation=2).model_dump()
        small = dict(vocab_size=8192, tokens=dict(train=101000, val=15493))
        large = dict(vocab_size=8192, tokens=dict(train=1587273, val=594931))
        self.assertEqual(training_plan(cfg, small)['steps'], 987)
        self.assertEqual(training_plan(cfg, large)['steps'], 987)
        self.assertLess(training_plan(cfg, large)['planned_epochs'], 2)
        legacy = {k: v for k, v in cfg.items() if k != 'budget_mode'}
        self.assertEqual(training_plan(legacy, large), training_plan(cfg, large))

    def test_pass_mode_worker_config_matches_preview_without_spawning(self):
        from scratchllm.links import TEXT_POLICY_VERSION
        meta = dict(vocab_size=16384, tokens=dict(train=1587273, val=594931), documents=dict(train=937, val=106), text_policy=TEXT_POLICY_VERSION)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data/new').mkdir(parents=True)
            (root/'data/new/meta.json').write_text(json.dumps(meta))
            jobs = Jobs(root, 'cpu')
            request = TrainRequest(data='data/new', out='runs/new', budget_mode='passes', max_epochs=20, steps=987)
            preview = jobs.preview_training(request)
            self.assertEqual(preview['steps'], 3876)
            self.assertEqual(preview['data'], 'data/new')
            self.assertEqual(preview['limit_reason'], 'data_passes')
            with patch('scratchllm.jobs.subprocess.Popen', side_effect=RuntimeError('worker launch intercepted')):
                with self.assertRaisesRegex(RuntimeError, 'worker launch intercepted'):
                    jobs.start('train', request)
            config = json.loads((root/'runs/new/web-config.json').read_text())
            self.assertEqual(config['steps'], preview['steps'])
            self.assertEqual(training_plan(config, meta)['steps'], preview['steps'])
            self.assertFalse((root/'runs/new/last.pt').exists())
            self.assertFalse(jobs.active())

    def test_dataset_catalog_lists_latest_complete_dataset_independently_of_model(self):
        meta = dict(vocab_size=300, tokens=dict(train=1000, val=300), documents=dict(train=10, val=2))
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch('scratchllm.jobs.subprocess.Popen') as spawn:
            root = Path(directory)
            (root/'old-model').mkdir()
            (root/'old-model/best.pt').write_text('fake')
            (root/'old-model/web-job.json').write_text(json.dumps({'data': 'data/old'}))
            for name, timestamp in [('old', 10), ('larger', 20), ('partial', 30)]:
                folder = root/'data'/name
                folder.mkdir(parents=True)
                (folder/'meta.json').write_text(json.dumps(meta))
                (folder/'tokenizer.json').write_text('{}')
                (folder/'train.bin').write_bytes(b'\0' * 4000)
                if name != 'partial':
                    (folder/'val.bin').write_bytes(b'\0' * 1200)
                os.utime(folder/'meta.json', (timestamp, timestamp))
            with TestClient(create_app(root/'old-model/best.pt', 'cpu', workspace=root)) as client:
                data = client.get('/datasets').json()
                self.assertEqual(data['latest'], 'data/larger')
                self.assertEqual([row['path'] for row in data['datasets']], ['data/larger', 'data/old'])
                self.assertEqual(client.get('/health').json()['dataset'], 'data/old')
                self.assertIsNone(client.get('/jobs').json()['job'])
                self.assertEqual(client.get('/').headers['cache-control'], 'no-store')
            spawn.assert_not_called()

    def test_early_stopping_tracks_small_gains_and_survives_resume(self):
        early = EarlyStopping()
        for step, loss in enumerate([6, 5.6, 5.55], 1):
            improved, stopped = early.observe(loss, step, 0.1, 3)
            self.assertTrue(improved)
            self.assertFalse(stopped)
        self.assertEqual(early.bad_evals, 1)
        resumed = EarlyStopping(**early.state_dict())
        resumed.observe(5.48, 4, 0.1, 3)
        self.assertEqual(resumed.bad_evals, 0)
        for step, loss in enumerate([5.8, 6, 7], 5):
            improved, stopped = resumed.observe(loss, step, 0.1, 3)
            self.assertFalse(improved)
        self.assertTrue(stopped)
        self.assertEqual((resumed.best, resumed.best_step), (5.48, 4))
        self.assertFalse(resumed.observe(8, 8, 0.1, 0)[1])
        with self.assertRaises(ValueError):
            resumed.observe(float('nan'), 9, 0.1, 3)

    def test_ema_tied_weights_resume_and_exception_restoration(self):
        model = self.model()
        ema = WeightEMA(model, 0.9)
        self.assertNotIn('head.weight', ema.shadow)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1)
            ema.update(model)
            for p in model.parameters():
                p.fill_(3)
            ema.update(model)
        restored = WeightEMA(model, 0.9)
        restored.load_state_dict(copy.deepcopy(ema.state_dict()))
        self.assertEqual(restored.updates, 2)
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with restored.apply(model):
                self.assertEqual(model.token.weight[0, 0].item(), 2)
                self.assertIs(model.token.weight, model.head.weight)
                raise RuntimeError('test exception')
        self.assertEqual(model.token.weight[0, 0].item(), 3)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_best_ema_checkpoint_preserves_raw_resume_and_prior_best(self):
        model = self.model()
        ema = WeightEMA(model, 0.98)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1)
            ema.update(model)
            for p in model.parameters():
                p.fill_(2)
        state = dict(model=model.state_dict(), model_config=model.config_dict(), tokenizer_json='test',
                     step=10, best_val=5.5, data_hashes={}, evaluation={'protocol': 'fixed-token-nll-v1'},
                     ema=ema.state_dict(), early_stopping=EarlyStopping(best=5.5, best_step=10).state_dict())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_checkpoints(root, state, model, ema, 'ema', True)
            best = torch.load(root/'best.pt', weights_only=True)
            last = torch.load(root/'last.pt', weights_only=True)
            self.assertEqual(best['selected_weights'], 'ema')
            self.assertEqual(best['model']['head.weight'][0, 0].item(), 1)
            self.assertEqual(last['model']['head.weight'][0, 0].item(), 2)
            self.assertEqual(model.token.weight[0, 0].item(), 2)
            resumed = self.model()
            resumed.load_state_dict(last['model'])
            self.assertIs(resumed.head.weight, resumed.token.weight)
            previous_best = (root/'best.pt').read_bytes()
            state['step'] = 20
            save_checkpoints(root, state, model, ema, 'raw', False)
            self.assertEqual((root/'best.pt').read_bytes(), previous_best)
            self.assertEqual(torch.load(root/'last.pt', weights_only=True)['step'], 20)

    def test_preview_uses_real_dimensions_and_does_not_start_a_job(self):
        config = TrainRequest(width=16, context=8, layers=2, heads=4, batch_size=4).model_dump()
        meta = dict(vocab_size=32, tokens=dict(train=10000, val=1000), documents=dict(train=79, val=10))
        self.assertEqual(training_plan(config, meta)['parameters'], sum(p.numel() for p in self.model().parameters()))
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch('scratchllm.jobs.subprocess.Popen') as spawn:
            root = Path(directory)
            (root/'fake.pt').write_text('fake')
            (root/'data').mkdir()
            (root/'data/meta.json').write_text(json.dumps(meta))
            with TestClient(create_app(root/'fake.pt', 'cpu', workspace=root)) as client:
                response = client.post('/training/plan', json=dict(config, data='data'))
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['parameters'], sum(p.numel() for p in self.model().parameters()))
                self.assertTrue(response.json()['full_validation'])
                self.assertIsNone(client.get('/jobs').json()['job'])
                self.assertEqual(client.post('/training/plan', json={'data': '../outside'}).status_code, 400)
                for field in ('max_epochs', 'ema_decay', 'label_smoothing', 'early_stopping_patience'):
                    self.assertEqual(client.post('/training/plan', json=dict(config, data='data', **{field: -1})).status_code, 422)
            spawn.assert_not_called()
            self.assertFalse((root/'runs').exists())


if __name__ == '__main__':
    unittest.main()
