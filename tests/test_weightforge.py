"""Self-contained tests for weight-space loading, arithmetic, probes, and the Qwen2 path.

Everything is built from synthetic checkpoints in a temp directory: no downloads and
no dependence on local training runs.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scratchllm.model import GPT, ModelConfig
from scratchllm.weightforge import (apply_vector, assert_compatible, discover,
                                    evaluate_nll, interpolate, load_checkpoint,
                                    model_from_state, ridge_loo_r2, scalar_statistics,
                                    task_vector, weight_features)
from scratchllm.weightforge.cli import build_parser
from scratchllm.weightforge.corpus import Checkpoint
from scratchllm.weightforge.functional import evaluate_hf_nll
from scratchllm.weightforge.hf_models import Qwen2Config, load_qwen2_state, rotate_half


def tiny_config(vocab: int = 64) -> ModelConfig:
    return ModelConfig(vocab_size=vocab, context=16, width=32, layers=2, heads=4,
                       kv_heads=2, ffn_width=64, architecture="modern")


def save_checkpoint(directory: Path, name: str, config: ModelConfig, seed: int) -> Path:
    torch.manual_seed(seed)
    model = GPT(config)
    path = directory / name / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": config.to_dict(),
                "tokenizer_json": "{}", "step": 1, "best_val": 0.5}, path)
    return path


def qwen_config(vocab: int = 96, hidden: int = 32, layers: int = 2, heads: int = 4,
                kv_heads: int = 2, inter: int = 64) -> Qwen2Config:
    return Qwen2Config(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                       num_attention_heads=heads, num_key_value_heads=kv_heads,
                       intermediate_size=inter)


def hf_state_dict(config: Qwen2Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """State dict using the exact HF Qwen2 naming convention."""
    torch.manual_seed(seed)
    head_dim = config.hidden_size // config.num_attention_heads
    kv_dim = config.num_key_value_heads * head_dim
    state = {
        "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size),
        "model.norm.weight": torch.ones(config.hidden_size),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        state[f"{prefix}.input_layernorm.weight"] = torch.ones(config.hidden_size)
        state[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(config.hidden_size)
        for proj, shape, bias in (("q_proj", (config.hidden_size, config.hidden_size), True),
                                  ("k_proj", (kv_dim, config.hidden_size), True),
                                  ("v_proj", (kv_dim, config.hidden_size), True),
                                  ("o_proj", (config.hidden_size, config.hidden_size), False)):
            state[f"{prefix}.self_attn.{proj}.weight"] = torch.randn(shape) * 0.05
            if bias:
                state[f"{prefix}.self_attn.{proj}.bias"] = torch.zeros(shape[0])
        state[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(config.intermediate_size,
                                                             config.hidden_size) * 0.05
        state[f"{prefix}.mlp.up_proj.weight"] = torch.randn(config.intermediate_size,
                                                           config.hidden_size) * 0.05
        state[f"{prefix}.mlp.down_proj.weight"] = torch.randn(config.hidden_size,
                                                             config.intermediate_size) * 0.05
    return state


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = tiny_config()
        save_checkpoint(self.root, "a", self.config, seed=0)
        save_checkpoint(self.root, "b", self.config, seed=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_checkpoint_roundtrip(self):
        checkpoint = load_checkpoint(self.root / "a" / "best.pt")
        self.assertIsInstance(checkpoint, Checkpoint)
        self.assertEqual(checkpoint.model_config["vocab_size"], 64)
        self.assertEqual(checkpoint.name, "a")
        self.assertEqual(checkpoint.num_parameters,
                         sum(t.numel() for t in checkpoint.state_dict.values()))

    def test_discover(self):
        found = discover(self.root)
        self.assertEqual([c.name for c in found], ["a", "b"])

    def test_assert_compatible_accepts_same_config(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        b = load_checkpoint(self.root / "b" / "best.pt")
        assert_compatible(a, b)

    def test_assert_compatible_rejects_different_config(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        save_checkpoint(self.root, "c", tiny_config(vocab=32), seed=2)
        c = load_checkpoint(self.root / "c" / "best.pt")
        with self.assertRaises(ValueError):
            assert_compatible(a, c)

    def test_interpolation_endpoints(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        b = load_checkpoint(self.root / "b" / "best.pt")
        at_zero = interpolate(a.state_dict, b.state_dict, 0.0)
        at_one = interpolate(a.state_dict, b.state_dict, 1.0)
        for key in a.state_dict:
            self.assertTrue(torch.equal(at_zero[key], a.state_dict[key]))
            self.assertTrue(torch.equal(at_one[key], b.state_dict[key]))

    def test_interpolation_midpoint(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        b = load_checkpoint(self.root / "b" / "best.pt")
        mid = interpolate(a.state_dict, b.state_dict, 0.5)
        for key in a.state_dict:
            expected = (a.state_dict[key].double() + b.state_dict[key].double()) / 2
            self.assertTrue(torch.allclose(mid[key].double(), expected))

    def test_task_vector_roundtrip(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        b = load_checkpoint(self.root / "b" / "best.pt")
        vector = task_vector(a.state_dict, b.state_dict)
        rebuilt = apply_vector(a.state_dict, vector, 1.0)
        for key in a.state_dict:
            self.assertTrue(torch.allclose(rebuilt[key], b.state_dict[key]))

    def test_weight_features(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        features = weight_features(a.state_dict)
        self.assertEqual(features.shape, (6 * len(a.state_dict),))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_scalar_statistics(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        b = load_checkpoint(self.root / "b" / "best.pt")
        alphas = [0.0, 0.5, 1.0]
        stats = scalar_statistics(a, b, alphas)
        self.assertEqual(set(stats), {"delta_norm", "relative_delta_norm", "cosine"})
        self.assertEqual(stats["delta_norm"][0], 0.0)
        self.assertGreater(stats["delta_norm"][2], stats["delta_norm"][1])
        self.assertAlmostEqual(stats["cosine"][0], 1.0, places=5)

    def test_model_from_state_forward(self):
        config = tiny_config()
        source = GPT(config)
        clone = model_from_state(source.state_dict(), config)
        ids = torch.randint(0, 64, (2, 16))
        logits, loss = clone(ids, ids)
        self.assertEqual(tuple(logits.shape), (2, 16, 64))
        self.assertTrue(torch.isfinite(loss))
        self.assertIs(clone.token.weight, clone.head.weight)

    def test_evaluate_nll_is_finite(self):
        a = load_checkpoint(self.root / "a" / "best.pt")
        model = model_from_state(a.state_dict, ModelConfig(**a.model_config))
        stream = np.arange(256, dtype="<u4") % 64
        windows = [(0, 16), (16, 16), (32, 16)]
        nll = evaluate_nll(model, stream, windows, 1, "cpu")
        self.assertTrue(np.isfinite(nll))
        self.assertGreater(nll, 0.0)

    def test_ridge_loo_recovers_linear_target(self):
        rng = np.random.RandomState(0)
        features = rng.randn(20, 5)
        targets = features @ np.array([1.0, -2.0, 0.5, 0.0, 1.5]) + 0.1
        r2, _ = ridge_loo_r2(features, targets, l2=1e-6)
        self.assertGreater(r2, 0.99)


class Qwen2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = qwen_config()
        cls.state = hf_state_dict(cls.config, seed=0)

    def test_rotate_half_swaps_halves(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = rotate_half(x)
        self.assertTrue(torch.equal(rotated, torch.tensor([[-3.0, -4.0, 1.0, 2.0]])))

    def test_load_state_dict_is_strict_and_shaped(self):
        model = load_qwen2_state(self.state, self.config, device="cpu")
        self.assertEqual(model.lm_head.weight.shape, (self.config.vocab_size,
                                                      self.config.hidden_size))
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_forward_shapes(self):
        model = load_qwen2_state(self.state, self.config, device="cpu")
        ids = torch.randint(0, self.config.vocab_size, (2, 12))
        logits, loss, _ = model(ids, ids)
        self.assertEqual(tuple(logits.shape), (2, 12, self.config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_uniform_model_nll_is_uniform_across_ragged_windows(self):
        """Zeroed weights give uniform logits, so ragged windows must still score ln(vocab)."""
        state = {key: torch.zeros_like(value) for key, value in self.state.items()}
        model = load_qwen2_state(state, self.config, device="cpu")
        stream = (np.arange(128) % self.config.vocab_size).astype("<u4")
        windows = [(0, 64), (64, 32), (96, 24)]
        nll = evaluate_hf_nll(model, stream, windows, 2, "cpu")
        self.assertAlmostEqual(nll, float(np.log(self.config.vocab_size)), places=4)

    def test_evaluate_hf_nll_matches_manual_cross_entropy(self):
        model = load_qwen2_state(self.state, self.config, device="cpu")
        stream = (np.arange(96) % self.config.vocab_size).astype("<u4")
        windows = [(0, 48), (48, 48)]
        nll = evaluate_hf_nll(model, stream, windows, 1, "cpu")

        total, tokens = 0.0, 0
        for start, length in windows:
            ids = torch.tensor(stream[start:start + length].astype("int64"))[None]
            with torch.inference_mode():
                logits, _, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
            total += loss.item() * (length - 1)
            tokens += length - 1
        self.assertAlmostEqual(nll, total / tokens, places=5)

    def test_rope_preserves_norms(self):
        from scratchllm.weightforge.hf_models import RotaryEmbedding, apply_rotary_pos_emb
        head_dim = 8
        rope = RotaryEmbedding(head_dim, theta=10000.0)
        q = torch.randn(2, 3, 5, head_dim)
        k = torch.randn(2, 3, 5, head_dim)
        positions = torch.arange(5)
        cos, sin = rope(q, positions)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertTrue(torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5))
        # position 0 is the identity rotation
        self.assertTrue(torch.allclose(q_out[..., 0, :], q[..., 0, :], atol=1e-6))


class ComposeTests(unittest.TestCase):
    def test_flat_delta_cosine(self):
        from scratchllm.weightforge.compose import flat_delta_cosine
        a = {"x": torch.tensor([1.0, 0.0, 0.0]), "y": torch.tensor([0.0, 2.0, 0.0])}
        b = {"x": torch.tensor([2.0, 0.0, 0.0]), "y": torch.tensor([0.0, 4.0, 0.0])}
        self.assertAlmostEqual(flat_delta_cosine(a, b), 1.0, places=6)

        c = {"x": torch.tensor([-1.0, 0.0, 0.0]), "y": torch.tensor([0.0, -2.0, 0.0])}
        self.assertAlmostEqual(flat_delta_cosine(a, c), -1.0, places=6)

        d = {"x": torch.tensor([0.0, 0.0, 0.0]), "y": torch.tensor([0.0, 0.0, 0.0])}
        self.assertTrue(np.isnan(flat_delta_cosine(a, d)))

    def test_flat_delta_norm(self):
        from scratchllm.weightforge.compose import flat_delta_norm
        vector = {"x": torch.tensor([3.0, 0.0]), "y": torch.tensor([0.0, 4.0])}
        self.assertAlmostEqual(flat_delta_norm(vector), 5.0, places=6)


class LowRankTests(unittest.TestCase):
    def test_projection_recovers_exact_coefficients(self):
        from scratchllm.weightforge.lowrank import projection_coefficients
        gram = np.eye(3)
        inner = np.array([0.7, 0.2, -0.4])
        coefficients = projection_coefficients(gram, inner)
        np.testing.assert_allclose(coefficients, inner, atol=1e-9)

    def test_projection_handles_correlated_basis(self):
        from scratchllm.weightforge.lowrank import projection_coefficients
        gram = np.array([[1.0, 1.0], [1.0, 1.0]])
        inner = np.array([0.5, 0.5])
        coefficients = projection_coefficients(gram, inner, rank=1)
        # rank-1 basis: coefficients reproduce the vector projection (length 0.5)
        self.assertAlmostEqual(float(coefficients @ gram @ coefficients) ** 0.5, 0.5,
                               places=6)

    def test_combine_is_linear(self):
        from scratchllm.weightforge.lowrank import combine
        base = {"w": torch.tensor([1.0, 1.0])}
        deltas = {"a": {"w": torch.tensor([1.0, 0.0])},
                  "b": {"w": torch.tensor([0.0, 2.0])}}
        state = combine(base, deltas, [2.0, 0.5])
        self.assertTrue(torch.allclose(state["w"], torch.tensor([3.0, 2.0])))


class NumericsTests(unittest.TestCase):
    def test_cosine_of_identical_bf16_tensor_is_exactly_one(self):
        import torch
        from scratchllm.weightforge.numerics import stable_cosine
        generator = torch.Generator().manual_seed(0)
        tensor = torch.randn(2_000_000, generator=generator).to(torch.bfloat16)
        self.assertAlmostEqual(stable_cosine(tensor, tensor), 1.0, places=12)

    def test_cosine_handles_chunk_boundaries(self):
        import torch
        from scratchllm.weightforge.numerics import _CHUNK, stable_cosine
        generator = torch.Generator().manual_seed(1)
        a = torch.randn(_CHUNK + 7, generator=generator)
        b = -a.clone()
        self.assertAlmostEqual(stable_cosine(a, b), -1.0, places=9)

    def test_stable_dot_matches_float64_reference(self):
        import torch
        from scratchllm.weightforge.numerics import stable_dot
        generator = torch.Generator().manual_seed(2)
        a = torch.randn(100_003, generator=generator).to(torch.bfloat16)
        b = torch.randn(100_003, generator=generator).to(torch.bfloat16)
        expected = (a.double() @ b.double()).item()
        self.assertAlmostEqual(stable_dot(a, b), expected, places=6)


class AdapterTests(unittest.TestCase):
    def test_normalize_key_strips_peft_prefixes(self):
        from scratchllm.weightforge.adapters import _normalize_key
        self.assertEqual(_normalize_key("base_model.model.model.layers.3.mlp.down_proj.lora_A.weight"),
                         "model.layers.3.mlp.down_proj.lora_A.weight")

    def test_scale_uses_alpha_over_rank(self):
        from scratchllm.weightforge.adapters import adapter_scale
        self.assertAlmostEqual(adapter_scale({"r": 1, "lora_alpha": 2}), 2.0)
        self.assertAlmostEqual(adapter_scale({"r": 8, "lora_alpha": 16}), 2.0)

    def test_scale_honours_patterns(self):
        from scratchllm.weightforge.adapters import adapter_scale
        config = {"r": 8, "lora_alpha": 16,
                  "rank_pattern": {"down_proj": 2}, "alpha_pattern": {"down_proj": 8}}
        self.assertAlmostEqual(adapter_scale(config, "model.layers.0.mlp.down_proj"), 4.0)
        self.assertAlmostEqual(adapter_scale(config, "model.layers.0.self_attn.q_proj"), 2.0)


class VarianceTests(unittest.TestCase):
    def test_group_sum_of_squares_of_identical_vectors(self):
        import numpy as np
        from scratchllm.weightforge.variance import group_sum_of_squares
        gram = np.ones((3, 3))
        # three identical unit vectors, one group -> zero between-group variance
        self.assertAlmostEqual(group_sum_of_squares(gram, ["a", "a", "a"]), 0.0, places=9)

    def test_leave_one_out_on_identical_deltas_is_exact(self):
        import numpy as np
        from scratchllm.weightforge.variance import leave_one_out_from_gram
        gram = np.ones((4, 4))
        rows = leave_one_out_from_gram(gram)
        for row in rows:
            self.assertAlmostEqual(row["cosine"], 1.0, places=9)
            self.assertAlmostEqual(row["norm_ratio"], 1.0, places=9)

    def test_leave_one_out_on_orthogonal_basis_is_zero(self):
        import numpy as np
        from scratchllm.weightforge.variance import leave_one_out_from_gram
        gram = np.eye(4)
        rows = leave_one_out_from_gram(gram)
        for row in rows:
            self.assertAlmostEqual(abs(row["cosine"]), 0.0, places=9)

    def test_anova_split_on_orthogonal_factors(self):
        import numpy as np
        from scratchllm.weightforge.variance import two_way_anova
        gram = np.eye(4)
        recipe = ["r1", "r1", "r2", "r2"]
        data = ["d1", "d2", "d1", "d2"]
        result = two_way_anova(gram, recipe, data)
        # each factor splits 2 vs 2 with zero cross-group overlap -> half each
        self.assertAlmostEqual(result["recipe_share"] + result["data_share"] +
                               result["interaction_share"], 1.0, places=9)


class CLITests(unittest.TestCase):
    def test_smoke_parser(self):
        args = build_parser().parse_args([
            "smoke", "--base", "a.pt", "--dialogue", "b.pt",
            "--base-data", "base", "--dialogue-data", "dialogue"])
        self.assertEqual(args.command, "smoke")
        self.assertEqual(args.context, 512)

    def test_hf_smoke_parser(self):
        args = build_parser().parse_args([
            "hf-smoke", "--base-id", "Qwen/Qwen2.5-0.5B",
            "--tuned-id", "Qwen/Qwen2.5-0.5B-Instruct", "--val", "val.bin"])
        self.assertEqual(args.command, "hf-smoke")
        self.assertEqual(args.window, 1024)
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
