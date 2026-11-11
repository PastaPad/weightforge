"""Weight-space experiments on Hugging Face checkpoints (Qwen2 family).

This module is the HF counterpart of ``evaluate.py``: it scores HF state dicts
with the minimal Qwen2 implementation in ``hf_models.py`` and runs the same
interpolation / task-vector / probe protocol used for the scratchllm smoke test.

A pair is scored on one or more evaluation streams so that a delta can be judged
on its own task as well as on a control task.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..training import fixed_windows
from .corpus import Checkpoint, load_hf_checkpoint
from .evaluate import pearson, ridge_loo_r2, weight_features
from .functional import apply_vector, evaluate_hf_nll, interpolate, load_stream, task_vector
from .hf_models import Qwen2Config, load_qwen2_state


@dataclass
class HFEvalSet:
    """A token stream plus the windows scored by next-token NLL."""

    stream: np.memmap
    windows: list[tuple[int, int]]
    name: str = "english"

    @classmethod
    def from_file(cls, path, window: int = 1024, count: int = 4, seed: int = 0,
                  name: str = "english") -> "HFEvalSet":
        stream = load_stream(path)
        return cls(stream=stream, windows=fixed_windows(len(stream), window, count, seed),
                   name=name)


@dataclass
class PairResult:
    """Everything measured for one (base, tuned) checkpoint pair."""

    base_id: str
    tuned_id: str
    alphas: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    interpolation: dict[str, list[float]] = field(default_factory=dict)
    task_vector: dict[str, list[float]] = field(default_factory=dict)
    probe: dict = field(default_factory=dict)
    delta_per_tensor: dict[str, float] = field(default_factory=dict)
    connectivity: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "base": self.base_id,
            "tuned": self.tuned_id,
            "alphas": self.alphas,
            "scales": self.scales,
            "interpolation_nll": self.interpolation,
            "task_vector_nll": self.task_vector,
            "probe": self.probe,
            "connectivity": self.connectivity,
            "delta_per_tensor": self.delta_per_tensor,
        }


def score_state(state, config: Qwen2Config, eval_sets, device: str = "cpu",
                batch_size: int = 1) -> dict[str, float]:
    """Next-token NLL of one HF state dict on each evaluation set."""
    if isinstance(eval_sets, HFEvalSet):
        eval_sets = {eval_sets.name: eval_sets}
    model = load_qwen2_state(state, config, device=device)
    try:
        return {name: evaluate_hf_nll(model, eval_set.stream, eval_set.windows,
                                     batch_size, device)
                for name, eval_set in eval_sets.items()}
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def relative_delta_per_tensor(base_state, tuned_state) -> dict[str, float]:
    """Frobenius norm of each weight delta, relative to the base tensor norm."""
    result = {}
    for key in sorted(base_state):
        base = base_state[key].float()
        delta = tuned_state[key].float() - base
        norm = base.norm().item() or 1.0
        result[key] = delta.norm().item() / norm
    return result


def cosine_per_tensor(base_state, tuned_state) -> dict[str, float]:
    """Cosine similarity between matching tensors, flattened."""
    result = {}
    for key in sorted(base_state):
        a = base_state[key].float().reshape(-1)
        b = tuned_state[key].float().reshape(-1)
        if a.norm() == 0 or b.norm() == 0:
            continue
        result[key] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return result


def weight_similarity(base_state, tuned_state) -> dict:
    """Connectivity diagnostics for a pair.

    True parent/child finetunes sit close to cosine 1.0. Pairs that were retrained
    from a different starting point, or tuned very heavily, sit far lower, and
    their linear path is unlikely to be loss-connected.
    """
    cosines = cosine_per_tensor(base_state, tuned_state)
    values = list(cosines.values())
    deltas = relative_delta_per_tensor(base_state, tuned_state)
    weakest = sorted(cosines.items(), key=lambda kv: kv[1])[:10]
    return {
        "median_cosine": float(np.median(values)) if values else float("nan"),
        "min_cosine": float(min(values)) if values else float("nan"),
        "median_relative_delta": float(np.median(list(deltas.values()))) if deltas else float("nan"),
        "embedding_cosine": cosines.get("model.embed_tokens.weight"),
        "weakest_tensors": [{"key": key, "cosine": value} for key, value in weakest],
    }


def connectivity_label(median_cosine: float) -> str:
    """Coarse bucket for how linearly connected a pair is likely to be."""
    if median_cosine >= 0.99:
        return "sibling"
    if median_cosine >= 0.95:
        return "finetune"
    if median_cosine >= 0.7:
        return "continued-pretraining"
    return "unrelated"


def _as_eval_sets(eval_sets) -> dict[str, HFEvalSet]:
    if isinstance(eval_sets, HFEvalSet):
        return {eval_sets.name: eval_sets}
    return dict(eval_sets)


def run_pair(base: Checkpoint, tuned: Checkpoint, eval_sets, *,
             alphas: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
             scales: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
             device: str = "cpu", batch_size: int = 1, probe_l2: float = 10.0,
             progress=print) -> PairResult:
    """Interpolation, task-vector, and probe sweep for one compatible pair."""
    eval_sets = _as_eval_sets(eval_sets)
    config = Qwen2Config.from_dict(base.model_config)
    alphas = list(alphas)
    scales = list(scales)
    result = PairResult(base_id=base.metadata.get("model_id", base.name),
                        tuned_id=tuned.metadata.get("model_id", tuned.name),
                        alphas=alphas, scales=scales,
                        interpolation={name: [] for name in eval_sets},
                        task_vector={name: [] for name in eval_sets})

    for alpha in alphas:
        state = interpolate(base.state_dict, tuned.state_dict, alpha)
        row = score_state(state, config, eval_sets, device, batch_size)
        del state
        for name, value in row.items():
            result.interpolation[name].append(value)
        progress(f"  alpha={alpha:.2f} " +
                 " ".join(f"{name}={value:.4f}" for name, value in row.items()))

    vector = task_vector(base.state_dict, tuned.state_dict)
    for scale in scales:
        state = apply_vector(base.state_dict, vector, scale)
        row = score_state(state, config, eval_sets, device, batch_size)
        del state
        for name, value in row.items():
            result.task_vector[name].append(value)
        progress(f"  scale={scale:.2f} " +
                 " ".join(f"{name}={value:.4f}" for name, value in row.items()))

    features = [weight_features(interpolate(base.state_dict, tuned.state_dict, a))
                for a in alphas]
    result.probe = {
        "alphas": alphas,
        "scales": scales,
        "pearson_alpha_nll": {name: pearson(alphas, values)
                              for name, values in result.interpolation.items()},
        "ridge_loo_r2_interpolation_nll": {
            name: ridge_loo_r2(features, values, l2=probe_l2)[0]
            for name, values in result.interpolation.items()},
    }
    result.delta_per_tensor = relative_delta_per_tensor(base.state_dict, tuned.state_dict)
    result.connectivity = weight_similarity(base.state_dict, tuned.state_dict)
    result.connectivity["label"] = connectivity_label(result.connectivity["median_cosine"])
    return result


ARCH_KEYS = ("architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps",
             "rope_theta", "tie_word_embeddings")


def arch_signature(config: dict) -> dict:
    """Architecture-defining fields of an HF config.

    Provenance metadata is ignored, and so is ``max_position_embeddings``: the
    base and instruct repos often disagree on it (131072 vs 32768) while the
    tensors are identical. Weight shapes are checked separately.
    """
    return {key: config.get(key) for key in ARCH_KEYS}


def load_pair(base_id: str, tuned_id: str) -> tuple[Checkpoint, Checkpoint]:
    """Load two local HF checkpoints and require matching architecture and shapes."""
    base = load_hf_checkpoint(base_id)
    tuned = load_hf_checkpoint(tuned_id)
    if arch_signature(base.model_config) != arch_signature(tuned.model_config):
        raise ValueError(f"{base_id} and {tuned_id} differ in architecture")
    if base.state_dict.keys() != tuned.state_dict.keys():
        raise ValueError(f"{base_id} and {tuned_id} differ in state-dict keys")
    for key, tensor in base.state_dict.items():
        if tensor.shape != tuned.state_dict[key].shape:
            raise ValueError(f"{base_id} and {tuned_id} differ in shape for {key}")
    return base, tuned


def load_eval_specs(specs) -> dict[str, HFEvalSet]:
    """Build eval sets from a corpus spec (single ``eval`` or a list ``evals``)."""
    if isinstance(specs, dict):
        specs = [specs]
    eval_sets = {}
    for spec in specs:
        eval_set = HFEvalSet.from_file(spec["val"], window=spec.get("window", 1024),
                                       count=spec.get("windows", 4),
                                       seed=spec.get("seed", 0),
                                       name=spec.get("name", "english"))
        eval_sets[eval_set.name] = eval_set
    return eval_sets


def save_result(result: PairResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return path


def run_corpus(config_path, out_dir, *, device: str | None = None,
               skip_existing: bool = True, progress=print) -> list[PairResult]:
    """Run every pair in a corpus spec and write one JSON result per pair."""
    config = json.loads(Path(config_path).read_text())
    specs = config.get("evals") or [config["eval"]]
    eval_sets = load_eval_specs(specs)
    device = device or config.get("device", "cpu")
    batch_size = config.get("batch_size", 1)
    out_dir = Path(out_dir)

    results = []
    for pair in config["pairs"]:
        target = out_dir / f"{pair['id']}.json"
        if skip_existing and target.exists():
            progress(f"skip {pair['id']} (exists)")
            continue
        progress(f"pair {pair['id']}: {pair['base']} -> {pair['tuned']}")
        base, tuned = load_pair(pair["base"], pair["tuned"])
        result = run_pair(base, tuned, eval_sets, device=device, batch_size=batch_size,
                          alphas=config.get("alphas", (0.0, 0.25, 0.5, 0.75, 1.0)),
                          scales=config.get("scales", (0.0, 0.5, 1.0, 1.5, 2.0)),
                          progress=progress)
        payload = result.as_dict()
        payload["meta"] = {key: value for key, value in pair.items()
                           if key not in {"base", "tuned"}}
        payload["evals"] = specs
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")
        progress(f"  wrote {target}")
        results.append(result)
        del base, tuned
    return results


def summarize_corpus(out_dir) -> list[dict]:
    """Read per-pair result JSONs and derive comparable corpus statistics."""
    rows = []
    for path in sorted(Path(out_dir).glob("*.json")):
        data = json.loads(path.read_text())
        connectivity = data.get("connectivity", {})
        row = {
            "id": path.stem,
            "label": connectivity.get("label", "?"),
            "median_cosine": connectivity.get("median_cosine"),
            "median_relative_delta": connectivity.get("median_relative_delta"),
            "evals": {},
        }
        for name, values in data.get("interpolation_nll", {}).items():
            endpoints = (values[0], values[-1])
            row["evals"][name] = {
                "base": endpoints[0],
                "tuned": endpoints[1],
                "peak_gap": max(values) - max(endpoints),
            }
        rows.append(row)
    return rows
