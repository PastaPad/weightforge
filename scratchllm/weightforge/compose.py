"""Delta composition: does adding one weight edit to another behave additively?

Given a reference checkpoint and several tuned checkpoints, this measures each
delta alone, the sum of all deltas, and the interaction term between the
observed NLL and the additive prediction.

A negative interaction (actual NLL below the sum of individual effects) means the
deltas cooperate; a large positive interaction means they interfere and the
weights are being pushed out of the region either edit alone reaches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .corpus import Checkpoint, load_hf_checkpoint
from .functional import task_vector
from .hf_experiment import (HFEvalSet, arch_signature, load_eval_specs, score_state)
from .hf_models import Qwen2Config
from .numerics import stable_dot


@dataclass
class ComposeResult:
    base_id: str
    tuned_ids: list[str] = field(default_factory=list)
    base_nll: dict[str, float] = field(default_factory=dict)
    single_nll: dict[str, dict[str, float]] = field(default_factory=dict)
    combined_nll: dict[str, float] = field(default_factory=dict)
    interaction: dict[str, float] = field(default_factory=dict)
    delta_cosine: dict[str, float] = field(default_factory=dict)
    delta_norm: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "base": self.base_id,
            "tuned": self.tuned_ids,
            "base_nll": self.base_nll,
            "single_nll": self.single_nll,
            "combined_nll": self.combined_nll,
            "interaction": self.interaction,
            "delta_cosine": self.delta_cosine,
            "delta_norm": self.delta_norm,
        }


def flat_delta_cosine(vector_a, vector_b) -> float:
    """Cosine between two full-model deltas, computed without materializing them."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for key in vector_a:
        other = vector_b.get(key)
        if other is None:
            continue
        dot += stable_dot(vector_a[key], other)
        norm_a += stable_dot(vector_a[key], vector_a[key])
        norm_b += stable_dot(other, other)
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return dot / (norm_a ** 0.5 * norm_b ** 0.5)


def flat_delta_norm(vector) -> float:
    return sum(stable_dot(value, value) for value in vector.values()) ** 0.5


def compose(base: Checkpoint, tuned: dict[str, Checkpoint], eval_sets, *,
            device: str = "cpu", batch_size: int = 1, progress=print) -> ComposeResult:
    """Score deltas individually and jointly against an additive baseline."""
    eval_sets = {eval_sets.name: eval_sets} if isinstance(eval_sets, HFEvalSet) else eval_sets
    config = Qwen2Config.from_dict(base.model_config)
    result = ComposeResult(base_id=base.metadata.get("model_id", base.name),
                           tuned_ids=list(tuned))
    base_state = base.state_dict

    result.base_nll = score_state(base_state, config, eval_sets, device, batch_size)
    progress("  base " + " ".join(f"{k}={v:.4f}" for k, v in result.base_nll.items()))

    vectors = {}
    for name, checkpoint in tuned.items():
        vectors[name] = task_vector(base_state, checkpoint.state_dict)
        result.delta_norm[name] = flat_delta_norm(vectors[name])
        result.single_nll[name] = score_state(checkpoint.state_dict, config, eval_sets,
                                              device, batch_size)
        progress(f"  {name:24s} " +
                 " ".join(f"{k}={v:.4f}" for k, v in result.single_nll[name].items()))

    names = list(vectors)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            result.delta_cosine[f"{first}|{second}"] = flat_delta_cosine(vectors[first],
                                                                        vectors[second])

    total = {key: base_state[key] for key in base_state}
    for vector in vectors.values():
        total = {key: total[key] + vector[key] for key in total}
    result.combined_nll = score_state(total, config, eval_sets, device, batch_size)
    del total
    progress("  combined " + " ".join(f"{k}={v:.4f}" for k, v in result.combined_nll.items()))

    for eval_name, actual in result.combined_nll.items():
        predicted = result.base_nll[eval_name] + sum(
            rows[eval_name] - result.base_nll[eval_name]
            for rows in result.single_nll.values())
        result.interaction[eval_name] = actual - predicted
    return result


def run_compose(base_id: str, tuned_ids: list[str], eval_sets, *, device: str = "cpu",
                batch_size: int = 1, progress=print) -> ComposeResult:
    """Load a base plus every tuned checkpoint and compose their deltas."""
    base = load_hf_checkpoint(base_id)
    tuned = {}
    for model_id in tuned_ids:
        checkpoint = load_hf_checkpoint(model_id)
        if arch_signature(checkpoint.model_config) != arch_signature(base.model_config):
            raise ValueError(f"{model_id} differs in architecture from {base_id}")
        tuned[model_id] = checkpoint
    return compose(base, tuned, eval_sets, device=device, batch_size=batch_size,
                   progress=progress)


def save_compose(result: ComposeResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return path
