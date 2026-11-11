"""Can unseen weight deltas be reconstructed from a basis of known deltas?

This is the first zero-training "generation" experiment in weight space. Given a
base checkpoint and several tuned checkpoints, each tuned checkpoint defines a
delta d_i = w_i - w_base. For every held-out delta, this module:

1. builds the Gram matrix of all deltas (pairwise inner products),
2. projects the held-out delta onto the span of the others (optionally truncated
   to `rank`, optionally ridge-regularized),
3. reconstructs ``base + d_hat`` and scores it against the real tuned model.

Two things are reported, and they answer different questions:

* **delta-space fidelity** (``cosine``, ``norm_ratio``) says whether the basis
  actually spans the held-out delta. This is the quantity the generator needs.
* **NLL** says whether the reconstruction happens to behave like the target on
  the probes. A lower-than-true NLL is *not* success; it usually means the
  projection kept the harmless part of the delta and dropped the damaging part.

Checkpoints are loaded one at a time so a six-delta run stays far below the
15 GB memory of the reference machine.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .corpus import Checkpoint, load_hf_checkpoint
from .adapters import adapter_config, adapter_delta, is_adapter
from .functional import task_vector
from .hf_experiment import HFEvalSet, arch_signature, score_state
from .hf_models import Qwen2Config
from .numerics import stable_dot


def delta_inner(vector_a, vector_b) -> float:
    """Full-model inner product between two (possibly sparse) deltas."""
    total = 0.0
    for key in vector_a:
        other = vector_b.get(key)
        if other is None:
            continue
        total += stable_dot(vector_a[key], other)
    return total


def gram_matrix(deltas: dict[str, dict[str, torch.Tensor]],
                keys: list[str] | None = None) -> np.ndarray:
    """Pairwise inner products, optionally restricted to a set of tensors.

    Restriction matters when deltas have different supports: a LoRA that only
    touches one matrix and a LoRA that touches every matrix are nearly orthogonal
    simply because they write to different places. Comparing them on the shared
    tensors is the only fair test of directional agreement.
    """
    names = list(deltas)
    if keys is not None:
        deltas = {name: {key: deltas[name][key] for key in keys if key in deltas[name]}
                  for name in names}
    gram = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, first in enumerate(names):
        gram[i, i] = delta_inner(deltas[first], deltas[first])
        for j in range(i + 1, len(names)):
            value = delta_inner(deltas[first], deltas[names[j]])
            gram[i, j] = gram[j, i] = value
    return gram


def shared_support(deltas: dict[str, dict[str, torch.Tensor]]) -> list[str]:
    """Tensors that every delta touches."""
    return sorted(set.intersection(*(set(delta) for delta in deltas.values())))


def projection_coefficients(gram: np.ndarray, inner: np.ndarray, *,
                            rank: int | None = None, l2: float = 0.0) -> np.ndarray:
    """Coefficients of the projection of a delta onto the span of the basis.

    ``gram`` is the basis Gram matrix, ``inner`` the basis/held-out inner
    products. Eigen-directions with negligible mass are dropped; ``rank`` keeps
    only the dominant ones, ``l2`` adds ridge shrinkage.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > max(eigenvalues.max() * 1e-10, 0.0)
    if rank is not None:
        order = np.argsort(eigenvalues)[::-1][:rank]
        keep = np.zeros(len(eigenvalues), dtype=bool)
        keep[order] = True
    values = np.where(keep, eigenvalues, 0.0)
    return eigenvectors @ (np.where(keep, eigenvectors.T @ inner, 0.0) /
                           (values + l2 + 1e-30))


def combine(base_state, deltas, coefficients) -> dict[str, torch.Tensor]:
    """base + sum_i coefficient_i * delta_i, skipping zero coefficients.

    Deltas may be sparse (adapters only touch a few tensors), so keys are added
    into a copy of the base rather than zipped over the whole state.
    """
    result = dict(base_state)
    for coefficient, delta in zip(coefficients, deltas.values()):
        scale = float(coefficient)
        if scale == 0.0:
            continue
        for key, value in delta.items():
            result[key] = result[key] + scale * value
    return result


@dataclass
class LowRankResult:
    base_id: str
    tuned_ids: list[str] = field(default_factory=list)
    base_nll: dict[str, float] = field(default_factory=dict)
    gram: list[list[float]] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "base": self.base_id,
            "tuned": self.tuned_ids,
            "base_nll": self.base_nll,
            "gram": self.gram,
            "holds": self.holds,
        }


def collect_deltas(base_id: str, tuned_ids: list[str], eval_sets, *,
                   device: str = "cpu", batch_size: int = 1, progress=print):
    """Load every tuned checkpoint once, score it, and keep only its delta."""
    base = load_hf_checkpoint(base_id)
    config = Qwen2Config.from_dict(base.model_config)
    if isinstance(eval_sets, HFEvalSet):
        eval_sets = {eval_sets.name: eval_sets}

    deltas: dict[str, dict[str, torch.Tensor]] = {}
    true_nll: dict[str, dict[str, float]] = {}
    for model_id in tuned_ids:
        if is_adapter(model_id):
            delta = adapter_delta(base.state_dict, model_id)
            merged = combine(base.state_dict, {model_id: delta}, [1.0])
            true_nll[model_id] = score_state(merged, config, eval_sets,
                                             device, batch_size)
            del merged
            origin = f"adapter -> {adapter_config(model_id).get('base_model_name_or_path')}"
        else:
            checkpoint = load_hf_checkpoint(model_id)
            if arch_signature(checkpoint.model_config) != arch_signature(base.model_config):
                raise ValueError(f"{model_id} differs in architecture from {base_id}")
            true_nll[model_id] = score_state(checkpoint.state_dict, config, eval_sets,
                                             device, batch_size)
            delta = task_vector(base.state_dict, checkpoint.state_dict)
            del checkpoint
            origin = "full weights"
        deltas[model_id] = delta
        touched = len(delta)
        progress(f"  loaded {model_id.split('/')[-1]:44s} tensors={touched:3d} ({origin}) "
                 + " ".join(f"{k}={v:.4f}" for k, v in true_nll[model_id].items()))

    base_nll = score_state(base.state_dict, config, eval_sets, device, batch_size)
    progress("  base " + " ".join(f"{k}={v:.4f}" for k, v in base_nll.items()))
    return config, base, deltas, true_nll, base_nll


def leave_one_out(config, base_state, deltas, true_nll, eval_sets, *,
                  device: str = "cpu", batch_size: int = 1, ranks=(None, 2, 1),
                  l2: float = 1e-6, base_nll=None, base_id: str = "base",
                  progress=print) -> LowRankResult:
    """Reconstruct each delta from the others and score the result."""
    if isinstance(eval_sets, HFEvalSet):
        eval_sets = {eval_sets.name: eval_sets}
    names = list(deltas)
    gram = gram_matrix(deltas)
    result = LowRankResult(base_id=base_id, tuned_ids=names,
                           base_nll=base_nll or {}, gram=gram.tolist())

    for index, name in enumerate(names):
        entry = {"held_out": name, "true_nll": true_nll[name], "ranks": {}}
        others = [j for j in range(len(names)) if j != index]
        sub_gram = gram[np.ix_(others, others)]
        sub_inner = gram[index][others]
        true_norm = float(gram[index, index] ** 0.5)

        for rank in ranks:
            coefficients = projection_coefficients(sub_gram, sub_inner, rank=rank, l2=l2)
            full = np.zeros(len(names))
            for position, j in enumerate(others):
                full[j] = coefficients[position]
            state = combine(base_state, deltas, full)
            nll = score_state(state, config, eval_sets, device, batch_size)
            del state
            quadratic = float(full @ gram @ full)
            covariance = float(full @ gram[:, index])
            cosine = (covariance / (quadratic ** 0.5 * true_norm)
                      if quadratic > 0 and true_norm else 0.0)
            label = "full" if rank is None else f"rank{rank}"
            entry["ranks"][label] = {
                "nll": nll,
                "norm_ratio": (quadratic ** 0.5 / true_norm) if true_norm else float("nan"),
                "cosine": cosine,
                "coefficients": {other: float(full[j])
                                 for j, other in enumerate(names) if full[j] != 0.0},
            }
            progress(f"  hold out {name.split('/')[-1]:44s} {label:6s} "
                     f"cos={cosine:+.3f} " +
                     " ".join(f"{k}={v:.4f}" for k, v in nll.items()))
        result.holds.append(entry)
    return result


def run_lowrank(base_id: str, tuned_ids: list[str], eval_sets, *, device: str = "cpu",
                batch_size: int = 1, ranks=(None, 2, 1), l2: float = 1e-6,
                progress=print) -> LowRankResult:
    config, base, deltas, true_nll, base_nll = collect_deltas(
        base_id, tuned_ids, eval_sets, device=device, batch_size=batch_size,
        progress=progress)
    return leave_one_out(config, base.state_dict, deltas, true_nll, eval_sets,
                         device=device, batch_size=batch_size, ranks=ranks, l2=l2,
                         base_nll=base_nll, base_id=base_id, progress=progress)


def run_variance(base_id: str, cells: list[tuple[str, str, str]], *, device: str = "cpu",
                 eval_sets=None, batch_size: int = 1, progress=print) -> dict:
    """Two-way (recipe x data) decomposition of a controlled family of deltas.

    ``cells`` are ``(recipe, data, model_id)`` triples. When ``eval_sets`` is
    given each cell is also scored on the probes, which shows whether a factor
    that dominates the norm also dominates behaviour.
    """
    from .variance import (leave_one_out_from_gram, residual_cosines,
                           two_way_anova)

    base = load_hf_checkpoint(base_id)
    if eval_sets is not None and isinstance(eval_sets, HFEvalSet):
        eval_sets = {eval_sets.name: eval_sets}
    config = Qwen2Config.from_dict(base.model_config)

    deltas: dict[str, dict[str, torch.Tensor]] = {}
    nll: dict[str, dict[str, float]] = {}
    recipes, datas = [], []
    for recipe, data, model_id in cells:
        if is_adapter(model_id):
            delta = adapter_delta(base.state_dict, model_id)
        else:
            delta = task_vector(base.state_dict, load_hf_checkpoint(model_id).state_dict)
        deltas[model_id] = delta
        recipes.append(recipe)
        datas.append(data)
        line = f"  {recipe:8s} {data:10s} tensors={len(delta):3d}"
        if eval_sets:
            state = combine(base.state_dict, {model_id: delta}, [1.0])
            nll[model_id] = score_state(state, config, eval_sets, device, batch_size)
            del state
            line += " " + " ".join(f"{k}={v:.4f}" for k, v in nll[model_id].items())
        progress(line)

    gram = gram_matrix(deltas)
    norms = np.sqrt(np.diag(gram))
    cosines = gram / np.outer(norms, norms)
    # Scale-free decomposition: every delta is normalized first so the recipe
    # with the largest norm cannot dominate the sums of squares.
    normalized = gram / np.outer(norms, norms)

    shared = shared_support(deltas)
    overlap = None
    if shared:
        overlap_gram = gram_matrix(deltas, shared)
        overlap_norms = np.sqrt(np.diag(overlap_gram))
        overlap = {
            "tensors": shared,
            "gram": overlap_gram.tolist(),
            "norms": overlap_norms.tolist(),
            "cosines": (overlap_gram / np.outer(overlap_norms, overlap_norms)).tolist(),
            "anova": two_way_anova(overlap_gram, recipes, datas),
            "anova_normalized": two_way_anova(
                overlap_gram / np.outer(overlap_norms, overlap_norms), recipes, datas),
            "residual_cosines": residual_cosines(overlap_gram, recipes, datas),
        }

    result = {
        "base": base_id,
        "cells": [{"recipe": r, "data": d, "model": m} for r, d, m in cells],
        "anova": two_way_anova(gram, recipes, datas),
        "anova_normalized": two_way_anova(normalized, recipes, datas),
        "residual_cosines": residual_cosines(gram, recipes, datas),
        "leave_one_out": leave_one_out_from_gram(gram),
        "gram": gram.tolist(),
        "norms": norms.tolist(),
        "cosines": cosines.tolist(),
        "overlap": overlap,
        "nll": nll,
    }
    return result


def run_gram(base_id: str, model_ids: list[str], *, progress=print) -> dict:
    """Cosine structure of a set of deltas without any forward pass.

    Answers "do these deltas point the same way?" for full-weight checkpoints and
    adapters alike. Cheap: no model is scored, only differenced and dotted.
    """
    from .variance import leave_one_out_from_gram

    base = load_hf_checkpoint(base_id)
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for model_id in model_ids:
        if is_adapter(model_id):
            delta = adapter_delta(base.state_dict, model_id)
        else:
            delta = task_vector(base.state_dict, load_hf_checkpoint(model_id).state_dict)
        deltas[model_id] = delta
        progress(f"  {model_id:64s} tensors={len(delta):3d}")
    gram = gram_matrix(deltas)
    norms = np.sqrt(np.diag(gram))
    cosines = gram / np.outer(norms, norms)
    duplicates = [[model_ids[i], model_ids[j]]
                  for i in range(len(model_ids)) for j in range(i + 1, len(model_ids))
                  if cosines[i, j] > 0.9999 and abs(norms[i] - norms[j]) < 1e-6]
    return {
        "base": base_id,
        "models": model_ids,
        "gram": gram.tolist(),
        "norms": norms.tolist(),
        "cosines": cosines.tolist(),
        "duplicates": duplicates,
        "leave_one_out": leave_one_out_from_gram(gram),
    }


def save_variance(result: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def save_lowrank(result: LowRankResult, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return path
