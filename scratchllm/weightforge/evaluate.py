"""Weight-space analysis: probes, interpolation curves, and editing metrics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..training import fixed_windows
from .corpus import Checkpoint
from .functional import (apply_vector, evaluate_nll, interpolate, load_stream,
                         model_from_state, task_vector)


@dataclass
class EvalSet:
    """A token stream plus the windows (and optional loss mask) to score."""

    name: str
    stream: np.memmap
    windows: list[tuple[int, int]]
    mask: np.memmap | None = None


def load_eval_set(name, directory, *, context: int, split: str = "val", dialogue: bool = False,
                  seed: int = 0, batch_size: int = 4, eval_batches: int = 8,
                  max_windows: int | None = None) -> EvalSet:
    """Load a prepared dataset split for scoring.

    Dialogue splits use their conversation windows and assistant-only loss mask;
    plain text splits use fixed non-overlapping windows.
    """
    directory = Path(directory)
    stream = load_stream(directory / f"{split}.bin")
    if dialogue:
        windows = [tuple(row) for row in
                   json.loads((directory / f"{split}.windows.json").read_text())]
        mask = np.memmap(directory / f"{split}.mask.bin", dtype="u1", mode="r")
        if max_windows and len(windows) > max_windows:
            indices = np.random.default_rng(seed).choice(len(windows), max_windows, replace=False)
            windows = [windows[index] for index in sorted(indices)]
    else:
        windows = fixed_windows(len(stream), context, batch_size * eval_batches, seed)
        mask = None
    return EvalSet(name=name, stream=stream, windows=windows, mask=mask)


def interpolation_curve(checkpoint_a: Checkpoint, checkpoint_b: Checkpoint, alphas,
                        eval_sets, device: str = "cuda", batch_size: int = 4) -> list[dict]:
    """Score linear weight mixtures for each alpha."""
    rows = []
    for alpha in alphas:
        state = interpolate(checkpoint_a.state_dict, checkpoint_b.state_dict, alpha)
        rows.append(_score_state(state, checkpoint_a.model_config, eval_sets, device,
                                 batch_size, alpha=alpha))
    return rows


def task_vector_curve(checkpoint_a: Checkpoint, checkpoint_b: Checkpoint, scales,
                      eval_sets, device: str = "cuda", batch_size: int = 4) -> list[dict]:
    """Score base + scale * (task - base) for each scale."""
    vector = task_vector(checkpoint_a.state_dict, checkpoint_b.state_dict)
    rows = []
    for scale in scales:
        state = apply_vector(checkpoint_a.state_dict, vector, scale)
        rows.append(_score_state(state, checkpoint_a.model_config, eval_sets, device,
                                 batch_size, scale=scale))
    return rows


def _score_state(state, config, eval_sets, device, batch_size, **extra) -> dict:
    model = model_from_state(state, config, device)
    row = dict(extra)
    try:
        for eval_set in eval_sets:
            row[eval_set.name] = evaluate_nll(model, eval_set.stream, eval_set.windows,
                                              batch_size, device, eval_set.mask)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def weight_features(state_dict) -> np.ndarray:
    """Per-tensor statistics in state-dict key order."""
    features = []
    for key in sorted(state_dict):
        tensor = state_dict[key].float().reshape(-1)
        features.extend([
            tensor.mean().item(),
            tensor.std(unbiased=False).item(),
            tensor.min().item(),
            tensor.max().item(),
            tensor.norm().item(),
            tensor.abs().mean().item(),
        ])
    return np.asarray(features, dtype=np.float64)


def scalar_statistics(checkpoint_a: Checkpoint, checkpoint_b: Checkpoint,
                      alphas) -> dict[str, list[float]]:
    """Interpretable scalars along the interpolation path."""
    base = checkpoint_a.state_dict
    base_flat = torch.cat([base[key].reshape(-1).float() for key in sorted(base)])
    base_norm = base_flat.norm().item()
    stats = {"delta_norm": [], "relative_delta_norm": [], "cosine": []}
    for alpha in alphas:
        state = interpolate(base, checkpoint_b.state_dict, alpha)
        flat = torch.cat([state[key].reshape(-1).float() for key in sorted(state)])
        delta = flat - base_flat
        stats["delta_norm"].append(delta.norm().item())
        stats["relative_delta_norm"].append(delta.norm().item() / base_norm)
        stats["cosine"].append(
            torch.nn.functional.cosine_similarity(flat, base_flat, dim=0).item())
    return stats


def pearson(x, y) -> float:
    """Pearson correlation; NaN when undefined."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ridge_loo_r2(features, targets, l2: float = 10.0) -> tuple[float, np.ndarray]:
    """Leave-one-out R^2 for ridge regression with an unpenalized intercept."""
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if len(y) < 3:
        return float("nan"), np.full_like(y, np.nan)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    n, d = Z.shape
    penalty = l2 * np.eye(d)
    penalty[0, 0] = 0.0
    hat = Z @ np.linalg.solve(Z.T @ Z + penalty, Z.T)
    predicted = hat @ y
    diagonal = np.diag(hat)
    loo = (predicted - diagonal * y) / np.maximum(1.0 - diagonal, 1e-9)
    ss_res = float(np.sum((y - loo) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, loo
