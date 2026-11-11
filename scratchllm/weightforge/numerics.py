"""Numerically stable reductions over very large weight tensors.

Float32 accumulation over 100M+ bf16 elements loses enough precision to break
the diagnostics: the Qwen2 embedding (136M values) reported a cosine of 1.033
between two *bit-identical* tensors, and norms were off by 1.6%. Chunked float64
accumulation costs a few hundred MB of scratch memory and removes the problem.
"""
from __future__ import annotations

import torch

_CHUNK = 1 << 22  # 4M elements -> 32 MB in float64


def stable_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Inner product of two flattened tensors with float64 accumulation."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"size mismatch: {a.numel()} vs {b.numel()}")
    total = 0.0
    for start in range(0, a.numel(), _CHUNK):
        stop = start + _CHUNK
        total += float(a[start:stop].double() @ b[start:stop].double())
    return total


def stable_norm(a: torch.Tensor) -> float:
    """L2 norm with float64 accumulation."""
    return stable_dot(a, a) ** 0.5


def stable_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = stable_dot(a, b)
    denominator = stable_dot(a, a) ** 0.5 * stable_dot(b, b) ** 0.5
    return dot / denominator if denominator else float("nan")


def stable_relative_delta(base: torch.Tensor, delta: torch.Tensor) -> float:
    denominator = stable_norm(base)
    return stable_norm(delta) / denominator if denominator else float("nan")
