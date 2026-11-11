"""Instantiate models from weight dicts and evaluate them on token streams."""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

from ..model import GPT, ModelConfig
from ..training import evaluate
from .corpus import Checkpoint
from .hf_models import Qwen2ForCausalLM, Qwen2Config, load_qwen2_state


def model_from_state(state_dict: dict[str, torch.Tensor], config: ModelConfig,
                     device: str = "cpu") -> GPT:
    """Build a scratchllm GPT from a state dict and move it to the device in eval mode."""
    model = GPT(config)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_model(checkpoint: Checkpoint, device: str = "cpu") -> GPT:
    """Build a scratchllm model from a loaded checkpoint."""
    return model_from_state(checkpoint.state_dict, checkpoint.model_config, device)


def load_hf_model(checkpoint: Checkpoint, device: str = "cpu") -> Qwen2ForCausalLM:
    """Build a Qwen2 model from a loaded HF checkpoint."""
    config = Qwen2Config.from_dict(checkpoint.model_config)
    return load_qwen2_state(checkpoint.state_dict, config, device)


def interpolate(state_a, state_b, alpha: float) -> dict[str, torch.Tensor]:
    """Linear interpolation between two compatible state dicts."""
    if state_a.keys() != state_b.keys():
        raise ValueError("state-dict keys differ")
    result = {}
    for key in state_a:
        a, b = state_a[key], state_b[key]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(a.shape)} vs {tuple(b.shape)}")
        result[key] = torch.lerp(a, b, alpha)
    return result


def task_vector(state_a, state_b) -> dict[str, torch.Tensor]:
    """The weight delta that carries b's capability relative to a."""
    if state_a.keys() != state_b.keys():
        raise ValueError("state-dict keys differ")
    return {key: state_b[key] - state_a[key] for key in state_a}


def apply_vector(state, vector, scale: float) -> dict[str, torch.Tensor]:
    """Add a scaled task vector to a state dict."""
    return {key: state[key] + scale * vector[key] for key in state}


def load_stream(path) -> np.memmap:
    """Memory-map a prepared uint32 token stream."""
    return np.memmap(Path(path), dtype="<u4", mode="r")


def evaluate_nll(model, stream, windows, batch_size: int, device: str, mask=None) -> float:
    """Token-weighted next-token NLL using the trainer's evaluation protocol."""
    return evaluate(model, stream, windows, batch_size, device, contextlib.nullcontext, mask)


@torch.inference_mode()
def evaluate_hf_nll(model: Qwen2ForCausalLM, stream, windows, batch_size: int, device: str) -> float:
    """Token-weighted next-token NLL for a causal LM over fixed windows.

    ``labels`` are the same tokens as the inputs: the model shifts internally.
    """
    model.eval()
    total, tokens = 0.0, 0
    for start in range(0, len(windows), batch_size):
        rows = windows[start:start + batch_size]
        max_len = max(n for _, n in rows)
        x = np.zeros((len(rows), max_len), dtype=np.int64)
        y = np.full_like(x, -100)
        for row, (st, n) in enumerate(rows):
            x[row, :n] = stream[st:st + n]
            y[row, :n] = stream[st:st + n]
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        _, loss, _ = model(x, y)
        count = (y[:, 1:] != -100).sum().item()
        total += loss.item() * count
        tokens += count
    return total / tokens
