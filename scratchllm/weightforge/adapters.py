"""Load PEFT/LoRA adapter checkpoints as exact weight deltas.

Several small public derivative models ship only ``adapter_model.safetensors``
instead of merged weights. Merging is cheap and, unlike reading a merged
checkpoint, gives us the delta in closed form:

    W_tuned = W_base + (alpha / r) * B @ A

so the delta has exactly rank ``r`` per target tensor. That makes adapters the
cleanest possible test bed for delta reconstruction: we know the true rank of
the thing we are trying to predict.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .corpus import local_snapshot_path


def adapter_paths(model_id: str, revision: str | None = None) -> tuple[Path, Path]:
    path = local_snapshot_path(model_id, revision)
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    if not weights.exists() or not config.exists():
        raise FileNotFoundError(f"{model_id} is not a LoRA adapter checkpoint ({path})")
    return weights, config


def is_adapter(model_id: str, revision: str | None = None) -> bool:
    try:
        adapter_paths(model_id, revision)
    except (FileNotFoundError, OSError):
        return False
    return True


def adapter_config(model_id: str, revision: str | None = None) -> dict:
    _, config = adapter_paths(model_id, revision)
    return json.loads(config.read_text())


def adapter_scale(config: dict, key: str | None = None) -> float:
    """PEFT scaling factor ``alpha / r``, honouring per-module patterns."""
    rank = config.get("r", 8)
    alpha = config.get("lora_alpha", rank)
    if key is not None:
        for pattern, value in (config.get("rank_pattern") or {}).items():
            if pattern in key:
                rank = value
                break
        for pattern, value in (config.get("alpha_pattern") or {}).items():
            if pattern in key:
                alpha = value
                break
    return float(alpha) / float(rank)


def _normalize_key(key: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def adapter_delta(base_state: dict[str, torch.Tensor], model_id: str,
                  revision: str | None = None) -> dict[str, torch.Tensor]:
    """Delta to add to ``base_state`` so it becomes the adapter-merged model.

    Tensors the adapter does not touch are omitted, so the delta is sparse and
    cheap to keep in memory: a rank-1 adapter on one matrix costs a few MB.
    """
    from safetensors import safe_open

    weights, _ = adapter_paths(model_id, revision)
    config = adapter_config(model_id, revision)
    delta: dict[str, torch.Tensor] = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}

    with safe_open(weights, framework="pt", device="cpu") as handle:
        for raw_key in handle.keys():
            key = _normalize_key(raw_key)
            if key.endswith(".lora_A.weight"):
                pairs.setdefault(key[: -len(".lora_A.weight")], {})["A"] = handle.get_tensor(raw_key)
            elif key.endswith(".lora_B.weight"):
                pairs.setdefault(key[: -len(".lora_B.weight")], {})["B"] = handle.get_tensor(raw_key)

    for target, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            raise ValueError(f"adapter {model_id} has an incomplete pair for {target}")
        weight_key = f"{target}.weight" if not target.endswith(".weight") else target
        if weight_key not in base_state:
            raise KeyError(f"adapter target {weight_key} is not in the base state")
        update = (parts["B"].float() @ parts["A"].float()) * adapter_scale(config, target)
        weight = base_state[weight_key]
        if update.shape != weight.shape:
            raise ValueError(f"shape mismatch for {weight_key}: {tuple(update.shape)} "
                             f"vs {tuple(weight.shape)}")
        delta[weight_key] = update.to(weight.dtype)
    return delta
