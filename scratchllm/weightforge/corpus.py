"""Checkpoint discovery and weight-space tensor handling (scratchllm + HF)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Checkpoint:
    """A loaded model checkpoint with weights kept on CPU."""

    path: Path
    model_config: dict
    state_dict: dict[str, torch.Tensor]
    tokenizer_json: str | None = None
    step: int | None = None
    best_val: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_parameters(self) -> int:
        return sum(tensor.numel() for tensor in self.state_dict.values())

    @property
    def name(self) -> str:
        return self.path.parent.name or self.path.stem


def load_checkpoint(path, map_location: str = "cpu") -> Checkpoint:
    """Load a scratchllm best.pt/last.pt checkpoint without optimizer state."""
    path = Path(path)
    state = torch.load(path, map_location=map_location, weights_only=True)
    if "model_config" not in state or "model" not in state:
        raise ValueError(f"{path} is not a scratchllm checkpoint")
    return Checkpoint(
        path=path,
        model_config=state["model_config"],
        state_dict=state["model"],
        tokenizer_json=state.get("tokenizer_json"),
        step=state.get("step"),
        best_val=state.get("best_val"),
        metadata={key: value for key, value in state.items()
                  if key not in {"model", "model_config", "tokenizer_json"}},
    )


def local_snapshot_path(model_id: str, revision: str | None = None) -> Path:
    """Resolve a downloaded snapshot directory without HF's completeness check.

    ``snapshot_download(local_files_only=True)`` refuses a repo whose snapshot is
    missing files we deliberately excluded (GGUF/ONNX siblings), which is common
    for community repos. Reading the ref directly is enough for our use.
    """
    from huggingface_hub import constants

    root = Path(constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    ref = root / "refs" / (revision or "main")
    if ref.exists():
        path = root / "snapshots" / ref.read_text().strip()
        if path.is_dir():
            return path
    snapshots = sorted((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
    for path in reversed(snapshots):
        if (path / "config.json").exists():
            return path
    stubs = f"hf://{model_id}"
    raise FileNotFoundError(f"{stubs} is not downloaded ({root})")


def load_hf_checkpoint(model_id: str, revision: str | None = None,
                       map_location: str = "cpu") -> Checkpoint:
    """Load a Hugging Face model from the local cache (must be downloaded)."""
    from safetensors import safe_open

    path = local_snapshot_path(model_id, revision)

    with open(path / "config.json") as f:
        cfg = json.load(f)

    weights = path / "model.safetensors"
    if not weights.exists():
        shards = sorted(path.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no safetensors weights in {path}")
        state_dict = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device=map_location) as f:
                state_dict.update({k: f.get_tensor(k) for k in f.keys()})
    else:
        with safe_open(weights, framework="pt", device=map_location) as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

    tokenizer_json = ((path / "tokenizer.json").read_text(encoding="utf-8")
                      if (path / "tokenizer.json").exists() else None)

    return Checkpoint(
        path=path,
        model_config=cfg,
        state_dict=state_dict,
        tokenizer_json=tokenizer_json,
        metadata={"model_id": model_id, "revision": revision},
    )


def discover(root, pattern: str = "best.pt") -> list[Checkpoint]:
    """Load every matching checkpoint under a directory, sorted by path."""
    return [load_checkpoint(path) for path in sorted(Path(root).rglob(pattern))]


def assert_compatible(first: Checkpoint, second: Checkpoint) -> None:
    """Require identical architecture and state-dict structure for weight arithmetic."""
    if first.model_config != second.model_config:
        raise ValueError("checkpoints have different architectures")
    if first.state_dict.keys() != second.state_dict.keys():
        raise ValueError("checkpoints have different state-dict structure")
    for key in first.state_dict:
        if first.state_dict[key].shape != second.state_dict[key].shape:
            raise ValueError(f"shape mismatch for {key}")
