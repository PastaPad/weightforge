"""Prepare synthetic tasks and train a controlled corpus of small models.

The corpus is deliberately local and license-clean: every checkpoint is trained
from scratch on a prepared dataset, and the manifest records the exact
architecture, task, seed, and validation score for each model.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .corpus import load_checkpoint
from .tasks import generate


@dataclass
class CorpusEntry:
    """One trained model in the corpus."""

    task: str
    size: str
    seed: int
    path: str
    model_config: dict
    parameters: int
    step: int | None
    best_val: float | None
    parent: str | None = None


def _run(command) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def prepare_task(text_path, output_dir, vocab_size: int = 512, min_chars: int = 1) -> Path:
    """Tokenize a generated task text file into a base training dataset."""
    output_dir = Path(output_dir)
    if (output_dir / "meta.json").exists():
        print(f"skip existing dataset {output_dir}", flush=True)
        return output_dir
    _run([sys.executable, "-m", "scratchllm.prepare", "--input", text_path,
          "--output", output_dir, "--vocab-size", vocab_size,
          "--min-chars", min_chars, "--min-post-chars", min_chars, "--atomic-output"])
    return output_dir


def write_training_config(path, size: dict, seed: int, steps: int) -> Path:
    """Write a small modern-architecture training config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "modern",
        "context": size["context"],
        "width": size["width"],
        "layers": size["layers"],
        "heads": size["heads"],
        "kv_heads": size["kv_heads"],
        "ffn_width": size["ffn_width"],
        "dropout": 0.1,
        "batch_size": 4,
        "accumulation": 4,
        "budget_mode": "updates",
        "steps": steps,
        "learning_rate": 0.0003,
        "min_learning_rate": 0.00003,
        "warmup_steps": max(1, steps // 10),
        "weight_decay": 0.1,
        "eval_every": max(1, steps // 4),
        "eval_batches": 8,
        "seed": seed,
        "sampling": "shuffled",
        "max_epochs": 0,
        "early_stopping_patience": 0,
        "label_smoothing": 0,
        "ema_decay": 0,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def train_model(data_dir, config_path, out_dir, device: str = "cuda",
                init_from=None) -> Path:
    """Train one model and return its best checkpoint path."""
    out_dir = Path(out_dir)
    command = [sys.executable, "-m", "scratchllm.train", "--data", data_dir,
               "--config", config_path, "--out", out_dir, "--device", device]
    if init_from:
        command += ["--init-from", init_from]
    _run(command)
    return out_dir / "best.pt"


def build_corpus(config_path) -> list[CorpusEntry]:
    """Prepare tasks, train every (task, size, seed), and write a manifest."""
    config = json.loads(Path(config_path).read_text())
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=True)
    device = config.get("device", "cuda")
    steps = config.get("training", {}).get("steps", 200)
    seeds = config.get("seeds", [1])
    entries = []

    for task in config["tasks"]:
        task_id = task["id"]
        if task.get("generate"):
            text_path = root / "tasks" / f"{task_id}.txt"
            if not text_path.exists():
                generate(task_id, text_path, count=task.get("count", 20000),
                         seed=task.get("seed", 1234))
            data_dir = prepare_task(text_path, root / "prepared" / task_id,
                                    vocab_size=task.get("vocab_size", 512))
        else:
            data_dir = Path(task["data"])
        for size in config["sizes"]:
            for seed in seeds:
                out_dir = root / "models" / task_id / size["name"] / f"seed{seed}"
                checkpoint_path = out_dir / "best.pt"
                if not checkpoint_path.exists():
                    config_file = root / "configs" / f"{task_id}-{size['name']}-seed{seed}.json"
                    write_training_config(config_file, size, seed, steps)
                    train_model(data_dir, config_file, out_dir, device=device)
                else:
                    print(f"skip existing model {out_dir}", flush=True)
                checkpoint = load_checkpoint(checkpoint_path)
                entries.append(CorpusEntry(
                    task=task_id, size=size["name"], seed=seed,
                    path=str(checkpoint_path),
                    model_config=checkpoint.model_config.to_dict(),
                    parameters=checkpoint.num_parameters,
                    step=checkpoint.step, best_val=checkpoint.best_val))

    manifest = root / "manifest.json"
    manifest.write_text(json.dumps([asdict(entry) for entry in entries], indent=2) + "\n")
    print(f"wrote {manifest} with {len(entries)} models", flush=True)
    return entries
