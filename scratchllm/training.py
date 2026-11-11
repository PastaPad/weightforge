"""Data-sized schedules, repeatable NLL evaluation, and resumable regularization."""
import contextlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import model_config, parameter_count


def training_plan(cfg, meta):
    mode = cfg.get("budget_mode", "updates")
    if mode not in {"updates", "passes"}:
        raise ValueError("budget_mode must be updates or passes")
    for key in ("batch_size", "accumulation", "steps", "eval_every", "eval_batches"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["warmup_steps"] < 0 or (mode == "updates" and cfg["warmup_steps"] >= cfg["steps"]):
        raise ValueError("warmup_steps must be nonnegative and below the fixed update limit")
    for key in ("learning_rate", "min_learning_rate", "weight_decay", "max_epochs",
                "early_stopping_min_delta", "label_smoothing", "ema_decay"):
        value = cfg.get(key, 0)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < cfg["learning_rate"] or cfg["min_learning_rate"] > cfg["learning_rate"]:
        raise ValueError("Require 0 < learning rate and minimum LR <= learning rate")
    if cfg.get("label_smoothing", 0) >= 1 or cfg.get("ema_decay", 0) >= 1:
        raise ValueError("label_smoothing and ema_decay must be in [0, 1)")
    patience = cfg.get("early_stopping_patience", 0)
    if not isinstance(patience, int) or patience < 0:
        raise ValueError("early_stopping_patience must be a nonnegative integer")
    if cfg.get("sampling", "random") not in {"random", "shuffled"}:
        raise ValueError("sampling must be random or shuffled")
    if mode == "passes" and cfg.get("max_epochs", 0) <= 0:
        raise ValueError("Data-pass mode requires a positive number of passes")
    if cfg.get('precision', 'auto') not in {'auto', 'fp32', 'fp16', 'bf16'}:
        raise ValueError('Unknown precision')
    if (type(cfg.get('gradient_checkpointing', False)) is not bool
            or (cfg.get('optimizer_foreach') is not None and type(cfg['optimizer_foreach']) is not bool)):
        raise ValueError('Invalid checkpointing/optimizer setting')
    model = model_config(meta['vocab_size'], cfg)
    dialogue = meta.get('format') == 'dialogue-sft-v1'
    custom_text = meta.get('format') == 'text-finetune-v1'
    if custom_text and (model.context != meta['base_model_config']['context'] or cfg.get('sampling') != 'shuffled'):
        raise ValueError('Custom text fine-tuning requires its base context and shuffled token sampling')
    if dialogue and (model.context != meta['base_model_config']['context'] or cfg.get('sampling') != 'shuffled'):
        raise ValueError('Dialogue training requires its base context and shuffled example sampling')
    for split in ("train", "val"):
        if dialogue:
            if meta['documents'][split] < 1 or meta['supervised_tokens'][split] < 1:
                raise ValueError(f'{split} needs dialogue examples with assistant targets')
        elif custom_text and meta['tokens'][split] <= 1:
            raise ValueError(f'{split} needs at least two custom text tokens')
        elif not custom_text and meta["tokens"][split] <= model.context:
            raise ValueError(f"{split} needs more than {model.context} tokens")
    positions = meta["tokens"]["train"] - 1
    tokens_per_step = cfg["batch_size"] * cfg["accumulation"] * model.context
    budget_size = meta['documents']['train'] if dialogue else positions
    budget_per_step = cfg['batch_size'] * cfg['accumulation'] if dialogue else tokens_per_step
    requested_steps = cfg["steps"]
    if mode == "passes":
        # In this mode, an old fixed step count must not constrain a larger dataset.
        requested_steps = max(1, math.ceil(cfg["max_epochs"] * budget_size / budget_per_step))
        if requested_steps > 10000000:
            raise ValueError("Data-pass budget exceeds 10,000,000 updates; choose fewer passes")
    steps = requested_steps
    if mode == "updates" and cfg.get("max_epochs", 0):
        steps = min(steps, max(1, math.ceil(cfg["max_epochs"] * budget_size / budget_per_step)))
    warmup = cfg["warmup_steps"]
    if mode == "passes" or steps < cfg["steps"]:
        warmup = min(warmup, steps // 10)
    parameters = parameter_count(model)
    return dict(requested_steps=requested_steps, steps=steps, warmup_steps=warmup,
                eval_every=min(cfg["eval_every"], max(1, math.ceil(budget_size / budget_per_step))),
                tokens_per_step=tokens_per_step, train_tokens=meta["tokens"]["train"],
                train_documents=meta.get("documents", {}).get("train"),
                validation_documents=meta.get("documents", {}).get("val"),
                requested_epochs=requested_steps * budget_per_step / budget_size,
                planned_epochs=steps * budget_per_step / budget_size,
                parameters=parameters, unique_tokens_per_parameter=positions / parameters,
                sampling=cfg.get("sampling", "random"))


def learning_rate(step, cfg, plan):
    warmup, steps = plan["warmup_steps"], plan["steps"]
    if step < warmup:
        return cfg["learning_rate"] * (step + 1) / max(1, warmup)
    progress = min(1, max(0, (step - warmup) / max(1, steps - warmup - 1)))
    return cfg["min_learning_rate"] + 0.5 * (cfg["learning_rate"] - cfg["min_learning_rate"]) * (1 + math.cos(math.pi * progress))


def fixed_windows(length, context, limit, seed):
    """Nonoverlapping targets; include the partial tail, or a fixed subset of blocks."""
    count = math.ceil((length - 1) / context)
    indices = np.arange(count) if count <= limit else np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))
    return [(int(i * context), min(context, length - 1 - int(i * context))) for i in indices]


def target_count(windows, mask=None):
    return sum(n if mask is None else int(mask[start + 1:start + n + 1].sum()) for start, n in windows)


def evaluation_plan(streams, cfg, dialogue_windows=None, masks=None):
    windows = {}
    for i, (split, stream) in enumerate(streams.items()):
        limit, seed = cfg['batch_size'] * cfg['eval_batches'], cfg['seed'] + i + 1
        if dialogue_windows is None:
            windows[split] = fixed_windows(len(stream), cfg['context'], limit, seed)
        else:
            rows = dialogue_windows[split]
            indices = range(len(rows)) if len(rows) <= limit else sorted(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
            windows[split] = [rows[index] for index in indices]
    tokens = {split: target_count(rows, masks[split] if masks else None) for split, rows in windows.items()}
    return {"protocol": "fixed-assistant-nll-v1" if masks else "fixed-token-nll-v1", "windows": windows,
            "tokens": tokens,
            "coverage": {split: tokens[split] / (int(masks[split].sum()) if masks else len(streams[split]) - 1) for split in streams}}


def make_batch(stream, windows, device, mask=None):
    length = max(n for _, n in windows)
    x = np.zeros((len(windows), length), dtype=np.int64)
    y = np.full_like(x, -100)
    for row, (start, size) in enumerate(windows):
        x[row, :size] = stream[start:start + size]
        y[row, :size] = stream[start + 1:start + size + 1]
        if mask is not None:
            y[row, :size] = np.where(mask[start + 1:start + size + 1], y[row, :size], -100)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.inference_mode()
def evaluate(model, stream, windows, batch_size, device, autocast=contextlib.nullcontext, mask=None):
    """Unsmoothed token-weighted NLL, with dropout off and no training RNG use."""
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        for start in range(0, len(windows), batch_size):
            rows = windows[start:start + batch_size]
            count = target_count(rows, mask)
            with autocast():
                _, loss = model(*make_batch(stream, rows, device, mask))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite evaluation loss")
            total += loss.item() * count
            tokens += count
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("Evaluation needs at least one target token")
    return total / tokens


class ShuffledStream:
    """Visit each target once per pass; rotate chunk boundaries between passes."""
    def __init__(self, length, context, seed):
        self.length, self.context = length, context
        self.rng = np.random.default_rng(seed)
        self.windows, self.cursor, self.passes = [], 0, 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == len(self.windows):
                offset = int(self.rng.integers(1, self.context + 1))
                boundaries = [0, *range(offset, self.length - 1, self.context), self.length - 1]
                windows = [(a, b - a) for a, b in zip(boundaries, boundaries[1:])]
                self.windows = [windows[i] for i in self.rng.permutation(len(windows))]
                self.cursor = 0
                self.passes += 1
            n = min(count - len(result), len(self.windows) - self.cursor)
            result.extend(self.windows[self.cursor:self.cursor + n])
            self.cursor += n
        return result

    def state_dict(self):
        return dict(length=self.length, context=self.context, windows=self.windows, cursor=self.cursor,
                    passes=self.passes, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if (state["length"], state["context"]) != (self.length, self.context):
            raise ValueError("Sampler dimensions changed")
        self.windows, self.cursor, self.passes = state["windows"], state["cursor"], state["passes"]
        self.rng.bit_generator.state = state["rng"]


@dataclass
class EarlyStopping:
    best: float = float("inf")
    reference: float = float("inf")
    best_step: int = 0
    bad_evals: int = 0

    def observe(self, loss, step, min_delta, patience):
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        improved = loss < self.best
        if improved:
            self.best, self.best_step = loss, step
        # Anchor to the last meaningful improvement so several small gains can accumulate.
        if loss < self.reference - min_delta:
            self.reference, self.bad_evals = loss, 0
        else:
            self.bad_evals += 1
        return improved, bool(patience and self.bad_evals >= patience)

    def state_dict(self):
        return asdict(self)


class WeightEMA:
    def __init__(self, model, decay):
        self.decay, self.updates = decay, 0
        # named_parameters deduplicates the tied embedding/output matrix.
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Start with the first updated weights, then a capped cumulative average.
        decay = min(self.decay, (self.updates - 1) / self.updates)
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.detach(), 1 - decay)

    @contextlib.contextmanager
    def apply(self, model):
        with torch.no_grad():
            saved = {name: p.detach().clone() for name, p in model.named_parameters()}
            try:
                for name, p in model.named_parameters():
                    p.copy_(self.shadow[name])
                yield
            finally:
                for name, p in model.named_parameters():
                    p.copy_(saved[name])

    def state_dict(self):
        return dict(decay=self.decay, updates=self.updates, shadow=self.shadow)

    def load_state_dict(self, state):
        if state["decay"] != self.decay or state["shadow"].keys() != self.shadow.keys():
            raise ValueError("EMA configuration changed")
        self.updates = state["updates"]
        for name, p in self.shadow.items():
            p.copy_(state["shadow"][name])
