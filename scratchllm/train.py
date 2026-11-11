"""CUDA training with accumulation, validation, atomic checkpoints, and resume."""
import argparse
import contextlib
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from .model import GPT, model_config, inherit_architecture
from .links import TEXT_POLICY_VERSION
from .dialogue import FORMAT, TEXT_FORMAT, CHAT_FORMAT
from .dialogue_training import ShuffledExamples, initialize_from, load_dialogue, finetuning_provenance
from .training import (EarlyStopping, ShuffledStream, WeightEMA, evaluate, evaluation_plan,
                       learning_rate, make_batch, training_plan, target_count)


def atomic_save(state, path):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def save_checkpoints(out, state, model, ema, selected, improved):
    # Optimizer/RNG resume always uses raw weights, even when EMA wins validation.
    atomic_save(state, out / "last.pt")
    if improved:
        with ema.apply(model) if selected == "ema" else contextlib.nullcontext():
            inference = {k: state[k] for k in ("model_config", "tokenizer_json", "step", "best_val", "data_hashes", "evaluation")}
            inference.update(model=model.state_dict(), selected_weights=selected)
            inference.update({key: state[key] for key in ('chat_format', 'base_checkpoint', 'base_checkpoint_sha256',
                                                        'dialogue_sources', 'finetuning_format', 'custom_corpus') if key in state})
            atomic_save(inference, out / "best.pt")


def main():
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--config", default="configs/response.json")
    p.add_argument("--out", default="runs/tiny")
    p.add_argument("--resume", action="store_true")
    p.add_argument('--init-from', help='Initialize fine-tuning from your selected best.pt; starts a fresh optimizer/schedule')
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], help='Override configuration precision (otherwise auto)')
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error('Resume loads last.pt; do not also supply --init-from')
    cfg = json.loads(Path(args.config).read_text())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Install a compatible CUDA PyTorch build or use --device cpu for a smoke test.")
    precision = args.precision or cfg.get('precision', 'auto')
    if precision not in {'auto', 'fp32', 'fp16', 'bf16'}:
        p.error('Unknown precision in configuration')
    if precision == "auto":
        # Pascal GPUs work well in fp32; newer tensor-core GPUs benefit from AMP.
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else (
            "fp16" if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7 else "fp32")
    if precision != "fp32" and device.type != "cuda":
        p.error("mixed precision requires CUDA in this trainer")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    def autocast():
        return torch.autocast("cuda", dtype=dtype) if precision != "fp32" else contextlib.nullcontext()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        p.error("output contains a checkpoint; use --resume or a fresh --out")
    tokenizer_json = (data / "tokenizer.json").read_text()
    meta = json.loads((data / "meta.json").read_text())
    dialogue = meta.get('format') == FORMAT
    custom_text = meta.get('format') == TEXT_FORMAT
    fine_tuning = dialogue or custom_text
    if fine_tuning and not (args.resume or args.init_from):
        p.error('Fine-tuning requires --init-from your selected best.pt (or --resume)')
    if args.init_from and not fine_tuning:
        p.error('--init-from requires a prepared dialogue or custom text fine-tuning dataset')
    if fine_tuning:
        cfg = inherit_architecture(cfg, meta['base_model_config'])
    if not args.resume and meta.get('text_policy') != TEXT_POLICY_VERSION:
        p.error('Prepare the corpus again into a fresh dataset to apply URL removal before a new run')
    try:
        plan = training_plan(cfg, meta)
    except ValueError as exc:
        p.error(str(exc))
    if hashlib.sha256(tokenizer_json.encode()).hexdigest() != meta["tokenizer_sha256"]:
        raise SystemExit("Tokenizer doesn't match prepared data")
    streams = {split: np.memmap(data / f"{split}.bin", dtype="<u4", mode="r") for split in ("train", "val")}
    for split, stream in streams.items():
        if not fine_tuning and len(stream) <= cfg["context"]:
            raise SystemExit(f"{split} needs more than {cfg['context']} tokens")
        if len(stream) != meta["tokens"][split]:
            raise SystemExit(f"{split} token file size doesn't match metadata")
    # Fingerprint streams to prevent resuming with a different corpus.
    def fingerprint(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    data_hashes = {split: fingerprint(data / f"{split}.bin") for split in streams}
    masks, dialogue_windows = load_dialogue(data, meta, streams, cfg['context']) if dialogue else ({}, None)
    if dialogue:
        data_hashes.update({f'{split}.{suffix}': fingerprint(data/f'{split}.{suffix}')
                             for split in streams for suffix in ('mask.bin', 'windows.json')})
    if fine_tuning:
        data_hashes['meta'] = fingerprint(data/'meta.json')
    rng = np.random.default_rng(cfg["seed"])
    eval_rng = np.random.default_rng(cfg["seed"] + 1)
    def random_batch(split, generator):
        stream = streams[split]
        starts = generator.integers(0, len(stream) - cfg["context"], size=cfg["batch_size"])
        chunk = np.stack([stream[i:i + cfg["context"] + 1] for i in starts]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1].copy()).to(device)
        y = torch.from_numpy(chunk[:, 1:].copy()).to(device)
        return x, y
    sampler = (ShuffledExamples(dialogue_windows['train'], cfg['seed']) if dialogue else
               ShuffledStream(len(streams["train"]), cfg["context"], cfg["seed"]) if cfg.get("sampling") == "shuffled" else None)
    evaluation = evaluation_plan(streams, cfg, dialogue_windows, masks)
    fixed_evaluation = evaluation
    model_cfg = model_config(meta['vocab_size'], cfg)
    model = GPT(model_cfg).to(device)
    model.gradient_checkpointing = cfg.get('gradient_checkpointing', False)
    if args.init_from:
        initialize_from(model, args.init_from, meta, tokenizer_json)
        print(f'INITIALIZED from {args.init_from}; exact tokenizer and base weights retained; '
              + ('assistant-only loss' if dialogue else 'all-text next-token loss'), flush=True)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=cfg["learning_rate"], betas=(0.9, 0.95),
                                  foreach=cfg.get('optimizer_foreach'))
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    ema = WeightEMA(model, cfg["ema_decay"]) if cfg.get("ema_decay", 0) else None
    early = EarlyStopping()
    step, tokens_seen, optimizer_updates, stop_reason = 0, 0, 0, None
    examples_seen = 0
    provenance = finetuning_provenance(meta)
    legacy = False
    if args.resume:
        # Checkpoints contain RNG/optimizer state. Only load your own trusted files.
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if state["training_config"] != cfg or state["model_config"] != model.config_dict() or state["precision"] != precision:
            raise SystemExit("Resume requires the same training config, model, and precision")
        if state["tokenizer_json"] != tokenizer_json or state["data_hashes"] != data_hashes:
            raise SystemExit("Resume corpus/tokenizer mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        tokens_seen = state.get("tokens_seen", step * plan["tokens_per_step"])
        optimizer_updates = state.get("optimizer_updates", step)
        examples_seen = state.get('examples_seen', 0)
        if dialogue and state.get('chat_format') != CHAT_FORMAT:
            raise SystemExit('Resume dialogue format mismatch')
        if custom_text and (state.get('finetuning_format') != TEXT_FORMAT or state.get('chat_format') != meta.get('chat_format')):
            raise SystemExit('Resume custom text format mismatch')
        stop_reason = state.get("stop_reason")
        early = EarlyStopping(**state["early_stopping"]) if "early_stopping" in state else EarlyStopping(best=state["best_val"], reference=state["best_val"])
        if "early_stopping" not in state and (out / "best.pt").is_file():
            early.best_step = torch.load(out / "best.pt", map_location="cpu", weights_only=True)["step"]
        legacy = state.get("evaluation", {}).get("protocol", "legacy-random-v1") == "legacy-random-v1"
        if legacy:
            # Preserve the original validation RNG/best score on old interrupted runs.
            plan.update(eval_every=cfg["eval_every"])
            evaluation = {"protocol": "legacy-random-v1"}
            print("Resuming legacy random-window validation; new runs use fixed token-weighted NLL.", flush=True)
        elif state["evaluation"] != evaluation or state["plan"] != plan:
            raise SystemExit("Resume evaluation/schedule mismatch")
        if sampler:
            sampler.load_state_dict(state["sampler"])
        if ema:
            ema.load_state_dict(state["ema"])
        rng.bit_generator.state = state["rng"]
        eval_rng.bit_generator.state = state["eval_rng"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"{plan['parameters']:,} parameters; {device}; {precision}", flush=True)
    print(f"MODEL {model.cfg.architecture}; activation checkpointing={model.gradient_checkpointing}; "
          f"AdamW foreach={cfg.get('optimizer_foreach')}", flush=True)
    print(f"DATA {plan['train_tokens']:,} train tokens in {plan['train_documents']} documents; "
          f"{plan['tokens_per_step']:,} maximum tokens/update; {plan['requested_epochs']:.1f} requested passes", flush=True)
    budget = f"{cfg['max_epochs']:g} data passes" if cfg.get("budget_mode") == "passes" else f"{cfg['steps']} requested updates"
    pass_unit = 'dialogue-example' if dialogue else 'token-equivalent'
    print(f"PLAN {plan['steps']} updates from {budget}; at most {plan['planned_epochs']:.2f} {pass_unit} passes; "
          f"warmup {plan['warmup_steps']}; evaluate every {plan['eval_every']}; sampling={plan['sampling']}", flush=True)
    coverage = ("legacy random validation windows" if legacy else
                f"fixed validation covers {fixed_evaluation['tokens']['val']:,} targets ({fixed_evaluation['coverage']['val']:.1%})")
    print(f"EVAL {evaluation['protocol']}; {coverage}; patience={cfg.get('early_stopping_patience', 0)}; "
          f"EMA={cfg.get('ema_decay', 0)}; label smoothing={cfg.get('label_smoothing', 0)}", flush=True)
    if stop_reason == "early_stopping" or step >= plan["steps"]:
        print(f"Run already finished ({stop_reason or 'step_limit'}); best.pt is the selected inference checkpoint.", flush=True)
        return
    (out / "plan.json").write_text(json.dumps(dict(plan, evaluation=evaluation), indent=2) + "\n")
    start = time.monotonic()
    while step < plan["steps"]:
        lr = learning_rate(step, cfg, plan)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        windows = sampler.take(cfg["accumulation"] * cfg["batch_size"]) if sampler else None
        step_tokens = target_count(windows, masks.get('train')) if windows else plan["tokens_per_step"]
        for micro in range(cfg["accumulation"]):
            if windows:
                rows = windows[micro * cfg["batch_size"]:(micro + 1) * cfg["batch_size"]]
                x, y = make_batch(streams["train"], rows, device, masks.get('train'))
                weight = target_count(rows, masks.get('train')) / step_tokens
            else:
                x, y = random_batch("train", rng)
                weight = 1 / cfg["accumulation"]
            with autocast():
                _, loss = model(x, y, label_smoothing=cfg.get("label_smoothing", 0))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; try fp32 or lower the learning rate")
            total_loss += loss.detach().item() * weight
            scaler.scale(loss * weight).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if precision != "fp16" and not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradients")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            optimizer_updates += 1
            if ema:
                ema.update(model)
        step += 1
        tokens_seen += step_tokens
        if dialogue:
            examples_seen += len(windows)
        epochs = examples_seen / meta['documents']['train'] if dialogue else tokens_seen / (len(streams["train"]) - 1)
        print(f"step {step}/{plan['steps']} objective={total_loss:.4f} lr={lr:.6g} epochs={epochs:.2f}", flush=True)
        if stopping or step % plan["eval_every"] == 0 or step == plan["steps"]:
            if legacy:
                model.eval()
                with torch.inference_mode(), autocast():
                    losses = [model(*random_batch("val", eval_rng))[1].item() for _ in range(cfg["eval_batches"])]
                raw_val = sum(losses) / len(losses)
            else:
                raw_val = evaluate(model, streams["val"], evaluation["windows"]["val"], cfg["batch_size"], device, autocast, masks.get('val'))
            ema_val, selected = None, "raw"
            if ema:
                with ema.apply(model):
                    ema_val = evaluate(model, streams["val"], evaluation["windows"]["val"], cfg["batch_size"], device, autocast, masks.get('val'))
                if ema_val < raw_val:
                    selected = "ema"
            val = ema_val if selected == "ema" else raw_val
            with ema.apply(model) if selected == "ema" else contextlib.nullcontext():
                train_eval = evaluate(model, streams["train"], fixed_evaluation["windows"]["train"], cfg["batch_size"], device, autocast, masks.get('train'))
            improved, early_stop = early.observe(val, step, cfg.get("early_stopping_min_delta", 0), cfg.get("early_stopping_patience", 0))
            stop_reason = ("early_stopping" if early_stop else "user_stop" if stopping else
                           "pass_budget" if step == plan["steps"] and cfg.get("budget_mode") == "passes" else
                           "epoch_budget" if step == plan["steps"] and step < cfg["steps"] else
                           "step_limit" if step == plan["steps"] else None)
            state = dict(model=model.state_dict(), model_config=model.config_dict(), optimizer=optimizer.state_dict(),
                          scaler=scaler.state_dict(), training_config=cfg, tokenizer_json=tokenizer_json, data_hashes=data_hashes,
                          step=step, best_val=early.best, precision=precision, rng=rng.bit_generator.state,
                          eval_rng=eval_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                          plan=plan, evaluation=evaluation, tokens_seen=tokens_seen, optimizer_updates=optimizer_updates,
                          early_stopping=early.state_dict(), stop_reason=stop_reason,
                           sampler=sampler.state_dict() if sampler else None, ema=ema.state_dict() if ema else None)
            state.update(provenance, examples_seen=examples_seen)
            save_checkpoints(out, state, model, ema, selected, improved)
            metrics = dict(step=step, train_loss=train_eval, train_eval_loss=train_eval, optimization_loss=total_loss,
                           val_loss=val, raw_val_loss=raw_val, ema_val_loss=ema_val, selected_weights=selected,
                           generalization_gap=val - train_eval, perplexity=math.exp(min(val, 80)),
                           best_val_loss=early.best, best_step=early.best_step, bad_evals=early.bad_evals,
                           tokens_seen=tokens_seen, epochs=epochs, optimizer_updates=optimizer_updates,
                           planned_steps=plan["steps"], stop_reason=stop_reason,
                           eval_protocol=evaluation["protocol"], validation_tokens=fixed_evaluation["tokens"]["val"] if not legacy else cfg["eval_batches"] * cfg["batch_size"] * cfg["context"],
                            lr=lr, elapsed_seconds=time.monotonic() - start)
            if dialogue:
                metrics.update(examples_seen=examples_seen, epoch_unit='dialogue examples', loss_targets='assistant replies')
            elif custom_text:
                metrics.update(epoch_unit='tokens', loss_targets='all custom text')
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(f"EVAL train NLL={train_eval:.4f}; validation NLL={val:.4f}; gap={val - train_eval:.4f}; "
                  f"weights={selected}; raw={raw_val:.4f}" + (f" EMA={ema_val:.4f}" if ema_val is not None else "") +
                  f"; best={early.best:.4f} at step {early.best_step}; stale={early.bad_evals}; saved checkpoint", flush=True)
            if stop_reason:
                print(f"STOP {stop_reason}; best.pt retained from step {early.best_step} (NLL {early.best:.4f}).", flush=True)
                if stopping and not early_stop:
                    print("Stopped after saving checkpoint. Resume with the same configuration and data.", flush=True)
                break


if __name__ == "__main__":
    main()
