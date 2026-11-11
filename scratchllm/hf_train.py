"""User-started LoRA/QLoRA fine-tuning of imported GGUF models on scraper text."""
import argparse
import copy
import gc
import json
import math
import random
import shutil
import signal
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from threading import Event

import numpy as np
import psutil
import torch

from .dialogue_training import ShuffledExamples
from .frontier import corpus_lock
from .hf_requests import HFTrainRequest, HFMergeRequest
from .huggingface import (DATA_FORMAT, atomic_json, check_resources, check_stop,
                          identity, load_import, require_dependencies, sha256)
from .links import TEXT_POLICY_VERSION
from .progress import CrawlProgress
from .training import EarlyStopping, fixed_windows, learning_rate, make_batch

RUN_FORMAT = 'huggingface-lora-run-v1'


def load_data(request, verify=False):
    imported = load_import(request.model, verify=verify)
    directory = Path(request.data)
    meta = json.loads((directory/'hf-meta.json').read_text())
    if (meta.get('format') != DATA_FORMAT or meta.get('text_policy') != TEXT_POLICY_VERSION or meta.get('dtype') != '<u4'
            or meta['artifact_id'] != imported['artifact_id'] or meta['vocabulary'] != imported['model']['vocabulary']):
        raise ValueError('Prepare scraper text for this exact imported model/tokenizer first')
    if not 8 <= meta['context'] <= imported['model']['context']:
        raise ValueError('Dataset context is outside the imported model context')
    for split in ('train', 'val'):
        path = directory/f'{split}.bin'
        if meta['tokens'][split] < 2 or path.stat().st_size != meta['tokens'][split] * 4:
            raise ValueError('Prepared token file does not match its metadata: ' + split)
        if verify and sha256(path) != meta['hashes'][split]:
            raise ValueError('Prepared token file changed: ' + split)
    return imported, meta


def training_preview(request):
    require_dependencies()
    imported, meta = load_data(request)
    windows = math.ceil((meta['tokens']['train'] - 1) / meta['context'])
    examples_per_step = request.batch_size * request.accumulation
    steps = max(1, math.ceil(request.max_epochs * windows / examples_per_step))
    if steps > 10000000:
        raise ValueError('Fine-tuning budget exceeds 10,000,000 updates; choose fewer passes')
    info = imported['model']
    adapter = request.rank * max(1, info['hidden_size']) * max(1, info['layers']) * 30
    embeddings = min(info['parameters'], info['embedding_parameters'])
    base_bytes = (info['parameters'] - embeddings) * 0.65 + embeddings * 4 if request.mode == 'qlora' else info['parameters'] * 4
    activation = request.batch_size * meta['context'] * (info['hidden_size'] * max(1, info['layers']) * 16 + info['vocabulary'] * 12)
    estimate = int(base_bytes + adapter * 16 + activation + 512 * 1024**2)
    available = (torch.cuda.mem_get_info(0)[0] if torch.cuda.is_available() else 0) if request.device == 'cuda' else psutil.virtual_memory().available
    return dict(steps=steps, warmup_steps=min(request.warmup_steps, steps // 10),
                eval_every=min(request.eval_every, max(1, math.ceil(windows / examples_per_step))),
                planned_epochs=steps * examples_per_step / windows, requested_epochs=request.max_epochs,
                epoch_unit='non-overlapping token windows', windows=windows, context=meta['context'],
                train_tokens=meta['tokens']['train'], validation_tokens=meta['tokens']['val'], documents=meta['documents'],
                parameters=info['parameters'], estimated_trainable_parameters=adapter,
                estimated_memory_bytes=estimate, available_memory_bytes=available, fits_estimate=estimate <= available * 0.9,
                model=request.model, data=request.data, mode=request.mode, device=request.device,
                artifact_id=imported['artifact_id'], corpus_report=meta['corpus_report'])


def build_model(request, imported, adapter_state=None):
    require_dependencies()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    kwargs = dict(local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation='eager')
    if request.mode == 'qlora':
        if request.device != 'cuda' or not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (6, 0):
            raise ValueError('QLoRA requires a CUDA GPU with compute capability 6.0 or newer')
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32), device_map={'': 0})
    elif request.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select CPU LoRA or configure CUDA')
    model = AutoModelForCausalLM.from_pretrained(request.model, **kwargs)
    if request.mode == 'qlora':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.to(request.device)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=request.rank, lora_alpha=request.alpha,
                                            lora_dropout=request.dropout, bias='none', target_modules='all-linear'))
    if adapter_state is not None:
        from peft import get_peft_model_state_dict
        if set(adapter_state) != set(get_peft_model_state_dict(model)):
            raise ValueError('Saved adapter does not match its model/LoRA configuration')
        set_peft_model_state_dict(model, adapter_state)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any('lora_' not in name for name, _ in trainable):
        raise ValueError('Expected only LoRA adapter weights to be trainable')
    return model


def windows(length, context):
    return fixed_windows(length, context, math.ceil((length - 1) / context), 0)


def batch(stream, rows, device):
    inputs, targets = make_batch(stream, rows, device)
    attention = (targets != -100).long()
    return inputs, targets, attention


def batch_loss(model, stream, rows, device):
    inputs, targets, attention = batch(stream, rows, device)
    logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
    # make_batch already shifts targets; do not apply the HF built-in second shift.
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


@torch.inference_mode()
def evaluate(model, stream, rows, batch_size, device):
    training = model.training
    model.eval()
    total, tokens = 0.0, 0
    try:
        for offset in range(0, len(rows), batch_size):
            part = rows[offset:offset + batch_size]
            count = sum(size for _, size in part)
            loss = batch_loss(model, stream, part, device)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite validation loss')
            total += loss.item() * count
            tokens += count
    finally:
        model.train(training)
    return total / tokens


def checkpoint_path(run, selection='last', verify=True):
    run = Path(run).resolve()
    pointer = json.loads((run/f'{selection}.json').read_text())
    path = (run/pointer['directory']).resolve()
    if not path.is_relative_to(run/'checkpoints'):
        raise ValueError('Invalid adapter checkpoint path')
    manifest = json.loads((path/'snapshot.json').read_text())
    if manifest.get('format') != RUN_FORMAT:
        raise ValueError('Invalid adapter checkpoint format')
    for name, digest in manifest['files'].items():
        file = (path/name).resolve()
        if not file.is_relative_to(path) or not file.is_file() or (verify and sha256(file) != digest):
            raise ValueError('Adapter checkpoint file is missing or changed: ' + name)
    return path


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def save_snapshot(run, model, state, improved):
    run = Path(run)
    snapshots = run/'checkpoints'
    snapshots.mkdir(parents=True, exist_ok=True)
    previous = [json.loads((run/f'{key}.json').read_text()) for key in ('best', 'last') if (run/f'{key}.json').is_file()]
    temporary = Path(tempfile.mkdtemp(prefix='.saving-', dir=snapshots))
    final = snapshots/f'step-{state["step"]:08d}-{uuid.uuid4().hex[:8]}'
    try:
        model.save_pretrained(temporary/'adapter', safe_serialization=True)
        torch.save(cpu_state(state), temporary/'state.pt')
        files = {str(path.relative_to(temporary)): sha256(path) for path in temporary.rglob('*') if path.is_file()}
        atomic_json(temporary/'snapshot.json', dict(format=RUN_FORMAT, files=files, step=state['step']))
        temporary.rename(final)
        pointer = dict(directory=str(final.relative_to(run)), step=state['step'])
        if improved:
            atomic_json(run/'best.json', pointer)
        atomic_json(run/'last.json', pointer)
        # Retain just best/last complete snapshots; never remove an unrecognized directory.
        best = json.loads((run/'best.json').read_text()) if (run/'best.json').is_file() else None
        for item in previous:
            old = (run/item['directory']).resolve()
            if item not in (best, pointer) and old.is_relative_to(snapshots.resolve()) and (old/'snapshot.json').is_file():
                shutil.rmtree(old)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def run_training(request, progress=None, cancel=None):
    with corpus_lock(request.out):
        return _run_training(request, progress, cancel)


def _run_training(request, progress=None, cancel=None):
    require_dependencies()
    from peft import get_peft_model_state_dict
    progress = progress or CrawlProgress(None, 0)
    cancel = cancel or Event()
    run = Path(request.out).resolve()
    if request.resume:
        record = json.loads((run/'hf-run.json').read_text())
        if record.get('format') != RUN_FORMAT:
            raise ValueError('Select a resumable Hugging Face fine-tuning run')
        request = HFTrainRequest.model_validate(dict(record['request'], resume=True, out=str(run)))
    request = request.model_copy(update={key: str(Path(getattr(request, key)).resolve()) for key in ('model', 'data', 'out')})
    imported, meta = load_data(request, verify=True)
    plan = training_preview(request)
    if not plan['fits_estimate']:
        raise ValueError(f"Estimated fine-tuning memory {plan['estimated_memory_bytes'] / 1024**3:.2f} GiB exceeds the usable "
                         f"{request.device} budget; choose a smaller model, lower context/batch/rank, or QLoRA")
    settings = request.model_dump(exclude={'resume'})
    data_hash = sha256(Path(request.data)/'hf-meta.json')
    saved = None
    if request.resume:
        saved = torch.load(checkpoint_path(run)/'state.pt', map_location='cpu', weights_only=True)
        if (saved['settings'] != settings or saved['data_hash'] != data_hash
                or saved['artifact_id'] != imported['artifact_id'] or saved['plan'] != {k: plan[k] for k in ('steps', 'warmup_steps', 'eval_every', 'windows', 'context')}):
            raise ValueError('Resume requires the original model, dataset, configuration and budget')
    elif run.exists() and any(run.iterdir()):
        raise ValueError('Choose a fresh fine-tuning run directory or resume its saved checkpoint')
    random.seed(request.seed)
    np.random.seed(request.seed)
    torch.manual_seed(request.seed)
    if request.device == 'cuda':
        torch.cuda.manual_seed_all(request.seed)
    progress.update(phase='loading', detail=f'Loading frozen base weights and {request.mode.upper()} adapters',
                    completed=0, budget=plan['steps'], unit='updates')
    model = build_model(request, imported, saved['adapter'] if saved else None)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=request.learning_rate, weight_decay=request.weight_decay, foreach=False)
    streams = {split: np.memmap(Path(request.data)/f'{split}.bin', dtype='<u4', mode='r') for split in ('train', 'val')}
    sampler = ShuffledExamples(windows(len(streams['train']), meta['context']), request.seed)
    evaluation = {split: fixed_windows(len(stream), meta['context'], request.batch_size * request.eval_batches, request.seed + index + 1)
                  for index, (split, stream) in enumerate(streams.items())}
    early, step, tokens_seen, examples_seen = EarlyStopping(), 0, 0, 0
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler'])
        early = EarlyStopping(**saved['early_stopping'])
        step, tokens_seen, examples_seen = saved['step'], saved['tokens_seen'], saved['examples_seen']
        torch.set_rng_state(saved['torch_rng'])
        if request.device == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        if saved['stop_reason'] == 'early_stopping' or step >= plan['steps']:
            progress.update(phase='completed', detail='This fine-tuning run has already finished', completed=step)
            return
    run.mkdir(parents=True, exist_ok=True)
    if not request.resume:
        atomic_json(run/'hf-run.json', dict(format=RUN_FORMAT, request=settings, artifact_id=imported['artifact_id'],
                                           data_hash=data_hash, source=imported['source'], plan=plan))
    progress.update(phase='training', detail=f'{sum(p.numel() for p in trainable):,} trainable adapter parameters; base weights frozen',
                    completed=step, budget=plan['steps'], trainable_parameters=sum(p.numel() for p in trainable))
    start = time.monotonic()
    while step < plan['steps']:
        objective, lr = 0.0, learning_rate(step, settings, plan)
        if not cancel.is_set():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr
            rows = sampler.take(request.batch_size * request.accumulation)
            count = sum(length for _, length in rows)
            for micro in range(request.accumulation):
                part = rows[micro * request.batch_size:(micro + 1) * request.batch_size]
                weight = sum(length for _, length in part) / count
                loss = batch_loss(model, streams['train'], part, request.device)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss; lower the learning rate')
                objective += loss.detach().item() * weight
                (loss * weight).backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not torch.isfinite(norm):
                raise ValueError('Non-finite adapter gradients')
            optimizer.step()
            step += 1
            tokens_seen += count
            examples_seen += len(rows)
            print(f'step {step}/{plan["steps"]} objective={objective:.4f} lr={lr:.6g}', flush=True)
            progress.update(completed=step)
        if cancel.is_set() or step % plan['eval_every'] == 0 or step == plan['steps']:
            val = evaluate(model, streams['val'], evaluation['val'], request.batch_size, request.device)
            train = evaluate(model, streams['train'], evaluation['train'], request.batch_size, request.device)
            improved, early_stop = early.observe(val, step, request.early_stopping_min_delta, request.early_stopping_patience)
            reason = 'early_stopping' if early_stop else 'user_stop' if cancel.is_set() else 'pass_budget' if step == plan['steps'] else None
            state = dict(settings=settings, artifact_id=imported['artifact_id'], data_hash=data_hash,
                         plan={k: plan[k] for k in ('steps', 'warmup_steps', 'eval_every', 'windows', 'context')},
                         adapter=get_peft_model_state_dict(model), optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
                         early_stopping=early.state_dict(), step=step, tokens_seen=tokens_seen, examples_seen=examples_seen,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if request.device == 'cuda' else None,
                         python_rng=random.getstate(), stop_reason=reason)
            save_snapshot(run, model, state, improved)
            metric = dict(step=step, planned_steps=plan['steps'], train_loss=train, train_eval_loss=train,
                          optimization_loss=objective, val_loss=val, perplexity=math.exp(min(val, 80)),
                          generalization_gap=val - train, best_val_loss=early.best, best_step=early.best_step,
                          selected_weights='LoRA adapter', epochs=examples_seen / plan['windows'], epoch_unit='window',
                          tokens_seen=tokens_seen, lr=lr, stop_reason=reason, elapsed_seconds=time.monotonic() - start)
            with (run/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(metric) + '\n')
            print(f'EVAL validation NLL={val:.4f}; best={early.best:.4f}; adapter checkpoint saved', flush=True)
            if reason:
                progress.update(phase='stopped' if reason == 'user_stop' else 'completed',
                                detail=f'FINE-TUNING {reason}; best adapter retained in {run}', completed=step, run=str(run))
                return


def merge_model(request, progress=None, cancel=None):
    with corpus_lock(request.run), corpus_lock(request.output):
        return _merge_model(request, progress, cancel)


def _merge_model(request, progress=None, cancel=None):
    require_dependencies()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    progress = progress or CrawlProgress(None, 0)
    run, output = Path(request.run).resolve(), Path(request.output).resolve()
    record = json.loads((run/'hf-run.json').read_text())
    source = Path(record['request']['model'])
    imported = load_import(source, verify=True)
    if imported['artifact_id'] != record['artifact_id']:
        raise ValueError('Fine-tuning base model changed')
    best = checkpoint_path(run, 'best')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a fresh merged-model directory')
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (run, source.resolve())):
        raise ValueError('Keep merged output separate from the base model and fine-tuning run')
    output.parent.mkdir(parents=True, exist_ok=True)
    check_resources(ram_bytes=imported['model']['parameters'] * 8 + 512 * 1024**2,
                    disk_bytes=imported['model']['parameters'] * (2 if request.precision == 'f16' else 4) + 128 * 1024**2,
                    directory=output.parent)
    progress.update(phase='merging', detail='Merging the best LoRA adapter into a complete Hugging Face model on CPU')
    temporary = Path(tempfile.mkdtemp(prefix='.hf-merge-', dir=output.parent))
    try:
        model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False, dtype=torch.float32)
        model = PeftModel.from_pretrained(model, best/'adapter', local_files_only=True).merge_and_unload(safe_merge=True)
        check_stop(cancel)
        model.to(torch.float16 if request.precision == 'f16' else torch.float32)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Merged weights overflowed; choose F32')
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size='1GB')
        AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False).save_pretrained(temporary)
        del model
        gc.collect()
        files = {str(path.relative_to(temporary)): sha256(path, cancel) for path in temporary.rglob('*') if path.is_file()}
        manifest = dict(imported, files=files, artifact_id=identity(files), precision=request.precision,
                        source=dict(imported['source'], adapter_run=str(run), adapter_snapshot_sha256=sha256(best/'snapshot.json')),
                        imported_at=time.time())
        atomic_json(temporary/'import.json', manifest)
        check_stop(cancel)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    progress.update(phase='completed', detail=f'MERGED HF MODEL READY {output}', output=str(output))
    return manifest


def adapter_archive(run, output):
    run = Path(run)
    best = checkpoint_path(run, 'best')
    record = json.loads((run/'hf-run.json').read_text())
    source = Path(record['request']['model'])
    imported = load_import(source)
    if imported['artifact_id'] != record['artifact_id']:
        raise ValueError('Fine-tuning base model changed')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((best/'adapter').rglob('*')):
            if file.is_file():
                archive.write(file, 'adapter/' + str(file.relative_to(best/'adapter')))
        for name in imported['files']:
            if 'token' in name or name.endswith(('.model', '.jinja')):
                archive.write(source/name, 'tokenizer/' + name)
        archive.writestr('run.json', json.dumps(record, indent=2))
        archive.writestr('base-import.json', json.dumps(imported, indent=2))
        archive.writestr('README.txt', 'PEFT LoRA adapter, not a standalone GGUF. Use with the exact imported base described in base-import.json.\n'
                          'The GUI can merge this adapter into a complete Hugging Face model for inference or llama.cpp conversion.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cancel = Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    progress = CrawlProgress(args.progress_file, 0)
    try:
        run_training(HFTrainRequest.model_validate(json.loads(Path(args.request).read_text())), progress, cancel)
    except Exception as exc:
        progress.update(phase='failed', detail=str(exc))
        raise


if __name__ == '__main__':
    main()
