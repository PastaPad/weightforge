"""Single-worker local corpus/training jobs, with file-backed logs and graceful stop."""
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from threading import Event, RLock, Thread

import psutil
from pydantic import BaseModel, Field, model_validator

from .training import training_plan
from .frontier import read_state
from .corpus import corpus_plan
from .sources import SourceHost
from .links import TEXT_POLICY_VERSION
from .kiwix import BASE_URL, DEFAULT_FILE, DEFAULT_PATH, archive_info, download_name
from .model import model_config, inherit_architecture
from .hf_requests import HFImportRequest, HFPrepareRequest, HFTrainRequest, HFMergeRequest, HFExportRequest

class CrawlRequest(BaseModel):
    hosts: list[SourceHost] = Field(default_factory=lambda: ["soyjakwiki.org"], min_length=1)
    output: str = "data/raw-text-v4.jsonl"
    pages_per_site: int = Field(default=100, ge=1, le=100000)
    depth: int = Field(default=3, ge=0, le=20)
    delay: float = Field(default=3, ge=0.5, le=300, allow_inf_nan=False)
    ignore_robots: bool = True
    headed: bool = True
    challenge_wait: int = Field(default=180, ge=0, le=600)
    min_chars: int = Field(default=300, ge=1, le=100000)
    min_post_chars: int = Field(default=40, ge=1, le=100000)
    resume: bool = False
    wikipedia_zim: str = DEFAULT_PATH
    wikipedia_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    wikipedia_all: bool = False


class KiwixRequest(BaseModel):
    url: str = BASE_URL + DEFAULT_FILE
    output: str = DEFAULT_PATH

    @model_validator(mode='after')
    def check_archive(self):
        download_name(self.url)
        if Path(self.output).suffix.lower() != '.zim':
            raise ValueError('Download output must end in .zim')
        return self


class ExportRequest(BaseModel):
    checkpoint: str = 'runs/response/best.pt'
    output: str = 'exports/model-f16.gguf'
    precision: Literal['f16', 'f32'] = 'f16'

    @model_validator(mode='after')
    def check_output(self):
        if Path(self.output).suffix.lower() != '.gguf':
            raise ValueError('Export output must end in .gguf')
        return self


class PrepareRequest(BaseModel):
    input: str = "data/raw-text-v4.jsonl"
    output: str = "data/prepared-no-links"
    vocab_size: int = Field(default=8192, ge=259, le=65536)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)
    min_chars: int = Field(default=300, ge=1, le=100000)
    min_post_chars: int = Field(default=40, ge=1, le=100000)
    atomic_output: bool = False


class DatasetRequest(BaseModel):
    crawl: CrawlRequest = Field(default_factory=lambda: CrawlRequest(hosts=['wikipedia.org'], wikipedia_all=True))
    prepare: PrepareRequest = Field(default_factory=PrepareRequest)
    auto_paths: bool = True

    @model_validator(mode='after')
    def whole_archive(self):
        if self.crawl.hosts != ['wikipedia.org'] or not self.crawl.wikipedia_all:
            raise ValueError('A whole-ZIM dataset requires only Wikipedia and wikipedia_all=true')
        if self.crawl.resume and self.auto_paths:
            raise ValueError('Resume requires the existing corpus and dataset paths; disable automatic paths')
        return self

    def stages(self):
        return [('crawl', self.crawl), ('prepare', self.prepare.model_copy(update={
            'input': self.crawl.output, 'atomic_output': True}))]


class TrainRequest(BaseModel):
    data: str = "data/prepared-no-links"
    out: str = "runs/response"
    resume: bool = False
    context: int = Field(default=512, ge=8, le=4096)
    width: int = Field(default=256, ge=16, le=2048)
    layers: int = Field(default=6, ge=1, le=48)
    heads: int = Field(default=8, ge=1, le=64)
    dropout: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)
    architecture: Literal['gpt', 'modern'] = 'gpt'
    kv_heads: int | None = Field(default=None, ge=1, le=64)
    ffn_width: int | None = Field(default=None, ge=16, le=16384)
    rope_theta: float = Field(default=10000, gt=0, allow_inf_nan=False)
    gradient_checkpointing: bool = False
    optimizer_foreach: bool | None = None
    precision: Literal['auto', 'fp32', 'fp16', 'bf16'] = 'auto'
    batch_size: int = Field(default=2, ge=1, le=256)
    accumulation: int = Field(default=8, ge=1, le=1024)
    budget_mode: Literal["updates", "passes"] = "passes"
    steps: int = Field(default=2500, ge=1, le=10000000)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1, allow_inf_nan=False)
    min_learning_rate: float = Field(default=0.00003, ge=0, le=0.1, allow_inf_nan=False)
    warmup_steps: int = Field(default=200, ge=0)
    weight_decay: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    eval_every: int = Field(default=100, ge=1)
    eval_batches: int = Field(default=32, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    sampling: Literal["shuffled", "random"] = "shuffled"
    max_epochs: float = Field(default=25, ge=0, le=10000, allow_inf_nan=False)
    early_stopping_patience: int = Field(default=6, ge=0, le=10000)
    early_stopping_min_delta: float = Field(default=0.01, ge=0, le=10, allow_inf_nan=False)
    label_smoothing: float = Field(default=0, ge=0, lt=1, allow_inf_nan=False)
    ema_decay: float = Field(default=0, ge=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_config(self):
        if not self.resume:
            model_config(8192, self.model_dump())
            if self.width % self.heads or self.min_learning_rate > self.learning_rate:
                raise ValueError("Width must divide into heads and minimum LR <= learning rate")
            if self.budget_mode == "updates" and self.warmup_steps >= self.steps:
                raise ValueError("Warmup must be below the fixed update limit")
            if self.budget_mode == "passes" and self.max_epochs <= 0:
                raise ValueError("Data-pass mode requires a positive number of passes")
        return self


class DialogueRequest(BaseModel):
    checkpoint: str = 'runs/response/best.pt'
    output: str = 'data/prepared-dialogue'
    sources: list[Literal['oasst1', 'dolly15k']] = Field(default_factory=lambda: ['oasst1', 'dolly15k'])
    input: str | None = None
    cache: str = 'data/dialogue-downloads'
    validation_fraction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)


class CustomCorpusRequest(BaseModel):
    input: str
    checkpoint: str = 'runs/response/best.pt'
    output: str = 'data/prepared-custom-text'
    validation_fraction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)


class FineTuneRequest(TrainRequest):
    checkpoint: str = 'runs/response/best.pt'
    data: str = 'data/prepared-dialogue'
    out: str = 'runs/dialogue'
    learning_rate: float = Field(default=0.00005, gt=0, le=0.1, allow_inf_nan=False)
    min_learning_rate: float = Field(default=0.000005, ge=0, le=0.1, allow_inf_nan=False)
    max_epochs: float = Field(default=3, ge=0, le=10000, allow_inf_nan=False)
    warmup_steps: int = Field(default=50, ge=0)
    early_stopping_patience: int = Field(default=3, ge=0, le=10000)
    early_stopping_min_delta: float = Field(default=0.001, ge=0, le=10, allow_inf_nan=False)
    sampling: Literal['shuffled'] = 'shuffled'


class PipelineRequest(BaseModel):
    crawl: CrawlRequest = Field(default_factory=CrawlRequest)
    prepare: PrepareRequest = Field(default_factory=PrepareRequest)
    train: TrainRequest = Field(default_factory=TrainRequest)
    auto_paths: bool = True
    dialogue: DialogueRequest | None = None
    finetune: FineTuneRequest = Field(default_factory=FineTuneRequest)

    @model_validator(mode="after")
    def new_run(self):
        if self.train.resume:
            raise ValueError("An automated collection pipeline needs a fresh training run")
        if self.dialogue and self.finetune.resume:
            raise ValueError('An automated dialogue stage needs a fresh fine-tuning run')
        if self.crawl.resume and self.auto_paths:
            raise ValueError("Resuming a collection requires its existing corpus path; turn off automatic paths")
        return self

    def stages(self):
        # Stage inputs always come from the preceding stage, not stale GUI paths.
        stages = [("crawl", self.crawl),
                ("prepare", self.prepare.model_copy(update={"input": self.crawl.output})),
                ("train", self.train.model_copy(update={"data": self.prepare.output}))]
        if self.dialogue:
            checkpoint = str(Path(self.train.out) / 'best.pt')
            stages.extend([('dialogue', self.dialogue.model_copy(update={'checkpoint': checkpoint})),
                           ('finetune', self.finetune.model_copy(update={'checkpoint': checkpoint, 'data': self.dialogue.output}))])
        return stages


def tail(path, limit=16000):
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - limit))
        return f.read().decode("utf-8", errors="replace")


class Jobs:
    LIVE = {"running", "stopping", "closing"}
    COORDINATORS = {'pipeline', 'dataset'}

    def __init__(self, root, device, state_path=None):
        self.root = Path(root).resolve()
        self.device = device
        self.process = None
        self.job = None
        self.lock = RLock()
        self.finished = Event()
        self.finished.set()
        self.state_path = self.path(state_path) if state_path else self.root / "runs" / "jobs" / "latest.json"
        if self.state_path.exists():
            self.job = json.loads(self.state_path.read_text())
            if self.job["status"] in self.LIVE:
                # Recover only recorded identities, never arbitrary PIDs that may have been reused.
                root_process = self._live_process(self.job.get("processes", [{}])[0])
                if root_process:
                    self._remember_children(root_process, self.job)
                survivors = self._cleanup(self.job.get("processes", []))
                self.job.update(status="failed" if survivors else "interrupted", returncode=None,
                                finished_at=datetime.now(timezone.utc).isoformat())
                self._append(self.job, "RECOVER previous server stopped before this job finished; owned processes cleaned up")
                self._persist(self.job)

    @staticmethod
    def _identity(process):
        return {"pid": process.pid, "created": process.create_time()}

    @staticmethod
    def _live_process(identity):
        try:
            process = psutil.Process(identity["pid"])
            if process.create_time() == identity["created"] and process.status() != psutil.STATUS_ZOMBIE:
                return process
        except (psutil.Error, KeyError):
            pass
        return None

    def _persist(self, job):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2) + "\n")
        tmp.replace(self.state_path)

    def _append(self, job, message):
        with (self.root / job["log_path"]).open("a") as log:
            log.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

    def _remember_children(self, parent, job):
        known = {(row["pid"], row["created"]) for row in job["processes"]}
        try:
            children = parent.children(recursive=True)
        except psutil.Error:
            return
        changed = False
        for child in children:
            try:
                identity = self._identity(child)
            except psutil.Error:
                continue
            key = (identity["pid"], identity["created"])
            if key not in known:
                known.add(key)
                job["processes"].append(identity)
                changed = True
        if changed:
            self._persist(job)

    def _cleanup(self, identities):
        # Camoufox starts its own session/process group: killpg(crawler) cannot own its lifecycle.
        live = [p for row in reversed(identities) if (p := self._live_process(row))]
        for process in live:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(live, timeout=2)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=2)
        return [row["pid"] for row in identities if self._live_process(row)]

    def _progress(self, job):
        path = job.get("progress_path")
        if path:
            try:
                return json.loads((self.root / path).read_text())
            except (OSError, ValueError):
                pass
        return None

    def browser_snapshot_path(self, job_id):
        with self.lock:
            if not self.job or self.job["id"] != job_id or not self.job.get("progress_path"):
                return None
            progress_path = self.job["progress_path"]
            if self.job["kind"] in self.COORDINATORS:
                progress_path = (self._progress(self.job) or {}).get("crawl_progress_path")
                if not progress_path:
                    return None
            path = self.path(progress_path).with_suffix(".browser.png")
            return path if path.is_file() else None

    def _watch(self, process, job, finished):
        try:
            parent = psutil.Process(process.pid)
            forced = False
            while True:
                with self.lock:
                    self._remember_children(parent, job)
                    deadline = job.get("stop_requested_at")
                    grace = 150 if job["kind"] in self.COORDINATORS else 120 if job["kind"] in {"train", 'finetune', 'hf-finetune'} else 20
                    overdue = deadline is not None and time.time() - deadline > grace
                code = process.poll()
                if code is not None:
                    break
                if overdue:
                    self._append(job, "STOP timeout; terminating remaining owned processes")
                    self._cleanup(job["processes"])
                    forced = True
                time.sleep(0.2)
            with self.lock:
                job["status"] = "closing"
                self._persist(job)
            survivors = self._cleanup(job["processes"])
            progress = self._progress(job) or {}
            with self.lock:
                stopped = job["stop_requested"] or progress.get("phase") == "stopped"
                job.update(status="failed" if survivors else "stopped" if stopped else "succeeded" if code == 0 else "failed",
                           returncode=code, forced_stop=forced, remaining_pids=survivors,
                           finished_at=datetime.now(timezone.utc).isoformat())
                self._append(job, f"JOB {job['status']}; worker exited {code}; "
                             + (f"remaining processes: {survivors}" if survivors else "all owned browser/worker processes closed"))
                self._persist(job)
        except Exception as exc:
            survivors = self._cleanup(job.get("processes", []))
            with self.lock:
                job.update(status="failed", error=str(exc), remaining_pids=survivors)
                self._append(job, f"SUPERVISOR ERROR {exc}")
                self._persist(job)
        finally:
            finished.set()

    def path(self, value):
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Choose a path inside the workspace")
        return path

    def active(self):
        with self.lock:
            return self.job is not None and self.job["status"] in self.LIVE

    def training(self):
        with self.lock:
            # Reserve GPU memory across stage transitions, even while collecting/preparing.
            return self.active() and self.job["kind"] in {"train", "finetune", "pipeline", 'hf-finetune'}

    def prepared_datasets(self):
        """Discover complete local datasets, newest metadata first; no model loading."""
        candidates = set((self.root / "data").glob("**/meta.json"))
        with self.lock:
            if self.job and self.job["kind"] in {"prepare", 'dialogue', 'custom-corpus'} and self.job["status"] == "succeeded":
                candidates.add(self.root / self.job["output"] / "meta.json")
            if self.job and self.job["kind"] in self.COORDINATORS:
                candidates.add(self.path(self.job["settings"]["prepare"]["output"]) / "meta.json")
                if self.job['settings'].get('dialogue'):
                    candidates.add(self.path(self.job['settings']['dialogue']['output']) / 'meta.json')
        # Also discover custom prepared locations already associated with GUI runs.
        for record in (self.root / "runs").glob("*/web-job.json"):
            try:
                candidates.add(self.path(json.loads(record.read_text())["data"]) / "meta.json")
            except (OSError, ValueError, KeyError, TypeError):
                continue
        datasets = {}
        for path in candidates:
            if any(part.startswith('.prepare-') for part in path.parts):
                continue
            try:
                directory = self.path(str(path.parent))
                meta = json.loads(path.read_text())
                if not (directory / "tokenizer.json").is_file():
                    continue
                if any(meta["tokens"][split] <= 1 or (directory / f"{split}.bin").stat().st_size != 4 * meta["tokens"][split]
                       for split in ("train", "val")):
                    continue
                if meta.get('format') == 'dialogue-sft-v1' and any(
                        (directory/f'{split}.mask.bin').stat().st_size != meta['tokens'][split]
                        or not (directory/f'{split}.windows.json').is_file() for split in ('train', 'val')):
                    continue
                name = str(directory.relative_to(self.root))
                datasets[name] = dict(path=name, documents=meta["documents"], tokens=meta["tokens"],
                                      vocab_size=meta["vocab_size"], prepared_at=path.stat().st_mtime,
                                       corpus_report=meta.get('corpus_report'), format=meta.get('format'),
                                       dialogue_report=meta.get('dialogue_report'), custom_report=meta.get('custom_report'),
                                       base_checkpoint=meta.get('base_checkpoint'), chat_format=meta.get('chat_format'))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        all_rows = sorted(datasets.values(), key=lambda row: (-row["prepared_at"], row["path"]))
        rows = [row for row in all_rows if row['format'] not in {'dialogue-sft-v1', 'text-finetune-v1'}]
        return {"datasets": rows, "latest": rows[0]["path"] if rows else None,
                'dialogue_datasets': [row for row in all_rows if row['format'] == 'dialogue-sft-v1'],
                'finetuning_datasets': [row for row in all_rows if row['format'] in {'dialogue-sft-v1', 'text-finetune-v1'}]}

    def finetune_config(self, request, meta):
        from .dialogue import load_base, validate_base
        checkpoint = self.path(request.checkpoint)
        state, _ = load_base(checkpoint)
        validate_base(state, meta, checkpoint)
        config = request.model_dump(exclude={'data', 'out', 'resume', 'checkpoint'})
        return inherit_architecture(config, state['model_config'])

    def preview_training(self, request):
        if request.resume:
            raise ValueError("Preview is for new runs. Resume uses the saved configuration and plan.")
        data = self.path(request.data)
        meta = json.loads((data / "meta.json").read_text())
        fine_tuning = isinstance(request, FineTuneRequest)
        config = self.finetune_config(request, meta) if fine_tuning else request.model_dump(exclude={"data", "out", "resume"})
        if not fine_tuning and meta.get('format') in {'dialogue-sft-v1', 'text-finetune-v1'}:
            raise ValueError('Use the fine-tuning controls for a dialogue or custom text dataset')
        plan = training_plan(config, meta)
        plan.update(data=str(data.relative_to(self.root)), budget_mode=request.budget_mode,
                    model_config=model_config(meta['vocab_size'], config).to_dict(),
                    gradient_checkpointing=config.get('gradient_checkpointing', False),
                    precision=config.get('precision', 'auto'),
                    corpus_report=meta.get('corpus_report'),
                    text_policy=meta.get('text_policy'), links_removed=meta.get('text_policy') == TEXT_POLICY_VERSION,
                    max_epochs=request.max_epochs,
                    limit_reason="data_passes" if request.budget_mode == "passes" else
                                 "pass_cap" if plan["steps"] < request.steps else "fixed_updates")
        limit = config['batch_size'] * config['eval_batches'] * config['context']
        plan["validation_token_budget"] = min(meta["tokens"]["val"] - 1, limit)
        plan["full_validation"] = limit >= meta["tokens"]["val"] - 1
        if fine_tuning and meta.get('format') == 'dialogue-sft-v1':
            plan.update(epoch_unit='dialogue examples', dialogue_report=meta['dialogue_report'],
                        base_checkpoint=request.checkpoint, model_config=meta['base_model_config'],
                        supervised_tokens=meta['supervised_tokens'],
                        validation_token_budget=min(meta['supervised_tokens']['val'], limit),
                        validation_examples=min(request.batch_size * request.eval_batches, meta['documents']['val']),
                         full_validation=request.batch_size * request.eval_batches >= meta['documents']['val'])
        elif fine_tuning:
            plan.update(epoch_unit='tokens', custom_report=meta['custom_report'],
                        base_checkpoint=request.checkpoint, loss_targets='all custom text')
        return plan

    def inspect_corpus(self, request):
        source = self.path(request.input)
        _, report = corpus_plan(source, request.validation_fraction, request.min_chars, request.min_post_chars)
        return dict(input=str(source.relative_to(self.root)), **report)

    def wikipedia_archives(self):
        rows = []
        for path in sorted((self.root / 'data').glob('**/*.zim')):
            try:
                info = archive_info(self.path(str(path)))
                rows.append(dict(info, path=str(path.relative_to(self.root))))
            except (ValueError, OSError):
                continue
        return dict(archives=rows, default_path=DEFAULT_PATH, default_url=BASE_URL + DEFAULT_FILE)

    def export_checkpoints(self):
        """List saved inference files without reading weights or taking the GPU lock."""
        rows = []
        for candidate in (self.root / 'runs').rglob('*.pt'):
            if candidate.name == 'last.pt':
                continue
            try:
                path = self.path(str(candidate))
                stat = path.stat()
                if path.is_file():
                    rows.append(dict(path=str(path.relative_to(self.root)), bytes=stat.st_size, modified_at=stat.st_mtime))
            except (ValueError, OSError):
                continue
        return dict(checkpoints=sorted(rows, key=lambda row: (-row['modified_at'], row['path'])))

    def _crawl_request(self, request):
        if request.resume:
            saved = read_state(self.path(request.output))['settings']
            if 'wikipedia.org' in saved['hosts'] and not saved.get('wikipedia_archive'):
                raise ValueError('This is a legacy Wikipedia web crawl. Choose a fresh corpus for Kiwix collection.')
            whole_archive = request.wikipedia_all or saved.get('wikipedia_all', False)
            request = request.model_copy(update={key: value for key, value in saved.items()
                                                if key in CrawlRequest.model_fields and key not in {'pages_per_site', 'wikipedia_all'}})
            request.pages_per_site = max(request.pages_per_site, saved['pages_per_site'])
            request.wikipedia_all = whole_archive
        if 'wikipedia.org' in request.hosts:
            archive_info(self.path(request.wikipedia_zim))
            if self.path(request.wikipedia_zim) == self.path(request.output):
                raise ValueError('The Wikipedia archive and output corpus must be different files')
        return request

    def start(self, kind, request):
        with self.lock:
            return self._start(kind, request)

    def _pipeline_request(self, request, job_id):
        request = request.model_copy(deep=True)
        if request.auto_paths:
            request.crawl.output = f"data/raw-pipeline-{job_id}.jsonl"
            request.prepare.output = f"data/prepared-pipeline-{job_id}"
            request.train.out = f"runs/pipeline-{job_id}"
            if request.dialogue:
                request.dialogue.output = f'data/dialogue-pipeline-{job_id}'
                request.finetune.out = f'runs/dialogue-pipeline-{job_id}'
        request.crawl = self._crawl_request(request.crawl)
        request.prepare.input = request.crawl.output
        request.train.data = request.prepare.output
        paths = [self.path(request.crawl.output), self.path(request.prepare.output), self.path(request.train.out)]
        if request.dialogue:
            request.dialogue.checkpoint = str(Path(request.train.out) / 'best.pt')
            request.finetune.checkpoint = request.dialogue.checkpoint
            request.finetune.data = request.dialogue.output
            cache = self.path(request.dialogue.cache)
            paths.extend([self.path(request.dialogue.output), self.path(request.finetune.out)])
            if any(cache.is_relative_to(path) or path.is_relative_to(cache) for path in paths):
                raise ValueError('Dialogue cache and pipeline outputs must not overlap')
        if paths[0].exists() and not paths[0].is_file():
            raise ValueError("Pipeline corpus output must be a file")
        for path in paths[1:]:
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise ValueError("Pipeline dataset and run directories must be empty; enable automatic output paths or choose fresh directories")
        for index, path in enumerate(paths):
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[index + 1:]):
                raise ValueError("Pipeline corpus, dataset, and model outputs must not overlap")
        if 'wikipedia.org' in request.crawl.hosts:
            archive = self.path(request.crawl.wikipedia_zim)
            if any(archive.is_relative_to(path) or path.is_relative_to(archive) for path in paths):
                raise ValueError('Keep the Wikipedia archive separate from pipeline outputs')
        return request

    def _dataset_request(self, request, job_id):
        request = request.model_copy(deep=True)
        if request.auto_paths:
            request.crawl.output = f'data/raw-zim-{job_id}.jsonl'
            request.prepare.output = f'data/prepared-zim-{job_id}'
        request.crawl = self._crawl_request(request.crawl)
        request.whole_archive()  # Resume must also be a Wikipedia-only collection.
        request.prepare.input = request.crawl.output
        request.prepare.atomic_output = True
        corpus, dataset, archive = (self.path(value) for value in
                                    (request.crawl.output, request.prepare.output, request.crawl.wikipedia_zim))
        paths = [corpus, dataset, archive]
        if any(path.is_relative_to(other) or other.is_relative_to(path)
               for index, path in enumerate(paths) for other in paths[index + 1:]):
            raise ValueError('Keep the ZIM archive, corpus, and dataset paths separate')
        if corpus.exists() and (not corpus.is_file() or not request.crawl.resume):
            raise ValueError('Choose a fresh corpus, or resume its saved collection')
        if dataset.exists() and (not dataset.is_dir() or any(dataset.iterdir())):
            raise ValueError('Dataset output must be empty; choose a fresh directory')
        return request

    def _start(self, kind, request):
        if self.active():
            raise ValueError("A job is already running. Stop it or wait for completion.")
        if self.job and any(self._live_process(row) for row in self.job.get("processes", [])):
            raise ValueError("Previous job processes have not exited yet; refusing to reuse the browser profile")
        job_id = uuid.uuid4().hex[:12]
        output = None
        if kind == "crawl":
            output = self.path(request.output)
            request = self._crawl_request(request)
            args = ["scratchllm.crawl", "--hosts", *request.hosts, "--output", str(output),
                    "--pages-per-site", str(request.pages_per_site), "--depth", str(request.depth), "--delay", str(request.delay),
                    "--challenge-wait", str(request.challenge_wait), "--profile", str(self.path("data/browser-profile")),
                    "--min-chars", str(request.min_chars), "--min-post-chars", str(request.min_post_chars)]
            if 'wikipedia.org' in request.hosts:
                args.extend(['--wikipedia-zim', str(self.path(request.wikipedia_zim)), '--wikipedia-seed', str(request.wikipedia_seed)])
                if request.wikipedia_all:
                    args.append('--wikipedia-all')
            if request.resume:
                args.append('--resume')
            if request.headed:
                args.append("--headed")
            if request.ignore_robots:
                args.append("--ignore-robots")
        elif kind == 'kiwix':
            output = self.path(request.output)
            args = ['scratchllm.kiwix', '--url', request.url, '--output', str(output)]
        elif kind == 'export':
            checkpoint, output = self.path(request.checkpoint), self.path(request.output)
            if not checkpoint.is_file() or checkpoint.name == 'last.pt':
                raise ValueError('Choose an existing inference checkpoint, such as best.pt')
            if output.exists() or (self.root / request.output).is_symlink() or checkpoint == output:
                raise ValueError('Choose a fresh GGUF output file')
            args = ['scratchllm.gguf_export', '--checkpoint', str(checkpoint), '--output', str(output),
                    '--precision', request.precision]
        elif kind == 'hf-export':
            from .hf_export import converter_status, validate_paths
            model, output = self.path(request.model), self.path(request.output)
            if (self.root/request.output).is_symlink():
                raise ValueError('Choose a fresh GGUF output file')
            validate_paths(model, output)
            converter = converter_status()
            if not converter['ready']:
                raise ValueError('Install the llama.cpp converter described in HUGGINGFACE.md: ' + ', '.join(converter['missing']))
            args = ['scratchllm.hf_export', '--model', str(model), '--output', str(output), '--precision', request.precision]
        elif kind == "prepare":
            source, output = self.path(request.input), self.path(request.output)
            if not source.is_file():
                raise ValueError("Corpus file does not exist. Choose an existing .jsonl/.txt path or upload a .txt in dataset preparation.")
            if output.exists() and any(output.iterdir()):
                raise ValueError("Prepared output is not empty. Choose a new directory to preserve existing datasets.")
            args = ["scratchllm.prepare", "--input", str(source), "--output", str(output),
                    "--vocab-size", str(request.vocab_size), "--validation-fraction", str(request.validation_fraction),
                    "--min-chars", str(request.min_chars), "--min-post-chars", str(request.min_post_chars)]
            if request.atomic_output:
                args.append('--atomic-output')
        elif kind == 'dialogue':
            checkpoint, output, cache = self.path(request.checkpoint), self.path(request.output), self.path(request.cache)
            if not checkpoint.is_file():
                raise ValueError('Train/select your base best.pt before preparing dialogue data')
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError('Dialogue output must be empty; choose a fresh dataset directory')
            if cache.is_relative_to(output) or output.is_relative_to(cache):
                raise ValueError('Keep dialogue downloads separate from the prepared dataset')
            args = ['scratchllm.dialogue', '--checkpoint', str(checkpoint), '--output', str(output),
                    '--sources', *request.sources, '--cache', str(cache), '--validation-fraction', str(request.validation_fraction)]
            if not request.sources and not request.input:
                raise ValueError('Select a dialogue source or a custom conversation JSONL')
            if request.input:
                source = self.path(request.input)
                if not source.is_file() or output == source or output in source.parents:
                    raise ValueError('Choose an existing conversation JSONL separate from the output directory')
                args.extend(['--input', str(source)])
        elif kind == 'custom-corpus':
            from .custom_corpus import validate_paths
            checkpoint, source, output = (self.path(value) for value in (request.checkpoint, request.input, request.output))
            validate_paths(checkpoint, source, output)
            args = ['scratchllm.custom_corpus', '--checkpoint', str(checkpoint), '--input', str(source), '--output', str(output),
                    '--validation-fraction', str(request.validation_fraction)]
        elif kind in {'hf-import', 'hf-prepare', 'hf-finetune', 'hf-merge'}:
            from .huggingface import require_dependencies
            require_dependencies()
            if kind == 'hf-finetune' and request.resume:
                output = self.path(request.out)
                record = json.loads((output/'hf-run.json').read_text())
                if not (output/'last.json').is_file():
                    raise ValueError('No saved Hugging Face adapter checkpoint to resume')
                request = HFTrainRequest.model_validate(dict(record['request'], out=request.out, resume=True))
            resolved = request.model_dump()
            for key in ('model', 'input', 'output', 'data', 'out', 'run'):
                if key in resolved:
                    resolved[key] = str(self.path(resolved[key]))
            output = Path(resolved['out'] if kind == 'hf-finetune' else resolved['output'])
            if kind != 'hf-finetune' or not request.resume:
                if output.exists() and (not output.is_dir() or any(output.iterdir())):
                    raise ValueError('Choose a fresh output directory (or Resume for an existing fine-tuning run)')
            for key in ('model', 'input', 'data', 'run'):
                if key in resolved:
                    source = Path(resolved[key])
                    if output.is_relative_to(source) or source.is_relative_to(output):
                        raise ValueError('Keep Hugging Face input and output paths separate')
            if kind == 'hf-merge':
                record = json.loads((Path(resolved['run'])/'hf-run.json').read_text())
                self.path(record['request']['model'])
            if kind == 'hf-prepare' and not Path(resolved['input']).is_file():
                raise ValueError('Source JSONL file does not exist. Select an existing file in 03 / Prepare scraper text: ' + request.input)
            if kind == 'hf-finetune' and not request.resume:
                from .hf_train import training_preview
                training_preview(HFTrainRequest.model_validate(resolved))
            request_path = self.path(f'runs/jobs/{job_id}.hf-request.json')
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(resolved, indent=2) + '\n')
            args = (['scratchllm.hf_train'] if kind == 'hf-finetune' else
                    ['scratchllm.huggingface', '--stage', kind.removeprefix('hf-'), '--cache', str(self.path('data/huggingface-cache'))])
            args.extend(['--request', str(request_path)])
        elif kind in {"train", 'finetune'}:
            output = self.path(request.out)
            config_path, record_path = output / "web-config.json", output / "web-job.json"
            if request.resume:
                if not (output / "last.pt").is_file() or not record_path.is_file():
                    raise ValueError("No resumable GUI checkpoint in this run directory")
                record = json.loads(record_path.read_text())
                if record.get('kind', 'train') != kind:
                    raise ValueError('Resume with the same training stage as the saved run')
                data = self.path(record["data"])
                config = json.loads(config_path.read_text())
                if record["device"] != self.device:
                    raise ValueError("Resume requires the original device")
            else:
                if output.exists() and any(output.iterdir()):
                    raise ValueError("Run directory is not empty. Resume it or choose a new directory.")
                data = self.path(request.data)
                config = request.model_dump(exclude={"data", "out", "resume", 'checkpoint'})
            if not (data / "meta.json").is_file():
                raise ValueError("Prepared dataset is missing. Prepare the corpus first.")
            meta = json.loads((data / 'meta.json').read_text())
            if kind == 'finetune':
                if not request.resume:
                    config = self.finetune_config(request, meta)
            elif meta.get('format') in {'dialogue-sft-v1', 'text-finetune-v1'}:
                raise ValueError('Use fine-tuning to preserve base weights for dialogue or custom text')
            if not request.resume and meta.get('text_policy') != TEXT_POLICY_VERSION:
                raise ValueError('Prepare the corpus into a fresh dataset first: this dataset predates URL removal. Existing v4 raw text can be prepared again.')
            plan = training_plan(config, meta)
            if not request.resume and request.budget_mode == "passes":
                config["steps"] = plan["steps"]
            output.mkdir(parents=True, exist_ok=True)
            if not request.resume:
                config_path.write_text(json.dumps(config, indent=2))
                record_path.write_text(json.dumps({"data": str(data.relative_to(self.root)), "device": self.device, 'kind': kind,
                                                   **({'checkpoint': request.checkpoint} if kind == 'finetune' else {})}))
            args = ["scratchllm.train", "--data", str(data), "--config", str(config_path), "--out", str(output), "--device", self.device]
            if request.resume:
                args.append("--resume")
            elif kind == 'finetune':
                args.extend(['--init-from', str(self.path(request.checkpoint))])
        elif kind in self.COORDINATORS:
            request = self._dataset_request(request, job_id) if kind == 'dataset' else self._pipeline_request(request, job_id)
            output = self.path(request.prepare.output if kind == 'dataset' else request.finetune.out if request.dialogue else request.train.out)
            request_path = self.path(f"runs/jobs/{job_id}.pipeline.json")
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(request.model_dump(), indent=2) + "\n")
            args = ["scratchllm.pipeline", "--request", str(request_path), "--root", str(self.root),
                    "--device", self.device, "--stage-state", f"runs/jobs/{job_id}.stage.json"]
            if kind == 'dataset':
                args.append('--dataset-only')
        else:
            raise ValueError("Unknown job type")
        log_dir = self.root / "runs" / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job_id}.log"
        progress_path = log_dir / f"{job_id}.progress.json"
        if kind in {"crawl", "pipeline", 'kiwix', 'dataset', 'export', 'custom-corpus', 'hf-import', 'hf-prepare', 'hf-finetune', 'hf-merge', 'hf-export'}:
            args.extend(["--progress-file", str(progress_path)])
        env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        env.setdefault("OMP_NUM_THREADS", "2")
        if kind in {'export', 'custom-corpus', 'hf-import', 'hf-prepare', 'hf-merge', 'hf-export'} or (kind == 'hf-finetune' and request.device == 'cpu'):
            env['CUDA_VISIBLE_DEVICES'] = ''
        if kind.startswith('hf-'):
            env['HF_HUB_DISABLE_XET'] = '1'  # Chunked HTTP download progress/cancellation; cached partials resume.
        with log_path.open("ab") as log:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", *args], cwd=self.root,
                                            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.job = {"id": job_id, "kind": kind, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                    "log_path": str(log_path.relative_to(self.root)), "output": str(output.relative_to(self.root)),
                    "progress_path": str(progress_path.relative_to(self.root)) if kind in {"crawl", "pipeline", 'kiwix', 'dataset', 'export', 'custom-corpus', 'hf-import', 'hf-prepare', 'hf-finetune', 'hf-merge', 'hf-export'} else None,
                    "stop_requested": False, "returncode": None,
                    "settings": dict(config, data=str(data.relative_to(self.root)), out=str(output.relative_to(self.root)), resume=request.resume,
                                     **({'checkpoint': record.get('checkpoint') if request.resume else request.checkpoint} if kind == 'finetune' else {}))
                                if kind in {"train", 'finetune'} else request.model_dump(),
                    "processes": [self._identity(psutil.Process(self.process.pid))]}
        self._persist(self.job)
        self.finished = Event()
        Thread(target=self._watch, args=(self.process, self.job, self.finished), daemon=True).start()
        return self.status()

    def stop(self):
        with self.lock:
            if self.active() and not self.job["stop_requested"]:
                self.job.update(stop_requested=True, stop_requested_at=time.time(), status="stopping")
                self._append(self.job, "STOP requested; waiting for graceful worker/browser shutdown")
                self._persist(self.job)
                # Leave Playwright alive so the crawler can close its browser and flush the profile.
                try:
                    self.process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return self.status()

    def status(self):
        with self.lock:
            if self.job is None:
                return {"job": None}
            result = {k: v for k, v in self.job.items() if k != "processes"}
        result["log"] = tail(self.root / result["log_path"])
        result["progress"] = self._progress(result)
        result["browser_snapshot"] = self.browser_snapshot_path(result["id"]) is not None
        if result['kind'] in {'export', 'hf-export'}:
            result['export_ready'] = bool((result.get('progress') or {}).get('phase') == 'completed'
                                          and self.path(result['output']).is_file())
        if result['kind'] == 'prepare':
            try:
                result['corpus_report'] = json.loads((self.root/result['output']/'report.json').read_text())
            except (OSError, ValueError):
                pass
        if result['kind'] == 'dialogue':
            try:
                result['dialogue_report'] = json.loads((self.root/result['output']/'report.json').read_text())
            except (OSError, ValueError):
                pass
        if result['kind'] == 'custom-corpus':
            try:
                meta = json.loads((self.root/result['output']/'meta.json').read_text())
                result.update(custom_report=meta['custom_report'], dataset_ready=True)
            except (OSError, ValueError, KeyError):
                pass
        if result['kind'] in {'hf-import', 'hf-prepare', 'hf-merge'}:
            result['hf_ready'] = (self.root/result['output']/('hf-meta.json' if result['kind'] == 'hf-prepare' else 'import.json')).is_file()
        if result['kind'] == 'hf-finetune':
            result['adapter_ready'] = (self.root/result['output']/'best.json').is_file()
            result['metrics'] = []
            for line in tail(self.root/result['output']/'metrics.jsonl').splitlines():
                try:
                    result['metrics'].append(json.loads(line))
                except ValueError:
                    pass
        if result['kind'] in {'crawl', 'pipeline', 'dataset'}:
            output = result.get('settings', {}).get('crawl', {}).get('output') if result['kind'] in self.COORDINATORS else result.get('output')
            try:
                if output:
                    saved = read_state(self.path(output))
                    result['resumable_crawl'] = dict(output=output, pending=saved['pending'], settings=saved['settings'])
            except (ValueError, OSError, sqlite3.Error):
                pass
        if result["kind"] in self.COORDINATORS:
            pipeline = result["progress"] or {}
            if result["status"] not in self.LIVE and pipeline.get("phase") not in {"completed", "failed", "stopped"}:
                # A killed coordinator cannot finalize its own snapshot. The supervisor's
                # terminal status remains authoritative after a crash or server restart.
                pipeline.update(phase=result["status"], detail=f"PIPELINE {result['status']}; remaining stages cancelled")
                for stage in pipeline.get("stages", []):
                    if stage["status"] in self.LIVE:
                        stage["status"] = result["status"]
                    elif stage["status"] == "pending":
                        stage["status"] = "cancelled"
            current = pipeline.get("current") or {}
            result.update(pipeline=pipeline, stage=pipeline.get("stage"),
                          progress=current.get("progress") if pipeline.get("stage") == "crawl" else None,
                          metrics=current.get("metrics", []))
            if result['kind'] == 'dataset':
                result['dataset_ready'] = (self.root / result['output'] / 'meta.json').is_file()
                result['resumable_dataset'] = bool(result.get('resumable_crawl') and not result['dataset_ready']
                                                   and result['status'] not in self.LIVE | {'succeeded'})
            elif (self.root / result["output"] / "best.pt").is_file():
                result["checkpoint"] = str(Path(result["output"]) / "best.pt")
        if result["kind"] in {"train", 'finetune'}:
            result["metrics"] = []
            for line in tail(self.root / result["output"] / "metrics.jsonl").splitlines():
                try:
                    result["metrics"].append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            result["checkpoint"] = str(Path(result["output"]) / "best.pt")
        return {"job": result}

    def close(self):
        if self.active():
            self.stop()
            grace = 160 if self.job["kind"] in self.COORDINATORS else 130 if self.job["kind"] in {"train", 'finetune', 'hf-finetune'} else 30
            if not self.finished.wait(timeout=grace):
                self._cleanup(self.job["processes"])
                self.finished.wait(timeout=5)
