"""Self-hosted completion API; one GPU request at a time."""
import argparse
import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import uvicorn
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field, model_validator

from .generate import Generator
from .jobs import Jobs, CrawlRequest, PrepareRequest, TrainRequest, PipelineRequest, DialogueRequest, FineTuneRequest, KiwixRequest, DatasetRequest, ExportRequest, CustomCorpusRequest
from .memory import MemoryStore
from .workspace import chat as workspace_chat
from .presets import hardware_presets
from .hf_requests import HFRepositoryRequest, HFImportRequest, HFPrepareRequest, HFTrainRequest, HFMergeRequest, HFExportRequest


class CompletionRequest(BaseModel):
    prompt: str = Field(default="", max_length=32000)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=3, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=100000)
    top_p: float = Field(default=0.9, gt=0, le=1, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2, allow_inf_nan=False)


class LoadRequest(BaseModel):
    checkpoint: str


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    content: str = Field(min_length=1, max_length=32000)


class ChatRequest(CompletionRequest):
    history: list[ChatMessage] = Field(default_factory=list, max_length=100)

    @model_validator(mode='after')
    def check_history(self):
        if len(self.history) % 2 or any(message.role != ('user' if index % 2 == 0 else 'assistant') for index, message in enumerate(self.history)):
            raise ValueError('History must contain alternating user/assistant pairs')
        if sum(len(message.content) for message in self.history) + len(self.prompt) > 128000:
            raise ValueError('Conversation is too long; clear older turns')
        return self


class WorkspaceRequest(CompletionRequest):
    session_id: str = Field(default='local', pattern=r'^[A-Za-z0-9_-]{1,64}$')
    goal: str | None = Field(default=None, max_length=2000)
    review: bool = False


class FeedbackRequest(BaseModel):
    episode_id: int = Field(ge=1)
    helpful: bool
    note: str = Field(default='', max_length=2000)


def create_app(checkpoint, device, workspace=None, state_path=None):
    jobs = Jobs(workspace or Path.cwd(), device)
    exports = Jobs(jobs.root, 'cpu', state_path='runs/exports/latest.json')
    state_file = jobs.path(state_path) if state_path else None
    if state_file and state_file.exists():
        # An explicitly configured state file takes precedence over the startup fallback.
        checkpoint = jobs.path(json.loads(state_file.read_text())["checkpoint"])
    generator = Generator(checkpoint, device) if Path(checkpoint).is_file() else None
    lock = Lock()
    loaded_checkpoint = str(checkpoint) if generator else None
    checkpoint_error = None if generator else f'Checkpoint file not found: {checkpoint}'

    def dataset_for(path):
        record = Path(path).resolve().parent / "web-job.json"
        if record.is_file():
            try:
                return json.loads(record.read_text()).get("data")
            except (OSError, ValueError):
                pass
        return None

    loaded_dataset = dataset_for(checkpoint) if generator else None
    memory = MemoryStore(jobs.path('runs/memory.sqlite3'))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.gather(asyncio.to_thread(jobs.close), asyncio.to_thread(exports.close))

    app = FastAPI(title="Scratch LLM", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).with_name("web") / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"status": "ok", "device": device, "model": generator.model.config_dict() if generator else None,
                 "checkpoint": loaded_checkpoint, "dataset": loaded_dataset,
                 "checkpoint_error": checkpoint_error,
                  'chat_format': getattr(generator, 'chat_format', None),
                   'features': dict(modern_architecture=True, memory_workspace=True, gguf_export=True, custom_corpus=True, huggingface=True, huggingface_export=True, text_preparation=True),
                "selection_persistent": state_file is not None, "training": jobs.training()}

    def require_checkpoint():
        if generator is None:
            raise HTTPException(409, 'No native checkpoint is loaded. Select a .pt checkpoint in Data & training to use the playground.')

    @app.post("/complete")
    def complete(request: CompletionRequest):
        with lock:
            require_checkpoint()
            if jobs.training():
                raise HTTPException(409, "Training or an automated pipeline has reserved the GPU. Stop the job or wait before generating.")
            generator.model.to(generator.device)
            return generator.complete(**request.model_dump())

    @app.post('/chat')
    def chat(request: ChatRequest):
        with lock:
            require_checkpoint()
            if jobs.training():
                raise HTTPException(409, 'Training or a pipeline has reserved the GPU')
            try:
                generator.model.to(generator.device)
                return generator.chat(**request.model_dump())
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.get('/architecture/presets')
    def presets():
        return hardware_presets()

    @app.post('/workspace/chat')
    def memory_chat(request: WorkspaceRequest):
        with lock:
            require_checkpoint()
            if jobs.training():
                raise HTTPException(409, 'Training or a pipeline has reserved the GPU')
            try:
                generator.model.to(generator.device)
                return workspace_chat(generator, memory, loaded_checkpoint, **request.model_dump())
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.get('/workspace/{session_id}')
    def memory_snapshot(session_id: str):
        try:
            return memory.snapshot(session_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete('/workspace/{session_id}')
    def forget_memory(session_id: str):
        with lock:
            try:
                memory.forget(session_id)
                return memory.snapshot(session_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.post('/workspace/{session_id}/feedback')
    def memory_feedback(session_id: str, request: FeedbackRequest):
        with lock:
            try:
                memory.feedback(session_id, **request.model_dump())
                return memory.snapshot(session_id)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc

    def start_job(kind, request):
        with lock:
            try:
                if kind in {"train", 'finetune', "pipeline", 'hf-finetune'} and not jobs.active():
                    if generator is not None:
                        generator.model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return jobs.start(kind, request)
            except (ValueError, OSError, KeyError, ImportError) as exc:
                raise HTTPException(400, str(exc)) from exc

    @app.post("/jobs/crawl")
    def crawl(request: CrawlRequest):
        return start_job("crawl", request)

    @app.post('/jobs/kiwix')
    def kiwix_download(request: KiwixRequest):
        return start_job('kiwix', request)

    @app.post('/jobs/zim-dataset')
    def zim_dataset(request: DatasetRequest):
        return start_job('dataset', request)

    @app.get('/exports/checkpoints')
    def export_checkpoints():
        return exports.export_checkpoints()

    @app.get('/exports')
    def export_status():
        return exports.status()

    @app.post('/exports')
    def export_model(request: ExportRequest):
        try:
            return exports.start('export', request)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/exports/stop')
    def stop_export():
        return exports.stop()

    @app.get('/exports/file')
    def download_export(job_id: str):
        job = exports.status().get('job')
        if (not job or job['id'] != job_id or job['status'] in Jobs.LIVE or not job.get('export_ready')):
            raise HTTPException(404, 'No completed GGUF for this export job')
        path = exports.path(job['output'])
        return FileResponse(path, media_type='application/octet-stream', filename=path.name)

    @app.get('/wikipedia/archives')
    def wikipedia_archives():
        return jobs.wikipedia_archives()

    @app.get('/wikipedia/catalog')
    def wikipedia_catalog():
        from .kiwix import catalog
        try:
            return {'archives': catalog()}
        except (OSError, ValueError) as exc:
            raise HTTPException(502, f'Cannot read Kiwix catalog: {exc}') from exc

    @app.post("/jobs/prepare")
    def prepare(request: PrepareRequest):
        return start_job("prepare", request)

    @app.post("/jobs/train")
    def train(request: TrainRequest):
        return start_job("train", request)

    @app.post("/jobs/pipeline")
    def pipeline(request: PipelineRequest):
        return start_job("pipeline", request)

    @app.post('/jobs/dialogue')
    def dialogue(request: DialogueRequest):
        return start_job('dialogue', request)

    @app.post('/corpus/upload')
    async def upload_corpus(request: Request, filename: str):
        from .custom_corpus import MAX_BYTES, UploadTooLarge, save_upload
        try:
            if int(request.headers.get('content-length', '0')) > MAX_BYTES:
                raise UploadTooLarge('Custom .txt files may be at most 64 MiB')
            return await save_upload(jobs.root, filename, request.stream())
        except UploadTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/corpus/uploads')
    def corpus_uploads():
        from .custom_corpus import MAX_BYTES, uploaded_files
        try:
            return dict(files=uploaded_files(jobs.root), max_bytes=MAX_BYTES)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/jobs/custom-corpus')
    def custom_corpus(request: CustomCorpusRequest):
        return start_job('custom-corpus', request)

    @app.get('/huggingface/status')
    def hf_status():
        from .huggingface import capabilities
        return capabilities(jobs.root)

    @app.post('/huggingface/repository')
    def hf_repository(request: HFRepositoryRequest):
        from .huggingface import repository_files
        try:
            return repository_files(request)
        except Exception as exc:
            raise HTTPException(400, f'Cannot list repository GGUF files: {exc}') from exc

    @app.get('/huggingface/assets')
    def hf_assets():
        from .huggingface import assets
        return assets(jobs.root)

    @app.post('/huggingface/plan')
    def hf_plan(request: HFTrainRequest):
        from .hf_train import training_preview
        try:
            if request.resume:
                raise ValueError('Preview is for a new run; Resume restores the saved budget')
            request = request.model_copy(update={key: str(jobs.path(getattr(request, key))) for key in ('model', 'data', 'out')})
            return training_preview(request)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/jobs/hf-import')
    def hf_import(request: HFImportRequest):
        return start_job('hf-import', request)

    @app.post('/jobs/hf-prepare')
    def hf_prepare(request: HFPrepareRequest):
        return start_job('hf-prepare', request)

    @app.post('/jobs/hf-finetune')
    def hf_finetune(request: HFTrainRequest):
        return start_job('hf-finetune', request)

    @app.post('/jobs/hf-merge')
    def hf_merge(request: HFMergeRequest):
        return start_job('hf-merge', request)

    @app.post('/jobs/hf-export')
    def hf_export(request: HFExportRequest):
        return start_job('hf-export', request)

    @app.get('/huggingface/gguf')
    def hf_gguf(job_id: str):
        job = jobs.status().get('job')
        if (not job or job['id'] != job_id or job['kind'] != 'hf-export'
                or job['status'] != 'succeeded' or not job.get('export_ready')):
            raise HTTPException(404, 'No completed GGUF for this export job')
        path = jobs.path(job['output'])
        return FileResponse(path, media_type='application/octet-stream', filename=path.name)

    @app.get('/huggingface/adapter')
    def hf_adapter(run: str):
        from .hf_train import adapter_archive
        with jobs.lock:
            temporary = None
            try:
                path = jobs.path(run)
                if jobs.active() and jobs.job['kind'] == 'hf-finetune' and jobs.path(jobs.job['output']) == path:
                    raise HTTPException(409, 'Wait for this fine-tuning run to stop before downloading its adapter')
                record = json.loads((path/'hf-run.json').read_text())
                jobs.path(record['request']['model'])
                directory = jobs.path('runs/hf-downloads')
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=directory, suffix='.zip', delete=False) as stream:
                    temporary = Path(stream.name)
                adapter_archive(path, temporary)
                return FileResponse(temporary, media_type='application/zip', filename=path.name + '-adapter.zip',
                                    background=BackgroundTask(temporary.unlink, missing_ok=True))
            except (ValueError, OSError, KeyError) as exc:
                if temporary:
                    temporary.unlink(missing_ok=True)
                raise HTTPException(400, str(exc)) from exc

    @app.post('/jobs/finetune')
    def finetune(request: FineTuneRequest):
        return start_job('finetune', request)

    @app.post('/finetuning/plan')
    def finetuning_preview(request: FineTuneRequest):
        try:
            return jobs.preview_training(request)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/training/plan")
    def training_preview(request: TrainRequest):
        try:
            return jobs.preview_training(request)
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/datasets")
    def datasets():
        return jobs.prepared_datasets()

    @app.post('/corpus/inspect')
    def inspect_corpus(request: PrepareRequest):
        try:
            return jobs.inspect_corpus(request)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/jobs")
    def status():
        return jobs.status()

    @app.get("/jobs/browser", include_in_schema=False)
    def browser_snapshot(job_id: str):
        path = jobs.browser_snapshot_path(job_id)
        if path is None:
            raise HTTPException(404, "No browser snapshot is available for this job")
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/jobs/stop")
    def stop():
        return jobs.stop()

    @app.post("/model/load")
    def load(request: LoadRequest):
        nonlocal generator, loaded_checkpoint, loaded_dataset, checkpoint_error
        with lock:
            if jobs.active():
                raise HTTPException(409, "Wait for the current job to finish before loading a checkpoint")
            try:
                path = jobs.path(request.checkpoint)
                candidate = Generator(path, "cpu")
                if generator is not None:
                    generator.model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                candidate.device = torch.device(device)
                candidate.model.to(candidate.device)
                selected = str(path.relative_to(jobs.root))
                if state_file:
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = state_file.with_suffix(".tmp")
                    temporary.write_text(json.dumps({"checkpoint": selected}, indent=2) + "\n")
                    temporary.replace(state_file)
                generator = candidate
                loaded_checkpoint = selected
                loaded_dataset = dataset_for(path)
                checkpoint_error = None
            except Exception as exc:
                raise HTTPException(400, f"Cannot load checkpoint: {exc}") from exc
            return {"checkpoint": loaded_checkpoint, "dataset": loaded_dataset, "model": generator.model.config_dict()}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="runs/tiny/best.pt")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--state", help="Workspace-local JSON file that remembers the selected checkpoint across restarts")
    args = p.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.device, state_path=args.state), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
