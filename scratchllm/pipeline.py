"""Server-side dataset and training pipelines, with fail-fast stages and cancellation."""
import argparse
import json
import signal
import time
from pathlib import Path
from threading import Event

from .jobs import Jobs, PipelineRequest, DatasetRequest


def run_pipeline(request, root, device, progress_file, stage_state, stopping):
    root = Path(root).resolve()
    progress_file = Path(progress_file)
    stages = request.stages()
    dataset_only = isinstance(request, DatasetRequest)
    outputs = dict(corpus=request.crawl.output, dataset=request.prepare.output)
    if not dataset_only:
        final_run = request.finetune.out if request.dialogue else request.train.out
        outputs.update(run=final_run, checkpoint=str(Path(final_run) / 'best.pt'))
    snapshot = dict(phase="running", stage=None, stage_index=0,
                    stages=[dict(kind=kind, status="pending") for kind, _ in stages],
                    outputs=outputs,
                    started_at=time.time(), current=None)
    jobs = Jobs(root, device, state_path=stage_state)
    if not dataset_only and request.dialogue:
        snapshot['outputs'].update(base_run=request.train.out, dialogue_dataset=request.dialogue.output)

    def save(detail=None):
        if detail:
            snapshot["detail"] = detail
            print(detail, flush=True)
        snapshot["updated_at"] = time.time()
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(progress_file)

    def finish(phase, detail):
        snapshot.update(phase=phase, finished_at=time.time())
        for stage in snapshot["stages"]:
            if stage["status"] == "pending":
                stage["status"] = "cancelled"
        save(detail)

    def forward_log(stream):
        text = stream.read()
        if text:
            print(text, end="", flush=True)

    try:
        save('PIPELINE queued: ' + ' → '.join(kind for kind, _ in stages))
        for index, (kind, settings) in enumerate(stages):
            if stopping.is_set():
                finish("stopped", "PIPELINE stopped; remaining stages cancelled")
                return 0
            stage = snapshot["stages"][index]
            snapshot.update(stage=kind, stage_index=index + 1, current=None)
            stage.update(status="running", started_at=time.time())
            save(f"PIPELINE STAGE {index + 1}/{len(stages)}: {kind}")
            current = jobs.start(kind, settings)["job"]
            stage.update(job_id=current["id"], output=current["output"], log_path=current["log_path"])
            if kind == "crawl":
                snapshot["crawl_progress_path"] = current["progress_path"]
            # Each stage keeps its own log; stream it into the top-level pipeline log too.
            with (root / current["log_path"]).open() as log:
                while True:
                    if stopping.is_set():
                        jobs.stop()
                    current = jobs.status()["job"]
                    forward_log(log)
                    snapshot["current"] = {key: value for key, value in current.items() if key != "log"}
                    save()
                    if current["status"] not in Jobs.LIVE:
                        break
                    time.sleep(0.2)
                forward_log(log)
            stage.update(status=current["status"], returncode=current["returncode"], finished_at=time.time())
            if kind == 'prepare' and current.get('corpus_report'):
                snapshot['corpus_report'] = current['corpus_report']
            if kind == 'dialogue' and current.get('dialogue_report'):
                snapshot['dialogue_report'] = current['dialogue_report']
            if current["status"] != "succeeded" or stopping.is_set():
                stopped = stopping.is_set() or current["status"] == "stopped"
                finish("stopped" if stopped else "failed",
                       f"PIPELINE {'stopped' if stopped else 'failed'} during {kind}; remaining stages cancelled")
                return 0 if stopped else 1
            save(f"PIPELINE STAGE {index + 1}/{len(stages)} complete: {kind}")
        if dataset_only:
            dataset = root / request.prepare.output
            meta = json.loads((dataset / 'meta.json').read_text())
            if not (dataset / 'tokenizer.json').is_file() or any(
                    meta['tokens'][split] < 2 or (dataset / f'{split}.bin').stat().st_size != 4 * meta['tokens'][split]
                    for split in ('train', 'val')):
                raise RuntimeError('Preparation exited without a complete dataset')
            finish('completed', f'DATASET DONE: {request.prepare.output}')
            return 0
        checkpoint = root / snapshot["outputs"]["checkpoint"]
        if not checkpoint.is_file():
            raise RuntimeError("Training exited without producing best.pt")
        finish("completed", f"PIPELINE DONE: best checkpoint is {snapshot['outputs']['checkpoint']}")
        return 0
    except Exception as exc:
        if snapshot["stage_index"]:
            snapshot["stages"][snapshot["stage_index"] - 1].update(status="failed", error=str(exc))
        finish("failed", f"PIPELINE ERROR: {exc}; remaining stages cancelled")
        return 1
    finally:
        jobs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Resolved pipeline request JSON")
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--stage-state", required=True)
    parser.add_argument('--dataset-only', action='store_true', help='Import the whole ZIM and prepare its dataset')
    args = parser.parse_args()
    request = (DatasetRequest if args.dataset_only else PipelineRequest).model_validate_json(Path(args.request).read_text())
    stopping = Event()
    def stop(*_):
        stopping.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    raise SystemExit(run_pipeline(request, args.root, args.device, args.progress_file, args.stage_state, stopping))


if __name__ == "__main__":
    main()
