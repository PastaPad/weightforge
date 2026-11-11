"""Exercise real pipeline supervision using file-only workers; never crawl or train."""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event


def driver(mode, arguments):
    from scratchllm import jobs, pipeline
    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        module = command[3]
        if module not in {"scratchllm.crawl", "scratchllm.prepare", "scratchllm.train", 'scratchllm.dialogue'}:
            raise AssertionError(f"Unexpected worker: {command}")
        # Only this fixture script can be launched, including for the training stage.
        kind = 'finetune' if '--init-from' in command else module.rsplit('.', 1)[1]
        return real_popen([sys.executable, "-u", __file__, "stage", mode, kind, *command[4:]], **kwargs)

    jobs.subprocess.Popen = spawn
    def fixture_finetune_config(self, request, meta):
        assert self.path(request.checkpoint).read_text() == 'fixture best marker'
        assert meta['format'] == 'dialogue-sft-v1'
        config = request.model_dump(exclude={'checkpoint', 'data', 'out', 'resume'})
        config.update({key: value for key, value in meta['base_model_config'].items() if key != 'vocab_size'})
        return config
    jobs.Jobs.finetune_config = fixture_finetune_config
    sys.argv = ["scratchllm.pipeline", *arguments]
    pipeline.main()


def stage(mode, kind, arguments):
    stopping = Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())

    def value(flag):
        return arguments[arguments.index(flag) + 1]

    def write(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    with Path('fixture-events.jsonl').open('a') as events:
        events.write(json.dumps(dict(kind=kind, arguments=arguments)) + '\n')
    print(f'FIXTURE READY {kind}', flush=True)
    if kind == 'crawl':
        write(value('--output'), {'text': 'Synthetic fixture text; no network access.'})
        write(value('--progress-file'), dict(phase='fetching', saved=3, completed=3, budget=3))
        Path(value('--progress-file')).with_suffix('.browser.png').write_bytes(b'fixture-screenshot')
    if mode == f'wait-{kind}':
        if not stopping.wait(40):
            raise SystemExit('Test did not stop the waiting worker')
    else:
        time.sleep(0.25)
    if mode == f'fail-{kind}':
        raise SystemExit(17)
    if kind == 'crawl':
        phase = 'stopped' if stopping.is_set() or mode == 'browser-closed' else 'completed'
        write(value('--progress-file'), dict(phase=phase, saved=3, completed=3, budget=3))
    elif kind == 'prepare' and not stopping.is_set():
        assert Path(value('--input')).is_file()
        output = Path(value('--output'))
        meta = dict(vocab_size=16384, tokens=dict(train=1587273, val=594931), documents=dict(train=937, val=106))
        from scratchllm.links import TEXT_POLICY_VERSION
        meta['text_policy'] = TEXT_POLICY_VERSION
        report = dict(protocol='grouped-text-v1', accepted_records=1043, groups=100, duplicates_removed=7,
                      too_short_removed=0, documents=meta['documents'], validation_character_fraction=0.1,
                      sources={'soyjak.st': dict(accepted_records=1043, characters=5000000, samples=[], tokens=meta['tokens'])})
        meta['corpus_report'] = report
        if mode == 'small-data':
            meta['tokens'] = dict(train=4, val=4)
        write(output/'meta.json', meta)
        write(output/'report.json', report)
        write(output/'tokenizer.json', {})
        for split, count in meta['tokens'].items():
            with (output/f'{split}.bin').open('wb') as stream:
                stream.truncate(4 * count)
    elif kind == 'dialogue' and not stopping.is_set():
        from scratchllm.links import TEXT_POLICY_VERSION
        checkpoint = Path(value('--checkpoint'))
        assert checkpoint.read_text() == 'fixture best marker'
        base_config = json.loads((checkpoint.parent/'web-config.json').read_text())
        report = dict(groups=8, duplicates_removed=1, over_context_removed=2, sources={
            'oasst1': dict(documents=dict(train=32, val=4), supervised_tokens=dict(train=6000, val=600))})
        meta = dict(format='dialogue-sft-v1', chat_format='user-assistant-v1', text_policy=TEXT_POLICY_VERSION,
                    vocab_size=16384, tokens=dict(train=9000, val=1000), documents=dict(train=32, val=4),
                    supervised_tokens=dict(train=6000, val=600), dialogue_report=report,
                    base_model_config=dict(vocab_size=16384, **{key: base_config[key] for key in ('context', 'width', 'layers', 'heads', 'dropout')}))
        output = Path(value('--output'))
        write(output/'meta.json', meta)
        write(output/'report.json', report)
        write(output/'tokenizer.json', {})
        for split, count in meta['tokens'].items():
            with (output/f'{split}.bin').open('wb') as stream:
                stream.truncate(4 * count)
            with (output/f'{split}.mask.bin').open('wb') as stream:
                stream.truncate(count)
            write(output/f'{split}.windows.json', [])
    elif kind in {'train', 'finetune'}:
        assert Path(value('--data'), 'meta.json').is_file()
        config = json.loads(Path(value('--config')).read_text())
        output = Path(value('--out'))
        if kind == 'finetune':
            assert Path(value('--init-from')).read_text() == 'fixture best marker'
        # These are dummy text files, never model weights or optimizer state.
        (output/'last.pt').write_text('fixture resume marker')
        if mode != 'missing-best':
            (output/'best.pt').write_text('fixture best marker')
        metric = dict(step=3, planned_steps=config['steps'], train_loss=1.5, train_eval_loss=1.5,
                      val_loss=2.0, perplexity=7.39, generalization_gap=0.5, epochs=0.1,
                      best_val_loss=1.9, best_step=2, selected_weights='raw',
                      stop_reason='user_stop' if stopping.is_set() else 'early_stopping')
        (output/'metrics.jsonl').write_text(json.dumps(metric) + '\n')
    print(f'FIXTURE DONE {kind}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'driver':
        driver(sys.argv[2], sys.argv[3:])
    else:
        stage(sys.argv[2], sys.argv[3], sys.argv[4:])
