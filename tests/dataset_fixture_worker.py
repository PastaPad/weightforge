"""Real data-only workers with a hard guard against launching model training."""
import json
import subprocess
import sys
import time
from pathlib import Path


def driver(mode, arguments):
    from scratchllm import jobs, pipeline
    real_popen = subprocess.Popen
    def spawn(command, **kwargs):
        module = command[3]
        if module not in {'scratchllm.crawl', 'scratchllm.prepare'}:
            raise AssertionError(f'Dataset build attempted a non-data worker: {module}')
        with Path('dataset-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(module=module, arguments=command[4:])) + '\n')
        return real_popen([sys.executable, '-u', __file__, 'worker', mode, module, *command[4:]], **kwargs)
    jobs.subprocess.Popen = spawn
    sys.argv = ['scratchllm.pipeline', *arguments]
    pipeline.main()


def worker(mode, module, arguments):
    sys.argv = [module, *arguments]
    if module == 'scratchllm.crawl':
        from scratchllm import crawl, kiwix
        original = kiwix.article_entry
        if mode == 'slow-crawl':
            def article(*args):
                time.sleep(0.1)
                return original(*args)
            kiwix.article_entry = article
        crawl.main()
    elif module == 'scratchllm.prepare':
        from scratchllm import prepare
        if mode == 'wait-prepare':
            def wait(args):
                (Path(args.output)/'partial.bin').write_bytes(b'incomplete dataset')
                print('FIXTURE READY atomic preparation', flush=True)
                time.sleep(60)
                raise AssertionError('Test did not stop preparation')
            prepare.prepare = wait
        prepare.main()
    else:
        raise AssertionError('Unexpected worker')


if __name__ == '__main__':
    if sys.argv[1] == 'driver':
        driver(sys.argv[2], sys.argv[3:])
    else:
        worker(sys.argv[2], sys.argv[3], sys.argv[4:])
