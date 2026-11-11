"""CPU-only, atomic GGUF export of complete imported/merged Hugging Face models."""
import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Event

import gguf
import numpy as np

from .frontier import corpus_lock
from .hf_requests import HFExportRequest
from .huggingface import (Stopped, check_resources, check_stop, field, inspect_gguf,
                          load_import, sha256, tokenizer_from_gguf)
from .progress import CrawlProgress

CONVERTER_REVISION = 'b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9'


def converter_status():
    directory = Path(os.environ.get('SCRATCHLLM_LLAMA_CPP', Path(__file__).resolve().parent.parent/'tools/llama.cpp')).resolve()
    required = ('convert_hf_to_gguf.py', 'conversion/__init__.py', 'gguf-py/gguf/__init__.py')
    missing = [name for name in required if not (directory/name).is_file()]
    if importlib.util.find_spec('sentencepiece') is None:
        missing.append('Python package sentencepiece')
    return dict(ready=not missing, directory=str(directory), missing=missing, tested_revision=CONVERTER_REVISION)


def validate_paths(model, output):
    model, output = Path(model).resolve(), Path(output)
    if output.suffix.lower() != '.gguf':
        raise ValueError('Export output must end in .gguf')
    if output.exists() or output.is_symlink():
        raise ValueError('Choose a fresh GGUF output file')
    output = output.resolve()
    if output.is_relative_to(model) or model.is_relative_to(output):
        raise ValueError('Keep the GGUF output separate from the Hugging Face model directory')
    load_import(model)
    return model, output


def verify_source(model, manifest, cancel):
    for name, expected in manifest['files'].items():
        check_stop(cancel)
        if sha256(model/name, cancel) != expected:
            raise ValueError('Imported model file changed: ' + name)


def stage_model(model, destination, manifest):
    from transformers import AutoTokenizer
    destination.mkdir()
    for name in manifest['files']:
        source, target = model/name, destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.safetensors':
            # The converter only reads weights; the source is hashed before and after conversion.
            target.symlink_to(source)
        else:
            shutil.copyfile(source, target)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    wrapped = tokenizer.encode('', add_special_tokens=True)
    before = [tokenizer.bos_token_id] if manifest['model']['add_bos'] else []
    add_eos = wrapped == before + [tokenizer.eos_token_id]
    if wrapped != before + ([tokenizer.eos_token_id] if add_eos else []):
        raise ValueError('Unsupported automatic special-token policy in imported tokenizer')
    config_path = destination/'tokenizer_config.json'
    config = json.loads(config_path.read_text())
    config.update(add_bos_token=bool(before), add_eos_token=add_eos)
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    return tokenizer


def run_converter(directory, model, output, precision, progress, cancel):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
    env.pop('NO_LOCAL_GGUF', None)
    command = [sys.executable, '-u', str(directory/'convert_hf_to_gguf.py'), str(model),
               '--outfile', str(output), '--outtype', precision]
    process = subprocess.Popen(command, env=env)
    try:
        while True:
            check_stop(cancel)
            try:
                code = process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                progress.update(bytes_written=output.stat().st_size if output.exists() else 0)
        check_stop(cancel)
        if code:
            raise ValueError(f'llama.cpp converter exited {code}; see the job log for the architecture/tokenizer error')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def verify_gguf(path, manifest, tokenizer, cancel):
    info = inspect_gguf(path)
    if any(info[key] != manifest['model'][key] for key in ('architecture', 'context', 'vocabulary', 'bos_id', 'eos_id', 'add_bos')):
        raise ValueError('Converted GGUF architecture, context or token metadata differs from the imported model')
    reader = gguf.GGUFReader(path)
    restored = tokenizer_from_gguf(path, info)
    if restored.get_vocab() != tokenizer.get_vocab():
        raise ValueError('Converted GGUF did not preserve the tokenizer vocabulary IDs')
    for name in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id'):
        if getattr(restored, name) != getattr(tokenizer, name):
            raise ValueError('Converted GGUF changed ' + name)
    if field(reader, 'tokenizer.chat_template') != tokenizer.chat_template:
        raise ValueError('Converted GGUF did not preserve the chat template')
    for text in ('', 'Hello world!', ' café\nעברית 中文 12345', '<|im_start|>user\nHello<|im_end|>'):
        if restored.encode(text) != tokenizer.encode(text):
            raise ValueError('Converted GGUF tokenizer behavior differs from the imported tokenizer')
    for tensor in reader.tensors:
        check_stop(cancel)
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}:
            data = tensor.data.reshape(-1)
            for offset in range(0, data.size, 1024 * 1024):
                check_stop(cancel)
                if not np.isfinite(data[offset:offset + 1024 * 1024]).all():
                    raise ValueError('Converted tensor is non-finite or overflows F16; choose F32: ' + tensor.name)
    return info, len(reader.tensors)


def export_model(request, progress=None, cancel=None):
    progress = progress or CrawlProgress(None, 0)
    cancel = cancel or Event()
    model, output = validate_paths(request.model, request.output)
    converter = converter_status()
    if not converter['ready']:
        raise ValueError('Install the llama.cpp converter described in HUGGINGFACE.md: ' + ', '.join(converter['missing']))
    directory = Path(converter['directory'])
    with corpus_lock(model), corpus_lock(output):
        manifest = load_import(model)
        progress.update(phase='verifying_source', detail='Verifying complete Hugging Face model weights and tokenizer')
        verify_source(model, manifest, cancel)
        manifest_hash = sha256(model/'import.json', cancel)
        output.parent.mkdir(parents=True, exist_ok=True)
        size = manifest['model']['parameters'] * (4 if request.precision == 'f32' else 2)
        check_resources(disk_bytes=size + 128 * 1024**2, directory=output.parent)
        with tempfile.TemporaryDirectory(prefix='.hf-export-', dir=output.parent) as name:
            temporary = Path(name)
            tokenizer = stage_model(model, temporary/'model', manifest)
            check_stop(cancel)
            progress.update(phase='converting', detail=f'Converting merged/imported weights to {request.precision.upper()} GGUF on CPU',
                            model=str(model), precision=request.precision)
            result = temporary/'model.gguf'
            run_converter(directory, temporary/'model', result, request.precision, progress, cancel)
            progress.update(phase='verifying', detail='Checking GGUF tensors, tokenizer IDs and chat template')
            info, tensors = verify_gguf(result, manifest, tokenizer, cancel)
            verify_source(model, manifest, cancel)
            if sha256(model/'import.json', cancel) != manifest_hash:
                raise ValueError('Model manifest changed during conversion')
            digest = sha256(result, cancel)
            with result.open('rb') as stream:
                os.fsync(stream.fileno())
            check_stop(cancel)
            os.link(result, output)
        report = dict(output=str(output), bytes=output.stat().st_size, sha256=digest, precision=request.precision,
                      architecture=info['architecture'], tensors=tensors, artifact_id=manifest['artifact_id'],
                      converter_sha256=sha256(directory/'convert_hf_to_gguf.py'))
        progress.update(phase='completed', detail=f'GGUF READY {output}', **report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', choices=['f16', 'f32', 'q8_0'], default='f16')
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    cancel = Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    progress = CrawlProgress(args.progress_file, 0)
    try:
        export_model(HFExportRequest(model=args.model, output=args.output, precision=args.precision), progress, cancel)
    except Stopped:
        progress.update(phase='stopped', detail='GGUF export stopped; start export again to retry')
        raise SystemExit(130)
    except Exception as exc:
        progress.update(phase='failed', detail=str(exc))
        raise


if __name__ == '__main__':
    main()
