"""Pinned GGUF downloads, trainable HF conversion, and scraper-corpus preparation."""
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path

import gguf
import numpy as np
import psutil

from .corpus import corpus_plan, prepared_source_report, records, validate_extraction
from .frontier import corpus_lock
from .hf_requests import HFImportRequest, HFPrepareRequest, HFMergeRequest
from .links import TEXT_POLICY_VERSION
from .prepare import validate_corpus
from .progress import CrawlProgress

IMPORT_FORMAT = 'huggingface-gguf-import-v1'
DATA_FORMAT = 'huggingface-corpus-v1'
SUPPORTED_ARCHITECTURES = ('llama', 'mistral', 'qwen2', 'qwen3', 'phi3', 'gpt2', 'gemma2',
                           'gemma3', 'falcon', 'bloom', 'stablelm', 'starcoder2', 'nemotron')
SHARD = re.compile(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$')


class Stopped(BaseException):
    pass


def check_stop(cancel=None):
    if cancel is not None and cancel.is_set():
        raise Stopped()


def sha256(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            check_stop(cancel)
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def capabilities(root):
    from .hf_export import converter_status
    packages = ('transformers', 'peft', 'accelerate', 'bitsandbytes', 'huggingface_hub', 'filelock', 'safetensors')
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    cuda = torch.cuda.is_available()
    return dict(ready=all(versions.values()), versions=versions, architectures=list(SUPPORTED_ARCHITECTURES),
                gguf_export=converter_status(),
                ram_available=psutil.virtual_memory().available, disk_free=shutil.disk_usage(root).free,
                cuda=cuda, gpu=torch.cuda.get_device_name(0) if cuda else None,
                vram_bytes=torch.cuda.get_device_properties(0).total_memory if cuda else 0,
                compute_capability=list(torch.cuda.get_device_capability(0)) if cuda else None,
                install=".venv/bin/python -m pip install -e '.[huggingface]'")


def require_dependencies():
    if any(importlib.util.find_spec(name) is None for name in ('transformers', 'peft', 'accelerate', 'safetensors', 'huggingface_hub', 'filelock')):
        raise ValueError("Install the Hugging Face extra: .venv/bin/python -m pip install -e '.[huggingface]'")


def repository_files(request):
    from huggingface_hub import HfApi
    info = HfApi().model_info(request.repo, revision=request.revision, files_metadata=True, timeout=30)
    files = {}
    for sibling in info.siblings or []:
        if not sibling.rfilename.lower().endswith('.gguf'):
            continue
        # Reuse filename validation before constructing any download request.
        HFImportRequest(repo=request.repo, filename=sibling.rfilename)
        lfs = sibling.lfs
        digest = getattr(lfs, 'sha256', None) if lfs else None
        if isinstance(lfs, dict):
            digest = lfs.get('sha256')
        files[sibling.rfilename] = dict(filename=sibling.rfilename, bytes=sibling.size, sha256=digest)
    groups = []
    for name, row in sorted(files.items()):
        match = SHARD.fullmatch(name)
        if match:
            prefix, index, count = match.groups()
            if int(index) != 1:
                continue
            names = [f'{prefix}-{index:05d}-of-{int(count):05d}.gguf' for index in range(1, int(count) + 1)]
            if any(name not in files for name in names):
                continue
            parts = [files[name] for name in names]
        else:
            parts = [row]
        if not all(type(part['bytes']) is int and part['bytes'] > 0 for part in parts):
            continue
        groups.append(dict(filename=name, bytes=sum(part['bytes'] for part in parts), parts=parts))
    card = getattr(info, 'card_data', None)
    return dict(repo=request.repo, revision=info.sha, files=groups,
                license=card.get('license') if card else None, gated=bool(info.gated))


def field(reader, name, default=None):
    item = reader.fields.get(name)
    return item.contents() if item is not None else default


def special_id(reader, key, vocabulary):
    value = field(reader, 'tokenizer.ggml.' + key + '_token_id')
    # GGUF's unsigned -1 sentinel means the tokenizer has no such token.
    if value in (None, -1, 2**32 - 1):
        return None
    if not isinstance(value, int) or not 0 <= value < vocabulary:
        raise ValueError('GGUF special-token ID is outside its vocabulary')
    return value


def inspect_gguf(path):
    reader = gguf.GGUFReader(path)
    architecture = field(reader, 'general.architecture')
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f'GGUF architecture {architecture!r} is not supported by this text fine-tuning importer. '
                         + 'Supported: ' + ', '.join(SUPPORTED_ARCHITECTURES))
    if field(reader, 'split.count', 1) != 1:
        raise ValueError('Import the complete GGUF shard set, beginning with its first shard')
    vocab = field(reader, 'tokenizer.ggml.tokens', [])
    context = field(reader, architecture + '.context_length', 0)
    if not vocab or not isinstance(context, int) or context < 8 or not reader.tensors:
        raise ValueError('GGUF must contain a vocabulary, context length, and model tensors')
    return dict(architecture=architecture, context=context, vocabulary=len(vocab),
                parameters=sum(int(t.n_elements) for t in reader.tensors),
                quantizations=sorted({t.tensor_type.name for t in reader.tensors}),
                bos_id=special_id(reader, 'bos', len(vocab)), eos_id=special_id(reader, 'eos', len(vocab)),
                add_bos=bool(field(reader, 'tokenizer.ggml.add_bos_token', architecture in {'llama', 'mistral', 'gemma2', 'gemma3', 'phi3'})))


def merge_shards(paths, output, progress=None, cancel=None):
    """Copy packed blocks without dequantizing; retain typed metadata, remove split keys."""
    readers = [gguf.GGUFReader(path) for path in paths]
    first = readers[0]
    if first.endianess != gguf.GGUFEndian.LITTLE or any(reader.endianess != first.endianess for reader in readers):
        raise ValueError('GGUF shard merging requires little-endian files')
    total = field(first, 'split.tensors.count')
    for index, reader in enumerate(readers):
        if field(reader, 'split.no') != index or field(reader, 'split.count') != len(paths):
            raise ValueError('GGUF shard numbering/count is incomplete or inconsistent')
        if field(reader, 'split.tensors.count') != total:
            raise ValueError('GGUF shards disagree about the total tensor count')
    tensors = [tensor for reader in readers for tensor in reader.tensors]
    if len(tensors) != total or len({t.name for t in tensors}) != len(tensors):
        raise ValueError('GGUF shard set has missing or duplicate tensors')
    values = {}
    writer = gguf.GGUFWriter(output, field(first, 'general.architecture'))
    try:
        for reader in readers:
            for name, item in reader.fields.items():
                if name.startswith(('GGUF.', 'split.')):
                    continue
                value = item.contents()
                if name in values and values[name] != value:
                    raise ValueError('Conflicting GGUF shard metadata: ' + name)
                if name not in values and name != 'general.architecture':
                    writer.add_key_value(name, value, item.types[0], item.types[-1] if len(item.types) > 1 else None)
                values[name] = value
        for tensor in tensors:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, tensor in enumerate(tensors, 1):
            check_stop(cancel)
            writer.write_tensor_data(tensor.data)
            if progress:
                progress.update(completed=index, budget=len(tensors))
        writer.flush()
    finally:
        writer.close()


def import_memory(info, precision):
    # Conservative CPU conversion allowance: dequantized tensors, model, and temporary copies.
    return dict(ram_bytes=8 * info['parameters'] + 512 * 1024**2,
                disk_bytes=info['parameters'] * (2 if precision == 'f16' else 4) + 128 * 1024**2)


def check_resources(ram_bytes=0, disk_bytes=0, directory='.'):
    ram, disk = psutil.virtual_memory().available, shutil.disk_usage(directory).free
    if ram_bytes > ram:
        raise ValueError(f'Estimated operation RAM {ram_bytes / 1024**3:.2f} GiB exceeds {ram / 1024**3:.2f} GiB available; choose a smaller model')
    if disk_bytes > max(0, disk - 128 * 1024**2):
        raise ValueError(f'Operation needs about {disk_bytes / 1024**3:.2f} GiB of free disk; {disk / 1024**3:.2f} GiB is available')


def tokenizer_from_gguf(path, info):
    from tokenizers import Tokenizer, processors
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from transformers.integrations.gguf import GGUF_TOKENIZER_MAPPING, convert_gguf_tokenizer
    reader = gguf.GGUFReader(path)
    tokens = field(reader, 'tokenizer.ggml.tokens')
    # Read complete score/type arrays (needed to reconstruct SentencePiece merges),
    # and normalize absent-token sentinels before the upstream builder indexes them.
    descriptor = {name: field(reader, 'tokenizer.' + key) for key, name in GGUF_TOKENIZER_MAPPING['tokenizer'].items()
                  if 'tokenizer.' + key in reader.fields}
    special = {}
    for name, key in [('bos', 'bos'), ('eos', 'eos'), ('pad', 'padding'), ('unk', 'unknown')]:
        index = special_id(reader, key, len(tokens))
        descriptor[name + '_token_id'] = index
        special[name + '_token'] = tokens[index] if index is not None else None
    backend, _ = convert_gguf_tokenizer(info['architecture'], descriptor)
    raw = json.loads(backend.to_str())
    # Some builders append their own default <|endoftext|>/<unk>. These are not model IDs.
    raw['added_tokens'] = [token for token in raw.get('added_tokens', []) if token['id'] < len(tokens)]
    def prefix_space(value):
        if isinstance(value, dict):
            if value.get('type') == 'Metaspace' and 'add_prefix_space' in descriptor:
                value['prepend_scheme'] = 'always' if descriptor['add_prefix_space'] else 'never'
            for child in value.values():
                prefix_space(child)
        elif isinstance(value, list):
            for child in value:
                prefix_space(child)
    prefix_space(raw)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(raw)), **special,
                                        chat_template=field(reader, 'tokenizer.chat_template'),
                                        model_max_length=info['context'], clean_up_tokenization_spaces=False)
    spellings = tokens
    if field(reader, 'tokenizer.ggml.model') == 'gpt2':
        spelling = bytes_to_unicode()
        alphabet = set(spelling.values())
        spellings = [token if set(token) <= alphabet else ''.join(spelling[b] for b in token.encode()) for token in tokens]
    elif info['architecture'].startswith('gemma'):
        spellings = ['▁' * len(token) if ' ' in token and not token.strip() else token for token in tokens]
    if len(tokenizer) != len(tokens) or any(tokenizer.convert_tokens_to_ids(token) != i for i, token in enumerate(spellings)):
        raise ValueError('GGUF tokenizer conversion did not preserve every vocabulary ID; this tokenizer is unsupported')
    if tokenizer.eos_token_id is None:
        raise ValueError('A GGUF EOS token is required to separate scraper documents')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Existing ID; padded targets are ignored.
    # Automatic special-token insertion follows the GGUF policy; corpus packing
    # calls encode(add_special_tokens=False) and places its own document boundaries.
    before = [tokenizer.bos_token] if info['add_bos'] and tokenizer.bos_token_id is not None else []
    after = [tokenizer.eos_token] if field(reader, 'tokenizer.ggml.add_eos_token', False) else []
    auto_tokens = [(token, tokenizer.convert_tokens_to_ids(token)) for token in dict.fromkeys(before + after)]
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single=before + ['$A'] + after, pair=before + ['$A'] + after + ['$B:1'] + after, special_tokens=auto_tokens)
    return tokenizer


def convert_gguf(path, output, source, precision='f16', progress=None, cancel=None):
    """Materialize a regular HF checkpoint atomically; no optimizer or model fitting."""
    require_dependencies()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GgufConfig
    output = Path(output)
    if output.resolve() == Path(path).resolve() or Path(path).resolve().is_relative_to(output.resolve()):
        raise ValueError('Keep GGUF input and imported output separate')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a fresh imported-model directory')
    info = inspect_gguf(path)
    resources = import_memory(info, precision)
    output.parent.mkdir(parents=True, exist_ok=True)
    check_resources(**resources, directory=output.parent)
    if progress:
        progress.update(phase='converting', detail='Reconstructing trainable weights and the original tokenizer on CPU',
                        architecture=info['architecture'], parameters=info['parameters'], resources=resources, budget=0, completed=0)
    temporary = Path(tempfile.mkdtemp(prefix='.hf-import-', dir=output.parent))
    try:
        tokenizer = tokenizer_from_gguf(path, info)
        check_stop(cancel)
        config = AutoConfig.from_pretrained(Path(path).parent, gguf_file=Path(path).name,
                                            local_files_only=True, trust_remote_code=False)
        for name in ('bos', 'eos', 'pad'):
            setattr(config, name + '_token_id', getattr(tokenizer, name + '_token_id'))
        if info['architecture'] == 'gpt2':
            # The upstream legacy GGUF mapper calls these n_ctx/feed_forward_length;
            # GPT2Config actually consumes n_positions/n_inner when reloaded.
            config.n_positions = info['context']
            config.n_inner = field(gguf.GGUFReader(path), 'gpt2.feed_forward_length', 4 * config.n_embd)
        model, loading = AutoModelForCausalLM.from_pretrained(Path(path).parent, gguf_file=Path(path).name,
            config=config,
            local_files_only=True, trust_remote_code=False, dtype=torch.float16 if precision == 'f16' else torch.float32,
            attn_implementation='eager', output_loading_info=True, quantization_config=GgufConfig(dequantize=True))
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Incomplete GGUF tensor conversion: ' + str(loading))
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            raise ValueError('Converted model/tokenizer vocabulary mismatch')
        # Transformers 5.20 retains its non-serializable GGUF quantizer even after
        # full dequantization. Detach that loader only after checking no packed
        # modules/weights remain; the published checkpoint is ordinary tensors.
        quantizer = getattr(model, 'hf_quantizer', None)
        if (quantizer is not None and getattr(quantizer, 'packed_modules', {})) or any(
                not parameter.is_floating_point() for parameter in model.parameters()):
            raise ValueError('GGUF loader retained packed weights instead of trainable tensors')
        if quantizer is not None:
            quantizer.remove_quantization_config(model)
        with torch.device('meta'):
            expected = AutoModelForCausalLM.from_config(copy.deepcopy(model.config), trust_remote_code=False)
        expected_shapes = {key: tensor.shape for key, tensor in expected.state_dict().items()}
        if expected_shapes != {key: tensor.shape for key, tensor in model.state_dict().items()}:
            raise ValueError('Converted GGUF weights do not match their saved architecture configuration')
        del expected
        for parameter in model.parameters():
            check_stop(cancel)
            if any(not torch.isfinite(block).all() for block in parameter.detach().reshape(-1).split(1024 * 1024)):
                raise ValueError('Converted weights are non-finite; choose F32 if F16 overflowed')
        # Bounded inference validates the conversion's forward path, without modifying weights.
        model.eval()
        with torch.inference_mode():
            probe = torch.tensor([[tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]])
            if not torch.isfinite(model(probe, use_cache=False).logits).all():
                raise ValueError('Imported model produced non-finite logits')
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size='1GB')
        tokenizer.save_pretrained(temporary)
        info.update(hf_architecture=model.config.model_type, parameters=sum(p.numel() for p in model.parameters()),
                    embedding_parameters=sum(p.numel() for module in (model.get_input_embeddings(), model.get_output_embeddings())
                                             if module is not None for p in module.parameters()),
                    hidden_size=getattr(model.config, 'hidden_size', 0), layers=getattr(model.config, 'num_hidden_layers', 0))
        del model
        gc.collect()
        files = {str(file.relative_to(temporary)): sha256(file, cancel) for file in sorted(temporary.rglob('*')) if file.is_file()}
        manifest = dict(format=IMPORT_FORMAT, source=source, model=info, precision=precision,
                        files=files, artifact_id=identity(files), imported_at=time.time(),
                        transformers_version=importlib.metadata.version('transformers'))
        atomic_json(temporary/'import.json', manifest)
        check_stop(cancel)
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def import_model(request, cache, progress=None, cancel=None):
    require_dependencies()
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from tqdm.auto import tqdm
    progress = progress or CrawlProgress(None, 0)
    output, cache = Path(request.output), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    record_path = Path(str(output) + '.import-request.json')
    wanted = request.model_dump(exclude={'output'})
    with corpus_lock(output):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Choose a fresh imported-model directory')
        if record_path.exists():
            pinned = json.loads(record_path.read_text())
            if pinned['request'] != wanted:
                raise ValueError('Interrupted import belongs to different settings; reuse them or choose a fresh output')
        else:
            progress.update(phase='resolving', detail='Resolving a pinned Hugging Face revision and GGUF file set')
            catalog = repository_files(request)
            selected = next((row for row in catalog['files'] if request.filename in [part['filename'] for part in row['parts']]), None)
            if selected is None:
                raise ValueError('GGUF file or complete shard set was not found in this repository revision')
            pinned = dict(request=wanted, repo=catalog['repo'], revision=catalog['revision'], file=selected, license=catalog['license'])
            atomic_json(record_path, pinned)
        paths, sources = [], []
        for part in pinned['file']['parts']:
            check_stop(cancel)
            existing = try_to_load_from_cache(request.repo, part['filename'], cache_dir=cache, revision=pinned['revision'])
            if not isinstance(existing, str):
                incomplete = cache / ('models--' + request.repo.replace('/', '--')) / 'blobs' / (str(part['sha256']) + '.incomplete')
                remaining = max(0, part['bytes'] - (incomplete.stat().st_size if incomplete.is_file() else 0))
                check_resources(disk_bytes=remaining, directory=cache)
            progress.update(phase='downloading', detail='Downloading / resuming ' + part['filename'],
                            filename=part['filename'], completed=0, budget=part['bytes'], unit='bytes')
            class DownloadProgress(tqdm):
                def update(self, amount=1):
                    check_stop(cancel)
                    result = super().update(amount)
                    now = time.monotonic()
                    if now - getattr(self, '_reported', 0) > 0.5:
                        progress.update(completed=self.n, budget=self.total or part['bytes'])
                        self._reported = now
                    return result
            path = Path(hf_hub_download(request.repo, part['filename'], revision=pinned['revision'],
                                       cache_dir=cache, tqdm_class=DownloadProgress))
            progress.update(phase='verifying', detail='Verifying ' + part['filename'])
            digest = sha256(path, cancel)
            if path.stat().st_size != part['bytes'] or (part['sha256'] and digest != part['sha256']):
                raise ValueError('Downloaded GGUF checksum/size differs from the pinned Hub file: ' + str(path))
            paths.append(path)
            sources.append(dict(part, sha256=digest))
        merged = None
        try:
            path = paths[0]
            if len(paths) > 1:
                check_resources(disk_bytes=sum(p.stat().st_size for p in paths), directory=output.parent)
                handle, name = tempfile.mkstemp(prefix='.hf-merged-', suffix='.gguf', dir=output.parent)
                os.close(handle)
                merged = Path(name)
                progress.update(phase='joining', detail='Joining GGUF shards without changing their packed weights', unit='tensors')
                merge_shards(paths, merged, progress, cancel)
                path = merged
            source = dict(repo=request.repo, revision=pinned['revision'], files=sources, license=pinned['license'])
            manifest = convert_gguf(path, output, source, request.precision, progress, cancel)
        finally:
            if merged:
                merged.unlink(missing_ok=True)
        progress.update(phase='completed', detail=f'IMPORTED MODEL READY {output}', output=str(output),
                        model=manifest['model'], artifact_id=manifest['artifact_id'], completed=1, budget=1, unit='models')
        return manifest


def load_import(directory, verify=False):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'import.json').read_text())
    if manifest.get('format') != IMPORT_FORMAT or identity(manifest['files']) != manifest['artifact_id']:
        raise ValueError('Select a completed, valid Hugging Face GGUF import')
    for name, digest in manifest['files'].items():
        file = (directory/name).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or (verify and sha256(file) != digest):
            raise ValueError('Imported model file is missing or changed: ' + name)
    return manifest


def prepare_corpus(request, progress=None, cancel=None):
    require_dependencies()
    from transformers import AutoTokenizer
    model, source, output = map(Path, (request.model, request.input, request.output))
    progress = progress or CrawlProgress(None, 0)
    imported = load_import(model, verify=True)
    if request.context > imported['model']['context']:
        raise ValueError('Dataset context exceeds the imported model context')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose a fresh Hugging Face dataset directory')
    if any(output.resolve().is_relative_to(path.resolve()) or path.resolve().is_relative_to(output.resolve()) for path in (model, source)):
        raise ValueError('Keep imported model, corpus and dataset paths separate')
    with corpus_lock(source), corpus_lock(output):
        progress.update(phase='splitting', detail='Inspecting scraper text, removing URLs/duplicates, splitting whole source groups')
        validate_corpus(source)
        source_hash = sha256(source, cancel)
        validation, report = corpus_plan(source, request.validation_fraction, request.min_chars, request.min_post_chars)
        if not report['ready']:
            raise ValueError('Collect at least two usable source groups before preparing fine-tuning data')
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        if len(tokenizer) != imported['model']['vocabulary']:
            raise ValueError('Imported tokenizer vocabulary changed')
        bos = imported['model']['bos_id'] if imported['model']['add_bos'] else None
        eos = imported['model']['eos_id']
        if eos is None:
            raise ValueError('Imported tokenizer needs an EOS token')
        temporary = Path(tempfile.mkdtemp(prefix='.prepare-hf-', dir=output.parent))
        try:
            counts, sizes = ({split: 0 for split in ('train', 'val')} for _ in range(2))
            for row in report['sources'].values():
                row['tokens'] = dict(train=0, val=0)
            progress.update(phase='tokenizing', detail='Encoding scraper text with the imported model tokenizer',
                            completed=0, budget=report['accepted_records'], unit='documents')
            with contextlib.ExitStack() as stack:
                binaries = {split: stack.enter_context((temporary/f'{split}.bin').open('wb')) for split in counts}
                texts = {split: stack.enter_context((temporary/f'{split}.jsonl').open('w', encoding='utf-8')) for split in counts}
                for index, row in enumerate(records(source, request.min_chars, request.min_post_chars), 1):
                    check_stop(cancel)
                    split = 'val' if row['group'] in validation else 'train'
                    ids = ([] if bos is None else [bos]) + tokenizer.encode(row['text'], add_special_tokens=False) + [eos]
                    if len(ids) < 2 or min(ids) < 0 or max(ids) >= len(tokenizer):
                        raise ValueError('Tokenizer produced an empty or out-of-vocabulary document')
                    np.asarray(ids, dtype='<u4').tofile(binaries[split])
                    texts[split].write(json.dumps(row, ensure_ascii=False) + '\n')
                    counts[split] += 1
                    sizes[split] += len(ids)
                    report['sources'][row['host']]['tokens'][split] += len(ids)
                    if index % 100 == 0:
                        progress.update(completed=index)
            if sha256(source, cancel) != source_hash:
                raise ValueError('Scraper corpus changed during preparation; retry from a completed collection')
            if counts != report['documents'] or min(sizes.values()) < 2:
                raise ValueError('Prepared corpus no longer matches its grouped split')
            meta = dict(format=DATA_FORMAT, text_policy=TEXT_POLICY_VERSION, model=str(model.resolve()),
                        artifact_id=imported['artifact_id'], context=request.context, vocabulary=len(tokenizer),
                        documents=counts, tokens=sizes, source=str(source.resolve()), source_sha256=source_hash,
                        corpus_report=report, dtype='<u4', loss='all-scraped-text-next-token',
                        hashes={split: sha256(temporary/f'{split}.bin', cancel) for split in counts})
            atomic_json(temporary/'hf-meta.json', meta)
            atomic_json(temporary/'report.json', report)
            check_stop(cancel)
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    progress.update(phase='completed', detail=f'HF DATASET READY {output}', output=str(output),
                    completed=report['accepted_records'], documents=counts, tokens=sizes)
    return meta


def assets(root):
    root = Path(root).resolve()
    result = dict(models=[], datasets=[], runs=[], corpora=[])
    for folder, pattern, key in [('models', 'import.json', 'models'), ('data', 'hf-meta.json', 'datasets'), ('runs', 'hf-run.json', 'runs')]:
        for path in (root/folder).rglob(pattern):
            if any(part.startswith(('.hf-', '.prepare-')) for part in path.parts):
                continue
            try:
                directory = path.parent.resolve()
                if not directory.is_relative_to(root):
                    continue
                info = json.loads(path.read_text())
                if key == 'models':
                    info = load_import(directory)
                result[key].append(dict(path=str(directory.relative_to(root)), modified_at=path.stat().st_mtime, info=info))
            except (OSError, ValueError, KeyError):
                continue
        result[key].sort(key=lambda row: row['modified_at'], reverse=True)
    for path in (root/'data').rglob('*.jsonl'):
        if path.name == 'metrics.jsonl' or any(part.startswith('.') for part in path.relative_to(root).parts):
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            with path.open(encoding='utf-8') as stream:
                first = json.loads(stream.readline(4 * 1024 * 1024))
            if isinstance(first.get('text'), str):
                prepared = prepared_source_report(resolved) is not None
                validate_extraction(first, prepared)
                result['corpora'].append(dict(path=str(resolved.relative_to(root)), bytes=path.stat().st_size,
                                              modified_at=path.stat().st_mtime,
                                              kind='prepared_split' if path.name in {'train.jsonl', 'val.jsonl'} else 'corpus',
                                              split=path.stem if path.name in {'train.jsonl', 'val.jsonl'} else None))
        except (OSError, ValueError, AttributeError):
            continue
    result['corpora'].sort(key=lambda row: row['modified_at'], reverse=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['import', 'prepare', 'merge'], required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--cache', default='data/huggingface-cache')
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    def stopped(*_):
        raise Stopped()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stopped)
    progress = CrawlProgress(args.progress_file, 0)
    try:
        data = json.loads(Path(args.request).read_text())
        if args.stage == 'import':
            import_model(HFImportRequest.model_validate(data), args.cache, progress)
        elif args.stage == 'prepare':
            prepare_corpus(HFPrepareRequest.model_validate(data), progress)
        else:
            from .hf_train import merge_model
            merge_model(HFMergeRequest.model_validate(data), progress)
    except Stopped:
        progress.update(phase='stopped', detail='Stopped; restart this operation to continue (Hub downloads are retained)')
        raise SystemExit(130)
    except Exception as exc:
        progress.update(phase='failed', detail=str(exc))
        raise


if __name__ == '__main__':
    main()
