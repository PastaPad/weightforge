"""Upload UTF-8 text and prepare a fine-tuning corpus with a frozen checkpoint tokenizer."""
import argparse
import codecs
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import tempfile
import time
import uuid
from pathlib import Path
from threading import Event

import numpy as np
import torch

from .corpus import content_hash
from .dialogue import CHAT_FORMAT, TEXT_FORMAT, fingerprint, load_base, message_text
from .frontier import corpus_lock
from .links import TEXT_POLICY_VERSION
from .progress import CrawlProgress

MAX_BYTES = 64 * 1024 * 1024
UPLOAD_DIRECTORY = 'data/custom-uploads'


class UploadTooLarge(ValueError):
    pass


class PreparationStopped(Exception):
    pass


def check_stop(cancel):
    if cancel is not None and cancel.is_set():
        raise PreparationStopped('Custom corpus preparation stopped')


def check_text(text):
    if any(ord(char) < 32 and char not in '\n\r\t\f' for char in text):
        raise ValueError('Upload plain UTF-8 text; binary/control bytes are not supported')


async def save_upload(root, filename, chunks):
    """Stream a bounded raw upload; filenames are metadata, never destination paths."""
    if (not filename or len(filename) > 200 or Path(filename).name != filename or '\\' in filename
            or Path(filename).suffix.lower() != '.txt'):
        raise ValueError('Choose a .txt file with a simple filename')
    check_text(filename)
    root = Path(root).resolve()
    directory = (root / UPLOAD_DIRECTORY).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Upload directory must stay inside the workspace')
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (uuid.uuid4().hex + '.txt')
    temporary = output.with_suffix('.part')
    record = output.with_suffix('.json')
    digest, size, nonempty = hashlib.sha256(), 0, False
    decoder = codecs.getincrementaldecoder('utf-8-sig')()
    published = False
    try:
        with temporary.open('xb') as dest:
            async for block in chunks:
                size += len(block)
                if size > MAX_BYTES:
                    raise UploadTooLarge('Custom .txt files may be at most 64 MiB')
                text = decoder.decode(block)
                check_text(text)
                nonempty |= bool(text.strip())
                digest.update(block)
                dest.write(block)
            tail = decoder.decode(b'', final=True)
            check_text(tail)
            if not (nonempty or tail.strip()):
                raise ValueError('The uploaded text is empty')
            dest.flush()
            os.fsync(dest.fileno())
        info = dict(path=str(output.relative_to(root)), filename=filename, bytes=size,
                    sha256=digest.hexdigest(), uploaded_at=time.time())
        record.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.link(temporary, output)
        published = True
        return info
    except UnicodeDecodeError as exc:
        raise ValueError('Save the corpus as UTF-8 .txt before uploading') from exc
    finally:
        temporary.unlink(missing_ok=True)
        if not published:
            record.unlink(missing_ok=True)


def uploaded_files(root):
    root = Path(root).resolve()
    directory = (root / UPLOAD_DIRECTORY).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Upload directory must stay inside the workspace')
    rows = []
    for record in directory.glob('*.json'):
        try:
            path = record.with_suffix('.txt')
            if record.is_symlink() or path.is_symlink() or not path.is_file():
                continue
            info = json.loads(record.read_text())
            if path.stat().st_size == info['bytes']:
                rows.append(dict(info, path=str(path.relative_to(root))))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row['uploaded_at'], reverse=True)


def validate_paths(checkpoint, source, output):
    checkpoint, source, output = (Path(path).resolve() for path in (checkpoint, source, output))
    if not checkpoint.is_file() or checkpoint.name == 'last.pt':
        raise ValueError('Select a saved inference checkpoint such as best.pt')
    if not source.is_file() or source.suffix.lower() != '.txt':
        raise ValueError('Upload or choose an existing .txt corpus inside the workspace')
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (checkpoint, source)):
        raise ValueError('Keep the corpus, checkpoint and prepared output paths separate')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Prepared output must be empty; choose a fresh dataset directory')


def prepare(checkpoint, source, output, validation_fraction=0.1, progress=None, cancel=None):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    checkpoint, source, output = map(Path, (checkpoint, source, output))
    validate_paths(checkpoint, source, output)
    progress = progress or CrawlProgress(None, 0)
    progress.update(phase='loading', detail='Reading the selected checkpoint and UTF-8 corpus on CPU')
    base_hash = fingerprint(checkpoint)
    state, tokenizer = load_base(checkpoint)
    if fingerprint(checkpoint) != base_hash:
        raise ValueError('Checkpoint changed while reading it; retry with a complete saved checkpoint')
    if 'optimizer' in state or state.get('chat_format') not in {None, CHAT_FORMAT}:
        raise ValueError('Select a supported inference checkpoint, not an optimizer/resume file')
    model_config, tokenizer_json, chat_format = state['model_config'], state['tokenizer_json'], state.get('chat_format')
    del state
    check_stop(cancel)
    with source.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise UploadTooLarge('Custom .txt files may be at most 64 MiB')
    source_hash, source_size = hashlib.sha256(raw).hexdigest(), len(raw)
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Save the corpus as UTF-8 .txt') from exc
    del raw
    check_text(text)
    filename = source.name
    try:
        upload = json.loads(source.with_suffix('.json').read_text())
        if upload.get('sha256') == source_hash:
            filename = upload['filename']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    paragraphs = re.split(r'\n[ \t]*\n+', text.replace('\r\n', '\n').replace('\r', '\n'))
    del text
    report = dict(format=TEXT_FORMAT, text_policy=TEXT_POLICY_VERSION, frozen_tokenizer=True,
                  loss='all-text-next-token', split_unit='paragraphs', input_records=len(paragraphs),
                  duplicates_removed=0, empty_removed=0, validation_fraction=validation_fraction,
                  source=dict(path=str(source), filename=filename, sha256=source_hash, bytes=source_size))
    rows, seen, encoded_seen = [], set(), set()
    progress.update(phase='tokenizing', detail='Removing URLs and duplicates; encoding with the existing tokenizer',
                    budget=len(paragraphs), completed=0)
    for index, text in enumerate(paragraphs):
        check_stop(cancel)
        text = message_text(text)
        group = content_hash(text)
        if not text:
            report['empty_removed'] += 1
        elif group in seen:
            report['duplicates_removed'] += 1
        else:
            seen.add(group)
            ids = np.asarray(tokenizer.encode(text).ids, dtype='<u4')
            encoded_hash = hashlib.sha256(ids.tobytes()).hexdigest()
            if not len(ids):
                report['empty_removed'] += 1
            elif encoded_hash in encoded_seen:
                report['duplicates_removed'] += 1
            else:
                encoded_seen.add(encoded_hash)
                rows.append((dict(id=index + 1, group=group, text=text), ids))
        if (index + 1) % 100 == 0 or index + 1 == len(paragraphs):
            progress.update(completed=index + 1)
    del paragraphs
    if len(rows) == 1:
        # A single block still has a held-out split, with no overlapping token windows.
        _, ids = rows[0]
        rows, seen = [], set()
        offset = 0
        report['split_unit'] = 'nonoverlapping token chunks'
        for index, chunk in enumerate(np.array_split(ids, max(2, math.ceil(len(ids) / model_config['context'])))):
            check_stop(cancel)
            group = hashlib.sha256(chunk.tobytes()).hexdigest()
            if len(chunk) and group not in seen:
                seen.add(group)
                rows.append((dict(id=index + 1, group=group, token_range=[offset, offset + len(chunk)]), chunk))
            elif len(chunk):
                report['duplicates_removed'] += 1
            offset += len(chunk)
    if len(rows) < 2:
        raise ValueError('Need at least two distinct text passages/chunks after URL removal and deduplication; add more text')
    groups = sorted(row['group'] for row, _ in rows)
    validation = set(groups[:max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))])
    counts, sizes = ({split: 0 for split in ('train', 'val')} for _ in range(2))
    report['groups'] = len(groups)
    bos, eos = (tokenizer.token_to_id(f'<|{token}|>') for token in ('bos', 'eos'))
    check_stop(cancel)
    with corpus_lock(output):
        validate_paths(checkpoint, source, output)
        temporary = Path(tempfile.mkdtemp(prefix=f'.prepare-{output.name}-', dir=output.parent))
        try:
            (temporary/'tokenizer.json').write_text(tokenizer_json, encoding='utf-8')
            progress.update(phase='writing', detail='Writing frozen-tokenizer training and validation streams',
                            budget=len(rows), completed=0)
            with contextlib.ExitStack() as stack:
                binaries = {split: stack.enter_context((temporary/f'{split}.bin').open('wb')) for split in counts}
                texts = {split: stack.enter_context((temporary/f'{split}.jsonl').open('w', encoding='utf-8')) for split in counts}
                for index, (row, ids) in enumerate(rows, 1):
                    check_stop(cancel)
                    split = 'val' if row['group'] in validation else 'train'
                    np.asarray([bos], dtype='<u4').tofile(binaries[split])
                    ids.tofile(binaries[split])
                    np.asarray([eos], dtype='<u4').tofile(binaries[split])
                    texts[split].write(json.dumps(dict(row, source_sha256=source_hash), ensure_ascii=False) + '\n')
                    sizes[split] += len(ids) + 2
                    counts[split] += 1
                    if index % 100 == 0 or index == len(rows):
                        progress.update(completed=index)
            report.update(documents=counts, tokens=sizes,
                          actual_validation_token_fraction=sizes['val'] / sum(sizes.values()))
            meta = dict(format=TEXT_FORMAT, chat_format=chat_format, text_policy=TEXT_POLICY_VERSION,
                        documents=counts, tokens=sizes, dtype='<u4', vocab_size=model_config['vocab_size'],
                        tokenizer_sha256=hashlib.sha256(tokenizer_json.encode()).hexdigest(),
                        base_checkpoint=str(checkpoint), base_checkpoint_sha256=base_hash,
                        base_model_config=model_config, custom_report=report)
            (temporary/'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            (temporary/'meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
            check_stop(cancel)
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    progress.update(phase='completed', detail=f'DATASET READY {output}; start fine-tuning when ready',
                    documents=counts, tokens=sizes)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='UTF-8 .txt file')
    parser.add_argument('--output', required=True, help='Fresh prepared dataset directory')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cancel = Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    progress = CrawlProgress(args.progress_file, 0)
    try:
        prepare(args.checkpoint, args.input, args.output, args.validation_fraction, progress, cancel)
    except PreparationStopped:
        progress.update(phase='stopped', detail='Preparation stopped; run preparation again to finish the dataset')
        raise SystemExit(130)
    except Exception as exc:
        progress.update(phase='failed', detail=str(exc))
        raise


if __name__ == '__main__':
    main()
