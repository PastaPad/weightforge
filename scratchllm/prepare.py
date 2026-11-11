"""Deduplicate documents, split before fitting BPE, and write uint32 token streams."""
import argparse
import hashlib
import json
import shutil
import signal
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .corpus import clean, corpus_plan, prepared_source_report, records, source_records, validate_extraction
from .links import TEXT_POLICY_VERSION
from .frontier import corpus_lock

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def validate_corpus(path):
    """Reject obsolete crawler output before creating files or fitting a tokenizer."""
    prepared = prepared_source_report(path) is not None
    for number, row in source_records(path):
        try:
            validate_extraction(row, prepared)
        except ValueError as exc:
            raise ValueError(f'Corpus line {number}: {exc}') from exc


def prepare(args):
    print('PREPARE scanning and grouping corpus text', flush=True)
    validation, report = corpus_plan(args.input, args.validation_fraction, args.min_chars, args.min_post_chars)
    if not report['ready']:
        raise ValueError(f"Need at least two distinct texts/threads/articles for a grouped train/validation split; got {report['groups']} groups. "
                         f"{report['too_short_removed']} records were below the minimum lengths "
                         f"({args.min_chars} document/article characters, {args.min_post_chars} post characters). "
                         "For short custom text, set the preparation minimum to 1.")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Prepared output must be empty; choose a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    print(f"CORPUS {report['accepted_records']} unique records in {report['groups']} groups; "
          f"{report['duplicates_removed']} duplicates removed; grouped validation {report['validation_character_fraction']:.1%} of characters", flush=True)
    for host, source in sorted(report['sources'].items()):
        print(f"SOURCE {host}: {source['accepted_records']} records, {source['characters']} characters; "
              f"{source['documents']['train']} train / {source['documents']['val']} validation", flush=True)
    # Spool cleaned data rather than retaining the corpus in RAM.
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, (out / "val.jsonl").open("w", encoding="utf-8") as val:
        for row in records(args.input, args.min_chars, args.min_post_chars):
            split = 'val' if row['group'] in validation else 'train'
            (val if split == 'val' else train).write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] += 1
            if sum(counts.values()) % 1000 == 0:
                print(f"PREPARE split {sum(counts.values())}/{report['accepted_records']} documents", flush=True)
    if not all(counts.values()):
        raise SystemExit(f"Both document splits need data; got {counts}. Collect more pages or adjust --validation-fraction.")

    def prepared_rows(split):
        with (out / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
                                  special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    print(f"PREPARE fitting BPE vocabulary on {counts['train']} training documents", flush=True)
    tokenizer.train_from_iterator((row['text'] for row in prepared_rows('train')), trainer=trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    sizes = {}
    for source in report['sources'].values():
        source['tokens'] = dict(train=0, val=0)
    for split in counts:
        total = 0
        print(f'PREPARE encoding {split}: {counts[split]} documents', flush=True)
        with (out / f"{split}.bin").open("wb") as dest:
            for number, row in enumerate(prepared_rows(split), 1):
                ids = [tokenizer.token_to_id(SPECIALS[1]), *tokenizer.encode(row['text']).ids, tokenizer.token_to_id(SPECIALS[2])]
                np.asarray(ids, dtype="<u4").tofile(dest)
                total += len(ids)
                report['sources'][row['host']]['tokens'][split] += len(ids)
                if number % 1000 == 0:
                    print(f'PREPARE encoded {split}: {number}/{counts[split]} documents', flush=True)
        sizes[split] = total
    meta = dict(documents=counts, tokens=sizes, vocab_size=tokenizer.get_vocab_size(),
                dtype="<u4", tokenizer_sha256=hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest(),
                corpus_report=report, text_policy=TEXT_POLICY_VERSION)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def prepare_atomic(args):
    """Publish only complete datasets; an interrupted preparation can start again safely."""
    output = Path(args.output)
    with corpus_lock(output):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError('Prepared output must be empty; choose a fresh directory')
        temporary = Path(tempfile.mkdtemp(prefix=f'.prepare-{output.name}-', dir=output.parent))
        try:
            prepare(SimpleNamespace(**(vars(args) | {'output': str(temporary)})))
            temporary.rename(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    print(f'DATASET READY {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw-text-v4.jsonl", help='JSONL corpus or UTF-8 .txt (paragraphs or one text per line)')
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--validation-fraction", type=float, default=0.05)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--min-post-chars", type=int, default=40)
    p.add_argument('--atomic-output', action='store_true', help='Publish the dataset only after successful preparation')
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1 or args.vocab_size < 259:
        p.error("validation fraction must be in (0,1), vocab size >= 259")
    previous = {}
    if args.atomic_output:
        def stopped(signum, *_):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stopped)
    try:
        (prepare_atomic if args.atomic_output else prepare)(args)
    except ValueError as exc:
        p.error(str(exc))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
