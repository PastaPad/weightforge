"""Human-written dialogue exports, frozen-tokenizer preparation, and shared chat format."""
import argparse
import contextlib
import gzip
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch
from tokenizers import Tokenizer

from .corpus import clean
from .links import TEXT_POLICY_VERSION
from .model import ModelConfig

FORMAT = 'dialogue-sft-v1'
TEXT_FORMAT = 'text-finetune-v1'
CHAT_FORMAT = 'user-assistant-v1'
PREFIX = {'user': '\nUser:\n', 'assistant': '\nAssistant:\n'}
SOURCES = {
    'oasst1': dict(repo='OpenAssistant/oasst1', revision='fdf72ae0827c1cda404aff25b6603abec9e3399b',
                   file='2023-04-12_oasst_ready.trees.jsonl.gz', size=34145252,
                   sha256='2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7',
                   license='Apache-2.0', attribution='OpenAssistant contributors / LAION'),
    'dolly15k': dict(repo='databricks/databricks-dolly-15k', revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
                     file='databricks-dolly-15k.jsonl', size=13085339,
                     sha256='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec',
                     license='CC-BY-SA-3.0', attribution='Copyright 2023 Databricks, Inc.; Wikipedia editors and contributors'),
}


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(name, cache):
    source = SOURCES[name]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / source['file']
    if path.exists():
        if path.stat().st_size != source['size'] or fingerprint(path) != source['sha256']:
            raise ValueError(f'Cached {name} differs from the pinned export: {path}')
        print(f'CACHE {name}: verified {path}', flush=True)
        return path
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
    print(f'DOWNLOAD {name}: {url}', flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=name + '-', suffix='.part', delete=False) as dest:
            temporary = Path(dest.name)
            with urlopen(Request(url, headers={'User-Agent': 'ScratchLLM/1.0'}), timeout=30) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > source['size']:
                        raise ValueError(f'{name} exceeds the pinned export size')
                    dest.write(block)
        if temporary.stat().st_size != source['size'] or fingerprint(temporary) != source['sha256']:
            raise ValueError(f'{name} download checksum/size mismatch')
        temporary.replace(path)
        return path
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def message_text(text):
    text = clean(text)
    for token in ('pad', 'bos', 'eos'):
        text = text.replace(f'<|{token}|>', f'[{token}]')
    return text


def prompt_tokens(tokenizer, messages):
    """Same segment boundaries for training and serving; no added vocabulary entries."""
    if not messages or len(messages) % 2 != 1:
        raise ValueError('Conversation must end with a user message')
    ids = [tokenizer.token_to_id('<|bos|>')]
    for index, message in enumerate(messages):
        role = 'user' if index % 2 == 0 else 'assistant'
        if message['role'] != role or not message_text(message['content']):
            raise ValueError('Conversation must alternate nonempty user and assistant messages')
        ids.extend(tokenizer.encode(PREFIX[role]).ids)
        ids.extend(tokenizer.encode(message_text(message['content'])).ids)
        if role == 'assistant':
            ids.append(tokenizer.token_to_id('<|eos|>'))
    ids.extend(tokenizer.encode(PREFIX['assistant']).ids)
    return ids


def encode_example(tokenizer, messages, context):
    if len(messages) < 2 or len(messages) % 2 or messages[-1]['role'] != 'assistant':
        raise ValueError('Training example must end with an assistant response')
    history = messages[:-1]
    answer = tokenizer.encode(message_text(messages[-1]['content'])).ids + [tokenizer.token_to_id('<|eos|>')]
    while True:
        prompt = prompt_tokens(tokenizer, history)
        if len(prompt) + len(answer) <= context + 1:
            return prompt + answer, [0] * len(prompt) + [1] * len(answer)
        if len(history) <= 1:
            return None  # Never label a cut-off answer as a complete response.
        history = history[2:]  # Drop complete oldest turns, keeping the current question.


def examples(name, path):
    """Only source-authored text; identifiers, scores, and labels are provenance."""
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if name == 'dolly15k':
                question = message_text(row['instruction'])
                context = message_text(row.get('context', ''))
                answer = message_text(row['response'])
                if not question or not answer:
                    continue
                group = hashlib.sha256((context or question).encode()).hexdigest()
                if context:
                    question += '\n\nContext:\n' + context
                yield dict(source=name, id=str(number), group=group, messages=[
                    dict(role='user', content=question), dict(role='assistant', content=answer)])
            elif name == 'oasst1':
                if row.get('tree_state') != 'ready_for_export':
                    continue
                def eligible(node):
                    return (node.get('lang') == 'en' and node.get('synthetic') is False
                            and not node.get('deleted') and node.get('review_result') is True
                            and bool(message_text(node.get('text', ''))))
                root = row['prompt']
                group = hashlib.sha256(message_text(root['text']).encode()).hexdigest()
                def visit(node, history):
                    role = 'prompter' if len(history) % 2 == 0 else 'assistant'
                    if node.get('role') != role or not eligible(node):
                        return
                    messages = [*history, dict(role='user' if role == 'prompter' else 'assistant',
                                              content=message_text(node['text']))]
                    if role == 'assistant':
                        yield dict(source=name, id=node['message_id'], tree=row['message_tree_id'],
                                   group=group, messages=messages)
                    children = [child for child in node.get('replies', []) if eligible(child)]
                    if role == 'prompter' and children:
                        # Retain the highest-ranked eligible human answer for this prompt.
                        children = [min(children, key=lambda child: (child.get('rank') if child.get('rank') is not None else 10**9,
                                                                    child['message_id']))]
                    for child in children:
                        yield from visit(child, messages)
                yield from visit(root, [])
            else:
                raise ValueError('Unknown dialogue dataset: ' + name)


def load_base(path):
    """Local inference checkpoint only; weights-only loading never imports remote code."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ModelConfig(**state['model_config'])
    tokenizer = Tokenizer.from_str(state['tokenizer_json'])
    if tokenizer.get_vocab_size() != cfg.vocab_size or any(tokenizer.token_to_id(f'<|{token}|>') is None for token in ('pad', 'bos', 'eos')):
        raise ValueError('Base checkpoint tokenizer/model vocabulary mismatch')
    return state, tokenizer


def validate_base(state, meta, checkpoint=None):
    if meta.get('format') not in {FORMAT, TEXT_FORMAT}:
        raise ValueError('Prepare a dialogue or custom text dataset with the frozen base tokenizer first')
    if meta['format'] == FORMAT and meta.get('chat_format') != CHAT_FORMAT:
        raise ValueError('Prepare a dialogue dataset with the supported chat format')
    if meta['format'] == TEXT_FORMAT and meta.get('chat_format') != state.get('chat_format'):
        raise ValueError('Custom text fine-tuning must preserve the checkpoint chat format')
    if state['model_config'] != meta['base_model_config']:
        raise ValueError('Fine-tuning dataset and base checkpoint architectures differ')
    if hashlib.sha256(state['tokenizer_json'].encode()).hexdigest() != meta['tokenizer_sha256']:
        raise ValueError('Fine-tuning tokenizer differs from the base checkpoint; token IDs must stay identical')
    if checkpoint is not None and fingerprint(checkpoint) != meta['base_checkpoint_sha256']:
        raise ValueError('Base checkpoint changed since preparation; choose its original checkpoint or prepare again')


def local_examples(path):
    """Expand complete custom conversations into assistant-answer examples."""
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                messages = row['messages']
                if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
                    raise ValueError('messages must contain complete user/assistant pairs')
                cleaned = []
                for index, message in enumerate(messages):
                    role = 'user' if index % 2 == 0 else 'assistant'
                    if message['role'] != role or not isinstance(message['content'], str):
                        raise ValueError('messages must alternate user and assistant text')
                    content = message_text(message['content'])
                    if not content:
                        raise ValueError('message text is empty after cleaning')
                    cleaned.append(dict(role=role, content=content))
                identity = hashlib.sha256(json.dumps(cleaned, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                group = row.get('group') or identity
                if not isinstance(group, str):
                    raise ValueError('group must be a string')
                for end in range(2, len(cleaned) + 1, 2):
                    yield dict(source='custom', id=f'{number}:{end//2}', group='custom:' + group,
                               messages=cleaned[:end], provenance={key: row[key] for key in
                               ('id', 'synthetic', 'language', 'reference_revision', 'word_order', 'source', 'split') if key in row})
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f'Conversation line {number}: {exc}') from exc


def prepare(checkpoint, output, sources=('oasst1', 'dolly15k'), cache='data/dialogue-downloads', validation_fraction=0.1, input=None):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Dialogue output must be empty; choose a fresh dataset directory')
    state, tokenizer = load_base(checkpoint)
    base_hash, model_config = fingerprint(checkpoint), state['model_config']
    tokenizer_json = state['tokenizer_json']
    del state
    selected = list(dict.fromkeys(sources))
    if (not selected and not input) or any(name not in SOURCES for name in selected):
        raise ValueError('Select at least one configured dialogue source')
    paths = {name: download(name, cache) for name in selected}
    descriptions = {name: dict(SOURCES[name]) for name in selected}
    if input:
        paths['custom'] = Path(input)
        descriptions['custom'] = dict(path=str(Path(input).resolve()), sha256=fingerprint(input),
                                      attribution='User-supplied conversations; per-row provenance retained', license='User supplied')
        selected.append('custom')
    report = dict(format=FORMAT, chat_format=CHAT_FORMAT, text_policy=TEXT_POLICY_VERSION,
                  loss='assistant-response-only', frozen_tokenizer=True, duplicates_removed=0,
                   over_context_removed=0, sources={name: dict(descriptions[name], examples=0,
                    documents=dict(train=0, val=0), supervised_tokens=dict(train=0, val=0)) for name in selected})
    rows, seen, encoded_seen = [], set(), set()
    for name in selected:
        for row in (local_examples(paths[name]) if name == 'custom' else examples(name, paths[name])):
            digest = hashlib.sha256(json.dumps(row['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest in seen:
                report['duplicates_removed'] += 1
                continue
            seen.add(digest)
            encoded = encode_example(tokenizer, row['messages'], model_config['context'])
            if encoded is None:
                report['over_context_removed'] += 1
                continue
            encoded_hash = hashlib.sha256(np.asarray(encoded[0], dtype='<u4').tobytes() + bytes(encoded[1])).hexdigest()
            if encoded_hash in encoded_seen:
                report['duplicates_removed'] += 1
                continue
            encoded_seen.add(encoded_hash)
            rows.append((row, *encoded))
            report['sources'][name]['examples'] += 1
        print(f"DIALOGUE {name}: {report['sources'][name]['examples']:,} fitting response examples", flush=True)
    groups = sorted({row['group'] for row, _, _ in rows})
    if len(groups) < 2:
        raise ValueError('Need at least two usable conversation/context groups at the base context length')
    # Hash ordering splits whole trees; Dolly records sharing a reference passage stay together.
    validation = set(groups[:max(1, min(len(groups) - 1, round(len(groups) * validation_fraction)))])
    report['groups'] = len(groups)
    if input and fingerprint(input) != descriptions['custom']['sha256']:
        raise ValueError('Custom conversation file changed during preparation')
    counts, sizes, targets = ({key: 0 for key in ('train', 'val')} for _ in range(3))
    output.mkdir(parents=True, exist_ok=True)
    (output/'tokenizer.json').write_text(tokenizer_json, encoding='utf-8')
    with contextlib.ExitStack() as stack:
        binaries = {split: stack.enter_context((output/f'{split}.bin').open('wb')) for split in counts}
        masks = {split: stack.enter_context((output/f'{split}.mask.bin').open('wb')) for split in counts}
        texts = {split: stack.enter_context((output/f'{split}.jsonl').open('w', encoding='utf-8')) for split in counts}
        windows = {split: [] for split in counts}
        for row, ids, mask in rows:
            split = 'val' if row['group'] in validation else 'train'
            np.asarray(ids, dtype='<u4').tofile(binaries[split])
            np.asarray(mask, dtype='u1').tofile(masks[split])
            windows[split].append([sizes[split], len(ids) - 1])
            texts[split].write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[split] += 1
            sizes[split] += len(ids)
            targets[split] += sum(mask)
            report['sources'][row['source']]['documents'][split] += 1
            report['sources'][row['source']]['supervised_tokens'][split] += sum(mask)
    for split in counts:
        (output/f'{split}.windows.json').write_text(json.dumps(windows[split]))
    meta = dict(format=FORMAT, chat_format=CHAT_FORMAT, text_policy=TEXT_POLICY_VERSION,
                documents=counts, tokens=sizes, supervised_tokens=targets, dtype='<u4', vocab_size=model_config['vocab_size'],
                tokenizer_sha256=hashlib.sha256(tokenizer_json.encode()).hexdigest(),
                base_checkpoint=str(checkpoint), base_checkpoint_sha256=base_hash, base_model_config=model_config,
                dialogue_report=report)
    (output/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output/'SOURCES.md').write_text('# Dialogue dataset attribution\n\n' + '\n\n'.join(
        f"## {name}\n" + json.dumps(descriptions[name], ensure_ascii=False, indent=2) +
        '\n\nModified: source-specific selection, URL removal, response expansion, grouped splitting, tokenization.' for name in selected) + '\n')
    # A dataset becomes discoverable only after every stream is complete.
    (output/'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'DONE frozen vocabulary: {meta["vocab_size"]:,}; examples {counts}; assistant target tokens {targets}', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Your trained base best.pt (supplies unchanged weights/vocabulary)')
    parser.add_argument('--output', default='data/prepared-dialogue')
    parser.add_argument('--sources', choices=list(SOURCES), nargs='*', default=None)
    parser.add_argument('--input', help='Custom UTF-8 conversation JSONL with messages and optional group/provenance')
    parser.add_argument('--cache', default='data/dialogue-downloads')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    args = parser.parse_args()
    sources = args.sources if args.sources is not None else ([] if args.input else list(SOURCES))
    prepare(args.checkpoint, args.output, sources, args.cache, args.validation_fraction, args.input)


if __name__ == '__main__':
    main()
