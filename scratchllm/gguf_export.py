"""Export local inference checkpoints to llama.cpp GGUF, entirely on the CPU."""
import argparse
import hashlib
import json
import os
import pickle
import signal
import tempfile
from pathlib import Path
from threading import Event

import gguf
import numpy as np
import torch
from tokenizers import Tokenizer, pre_tokenizers

from .dialogue import CHAT_FORMAT
from .model import GPT, ModelConfig
from .progress import CrawlProgress

EXPORT_VERSION = 'scratchllm-gguf-v1'
CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{%- if message['role'] != ('user' if loop.index0 % 2 == 0 else 'assistant') -%}
{{- raise_exception('Messages must alternate user/assistant, beginning with user') -}}
{%- endif -%}
{{- '\nUser:\n' if message['role'] == 'user' else '\nAssistant:\n' -}}
{{- message['content'] | trim -}}
{%- if message['role'] == 'assistant' -%}{{- eos_token -}}{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '\nAssistant:\n' -}}{%- endif -%}"""
LEGACY_NOTE = 'Original GPT uses exact GELU; llama.cpp GPT-2 uses the tanh GELU approximation.'


class ExportCancelled(Exception):
    pass


def check_stop(cancel):
    if cancel is not None and cancel.is_set():
        raise ExportCancelled('Export stopped')


def digest(stream, cancel=None):
    value = hashlib.sha256()
    while block := stream.read(4 * 1024 * 1024):
        check_stop(cancel)
        value.update(block)
    return value.hexdigest()


def load_snapshot(path, cancel=None):
    """One open descriptor stays on one complete snapshot across atomic training saves."""
    if path.name == 'last.pt':
        raise ValueError('Select an inference checkpoint such as best.pt; last.pt contains training/resume state')
    with path.open('rb') as source:
        before = os.fstat(source.fileno())
        sha256 = digest(source, cancel)
        source.seek(0)
        try:
            state = torch.load(source, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError('Select a weights-only inference checkpoint (best.pt), not a training/resume file') from exc
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint was modified in place during export; retry with a complete saved checkpoint')
    check_stop(cancel)
    if not isinstance(state, dict) or not {'model', 'model_config', 'tokenizer_json'} <= state.keys():
        raise ValueError('Expected a Scratch LLM inference checkpoint with weights, architecture and tokenizer')
    if 'optimizer' in state:
        raise ValueError('Export the inference best.pt rather than an optimizer/resume checkpoint')
    if state.get('chat_format') not in {None, CHAT_FORMAT}:
        raise ValueError('Unsupported checkpoint chat format')
    return state, sha256


def tokenizer_metadata(tokenizer_json, vocab_size):
    data = json.loads(tokenizer_json)
    model = data.get('model') or {}
    pre = data.get('pre_tokenizer') or {}
    if (model.get('type') != 'BPE' or pre.get('type') != 'ByteLevel'
            or pre.get('add_prefix_space') is not False or pre.get('use_regex', True) is not True
            or (data.get('decoder') or {}).get('type') != 'ByteLevel'
            or any(data.get(key) is not None for key in ('normalizer', 'post_processor', 'padding', 'truncation'))
            or any(model.get(key) for key in ('dropout', 'unk_token', 'continuing_subword_prefix',
                                              'end_of_word_suffix', 'byte_fallback', 'ignore_merges'))):
        raise ValueError('GGUF export requires the original ByteLevel BPE tokenizer with GPT-2 splitting and no transforms')
    tokenizer = Tokenizer.from_str(tokenizer_json)
    vocab = model.get('vocab', {})
    if (len(vocab) != vocab_size or tokenizer.get_vocab_size() != vocab_size
            or set(vocab.values()) != set(range(vocab_size))
            or not set(pre_tokenizers.ByteLevel.alphabet()) <= vocab.keys()):
        raise ValueError('Tokenizer IDs must be contiguous, match the model vocabulary, and include all 256 byte symbols')
    specials = {name: tokenizer.token_to_id(f'<|{name}|>') for name in ('pad', 'bos', 'eos')}
    if None in specials.values() or len(set(specials.values())) != 3:
        raise ValueError('Tokenizer must have distinct pad, BOS and EOS tokens')
    expected = set(specials.values())
    added = data.get('added_tokens', [])
    if ({row['id'] for row in added} != expected or len(added) != 3
            or any(not row.get('special') or row.get('content') != tokenizer.id_to_token(row['id'])
                   or any(row.get(key) for key in ('single_word', 'lstrip', 'rstrip', 'normalized')) for row in added)):
        raise ValueError('Only the original pad/BOS/EOS added-token behavior is supported')
    tokens = [tokenizer.id_to_token(index) for index in range(vocab_size)]
    merges = []
    for pair in model.get('merges', []):
        pair = pair.split(' ') if isinstance(pair, str) else pair
        if (len(pair) != 2 or any(piece not in vocab or ' ' in piece for piece in pair)
                or ''.join(pair) not in vocab):
            raise ValueError('Tokenizer contains an unsupported BPE merge')
        merges.append(' '.join(pair))
    if not merges:
        raise ValueError('A fitted BPE tokenizer with at least one merge is required for llama.cpp export')
    return tokens, merges, specials


def tensor_mapping(state, cfg):
    weights = state['model']
    # Meta tensors validate the complete schema without allocating another model's weights.
    with torch.device('meta'):
        expected = GPT(cfg).state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        raise ValueError('Checkpoint tensor names do not match its architecture')
    for name, tensor in weights.items():
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                or tensor.layout != torch.strided or tensor.shape != expected[name].shape):
            raise ValueError('Checkpoint tensor has an invalid shape or dtype: ' + name)
    if not torch.equal(weights['token.weight'], weights['head.weight']):
        raise ValueError('Checkpoint embedding/output weights are not tied as required by this model')
    tensors = [('token_embd.weight', weights['token.weight']), ('output_norm.weight', weights['norm.weight'])]
    modern = cfg.architecture == 'modern'
    if not modern:
        tensors.extend([('position_embd.weight', weights['position.weight']), ('output_norm.bias', weights['norm.bias'])])
    for index in range(cfg.layers):
        source, target = f'blocks.{index}.', f'blk.{index}.'
        mapping = {'ln1.weight': 'attn_norm.weight', 'ln2.weight': 'ffn_norm.weight', 'attn.proj.weight': 'attn_output.weight'}
        if modern:
            mapping.update({'attn.query.weight': 'attn_q.weight', 'mlp.gate.weight': 'ffn_gate.weight',
                            'mlp.up.weight': 'ffn_up.weight', 'mlp.down.weight': 'ffn_down.weight'})
            key, value = weights[source + 'attn.key_value.weight'].chunk(2, dim=0)
            # ScratchLLM and GGUF Llama both rotate adjacent coordinate pairs. No HF/Llama permutation.
            tensors.extend([(target + 'attn_k.weight', key), (target + 'attn_v.weight', value)])
        else:
            mapping.update({'ln1.bias': 'attn_norm.bias', 'ln2.bias': 'ffn_norm.bias', 'attn.qkv.weight': 'attn_qkv.weight',
                            'mlp.0.weight': 'ffn_up.weight', 'mlp.0.bias': 'ffn_up.bias',
                            'mlp.2.weight': 'ffn_down.weight', 'mlp.2.bias': 'ffn_down.bias'})
            tensors.extend([(target + 'attn_qkv.bias', torch.zeros(3 * cfg.width, device='cpu')),
                            (target + 'attn_output.bias', torch.zeros(cfg.width, device='cpu'))])
        tensors.extend((target + destination, weights[source + name]) for name, destination in mapping.items())
    return tensors


def export_gguf(checkpoint, output, precision='f16', progress=None, cancel=None):
    checkpoint, output = Path(checkpoint), Path(output)
    if precision not in {'f16', 'f32'}:
        raise ValueError('GGUF precision must be f16 or f32')
    if output.suffix.lower() != '.gguf':
        raise ValueError('Export output must end in .gguf')
    if output.exists() or output.is_symlink() or checkpoint.resolve() == output.resolve():
        raise ValueError('Choose a fresh GGUF output file; existing files are never overwritten')
    progress = progress or CrawlProgress(None, 0)
    progress.update(phase='loading', detail=f'Reading CPU checkpoint snapshot: {checkpoint}', completed=0)
    state, source_hash = load_snapshot(checkpoint, cancel)
    cfg = ModelConfig(**state['model_config'])
    tokens, merges, specials = tokenizer_metadata(state['tokenizer_json'], cfg.vocab_size)
    tensors = tensor_mapping(state, cfg)
    architecture = 'llama' if cfg.architecture == 'modern' else 'gpt2'
    progress.update(phase='converting', detail=f'Exporting {architecture} / {precision}: {len(tensors)} tensors',
                    budget=len(tensors), architecture=architecture, model_config=cfg.to_dict(), precision=precision,
                    checkpoint_sha256=source_hash, checkpoint_step=state.get('step'), chat_format=state.get('chat_format'),
                    compatibility_note=LEGACY_NOTE if architecture == 'gpt2' else None)
    check_stop(cancel)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.part', dir=output.parent)
    os.close(descriptor)
    temporary, writer = Path(name), None
    try:
        writer = gguf.GGUFWriter(temporary, architecture)
        writer.add_name(checkpoint.parent.name + '/' + checkpoint.stem)
        writer.add_description('Locally trained ScratchLLM checkpoint; ' + EXPORT_VERSION)
        writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16 if precision == 'f16' else gguf.LlamaFileType.ALL_F32))
        writer.add_context_length(cfg.context)
        writer.add_embedding_length(cfg.width)
        writer.add_block_count(cfg.layers)
        writer.add_feed_forward_length(cfg.ffn_width if architecture == 'llama' else 4 * cfg.width)
        writer.add_head_count(cfg.heads)
        writer.add_head_count_kv(cfg.kv_heads if architecture == 'llama' else cfg.heads)
        if architecture == 'llama':
            writer.add_layer_norm_rms_eps(1e-6)
            writer.add_rope_dimension_count(cfg.width // cfg.heads)
            writer.add_rope_freq_base(cfg.rope_theta)
        else:
            writer.add_layer_norm_eps(1e-5)
            writer.add_string('scratchllm.compatibility_note', LEGACY_NOTE)
        writer.add_tokenizer_model('gpt2')
        writer.add_tokenizer_pre('gpt-2')
        writer.add_token_list(tokens)
        writer.add_token_merges(merges)
        writer.add_token_types([gguf.TokenType.CONTROL if index in specials.values() else gguf.TokenType.NORMAL
                                for index in range(cfg.vocab_size)])
        writer.add_bos_token_id(specials['bos'])
        writer.add_eos_token_id(specials['eos'])
        writer.add_pad_token_id(specials['pad'])
        writer.add_add_bos_token(True)
        writer.add_add_eos_token(False)
        writer.add_add_space_prefix(False)
        if state.get('chat_format') == CHAT_FORMAT:
            writer.add_chat_template(CHAT_TEMPLATE)
        writer.add_string('scratchllm.export_version', EXPORT_VERSION)
        writer.add_string('scratchllm.checkpoint_sha256', source_hash)
        writer.add_string('scratchllm.model_config', json.dumps(cfg.to_dict(), sort_keys=True))
        writer.add_string('scratchllm.tokenizer_sha256', hashlib.sha256(state['tokenizer_json'].encode()).hexdigest())
        if isinstance(state.get('step'), int) and state['step'] >= 0:
            writer.add_uint64('scratchllm.checkpoint_step', state['step'])
        def dtype(tensor):
            return np.float16 if precision == 'f16' and tensor.ndim == 2 else np.float32
        for name, tensor in tensors:
            writer.add_tensor_info(name, tensor.shape, np.dtype(dtype(tensor)), tensor.numel() * np.dtype(dtype(tensor)).itemsize)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for number, (name, tensor) in enumerate(tensors, 1):
            check_stop(cancel)
            array = tensor.detach().float().cpu().contiguous().numpy().astype(dtype(tensor), copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f'Tensor {name} is non-finite or overflows {precision}; try f32 for finite large weights')
            writer.write_tensor_data(array)
            progress.update(completed=number, tensor=name)
        writer.flush()
        for stream in writer.fout:
            os.fsync(stream.fileno())
        writer.close()
        progress.update(phase='verifying', detail='Hashing complete GGUF before publication')
        with temporary.open('rb') as stream:
            sha256 = digest(stream, cancel)
        check_stop(cancel)
        # Link is an atomic, no-overwrite publication on the same filesystem.
        os.link(temporary, output)
        progress.update(phase='completed', detail=f'GGUF READY {output}', output=str(output), bytes=output.stat().st_size, sha256=sha256)
        return dict(output=str(output), sha256=sha256, bytes=output.stat().st_size, architecture=architecture,
                    checkpoint_sha256=source_hash, precision=precision, tensors=len(tensors))
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='A local best.pt inference checkpoint')
    parser.add_argument('--output', required=True, help='Fresh .gguf output path')
    parser.add_argument('--precision', choices=['f16', 'f32'], default='f16')
    parser.add_argument('--progress-file')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cancel = Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel.set())
    progress = CrawlProgress(args.progress_file, 0)
    try:
        export_gguf(args.checkpoint, args.output, args.precision, progress, cancel)
    except ExportCancelled:
        progress.update(phase='stopped', detail='Export stopped; rerun to export the checkpoint again')
        raise SystemExit(130)
    except Exception as exc:
        progress.update(phase='failed', detail=str(exc))
        raise


if __name__ == '__main__':
    main()
