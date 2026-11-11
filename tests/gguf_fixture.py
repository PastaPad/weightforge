"""Tiny random checkpoints and a CPU llama.cpp harness; never optimize model weights."""
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from scratchllm.dialogue import CHAT_FORMAT
from scratchllm.model import GPT, ModelConfig


def checkpoint(root, architecture='modern', dialogue=True):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([
        'The sky is blue. A blue planet has water and clouds.',
        "Hello! I'm reading about the planet. Can you explain its rings?",
        'The rings contain ice and dust. People study these planets with telescopes.',
        'Numbers: 12345, 67890. Tabs\tand\nnew lines. Swedish: Räksmörgås. 日本語. שלום.',
        '\nUser:\nHello\nAssistant:\nHello there.',
    ], trainer=trainers.BpeTrainer(vocab_size=384, min_frequency=1,
        special_tokens=['<|pad|>', '<|bos|>', '<|eos|>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size(), width=32, context=256, heads=4, layers=2, dropout=0,
                      architecture=architecture, **(dict(kv_heads=2, ffn_width=64) if architecture == 'modern' else {}))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(812)
        model = GPT(cfg).eval()
    state = dict(model=model.state_dict(), model_config=model.config_dict(), tokenizer_json=tokenizer.to_str(), step=17)
    if dialogue:
        state['chat_format'] = CHAT_FORMAT
    path = Path(root)/f'{architecture}-best.pt'
    torch.save(state, path)
    return path, model, tokenizer


@contextmanager
def llama_server(binary, model, directory, context=256):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    address = f'http://127.0.0.1:{port}'
    log_path = Path(directory)/'llama-server.log'
    with log_path.open('wb') as log:
        process = subprocess.Popen([str(binary), '-m', str(model), '--host', '127.0.0.1', '--port', str(port),
            '-ngl', '0', '-t', '2', '-tb', '2', '-c', str(context), '-np', '1', '--no-warmup',
            '--flash-attn', 'off', '--cache-type-k', 'f32', '--cache-type-v', 'f32'],
            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    def request(route, body=None):
        data = json.dumps(body).encode() if body is not None else None
        try:
            with urlopen(Request(address + route, data=data, headers={'Content-Type': 'application/json'}), timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            raise AssertionError(f'{route}: HTTP {exc.code}: {exc.read().decode()}') from exc
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(log_path.read_text())
            try:
                request('/health')
                break
            except (URLError, AssertionError):
                time.sleep(0.05)
        else:
            raise AssertionError('llama.cpp did not start: ' + log_path.read_text())
        yield request
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == '__main__':
    # Give a real export enough time for the stop/lifecycle regression check.
    import gguf
    from scratchllm.gguf_export import main
    original = gguf.GGUFWriter.write_tensor_data
    def slow(self, *args, **kwargs):
        time.sleep(0.1)
        return original(self, *args, **kwargs)
    gguf.GGUFWriter.write_tensor_data = slow
    main()
