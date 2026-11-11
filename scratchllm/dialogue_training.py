"""Conversation-bounded, assistant-masked batches and resumable example sampling."""
import json

import numpy as np

from .dialogue import FORMAT, TEXT_FORMAT, CHAT_FORMAT, validate_base, load_base


def initialize_from(model, checkpoint, meta, tokenizer_json):
    state, _ = load_base(checkpoint)
    validate_base(state, meta, checkpoint)
    if model.config_dict() != state['model_config'] or tokenizer_json != state['tokenizer_json']:
        raise ValueError('Fine-tuning must preserve base architecture and exact tokenizer JSON')
    model.load_state_dict(state['model'])


def finetuning_provenance(meta):
    if meta.get('format') not in {FORMAT, TEXT_FORMAT}:
        return {}
    provenance = dict(base_checkpoint=meta['base_checkpoint'], base_checkpoint_sha256=meta['base_checkpoint_sha256'])
    if meta['format'] == FORMAT:
        provenance.update(chat_format=CHAT_FORMAT, dialogue_sources=meta['dialogue_report']['sources'])
    else:
        provenance.update(finetuning_format=TEXT_FORMAT, chat_format=meta.get('chat_format'),
                          custom_corpus=meta['custom_report'])
    return provenance


def load_dialogue(data, meta, streams, context):
    masks, windows = {}, {}
    if meta.get('format') != FORMAT or context != meta['base_model_config']['context']:
        raise ValueError('Dialogue context must match its base model')
    for split, stream in streams.items():
        masks[split] = np.memmap(data/f'{split}.mask.bin', dtype='u1', mode='r')
        windows[split] = [tuple(row) for row in json.loads((data/f'{split}.windows.json').read_text())]
        if len(masks[split]) != len(stream) or np.any(masks[split] > 1):
            raise ValueError('Invalid dialogue loss mask')
        offset = 0
        for start, length in windows[split]:
            if start != offset or not 1 <= length <= context or start + length >= len(stream):
                raise ValueError('Invalid conversation boundaries')
            if masks[split][start] or not masks[split][start + 1:start + length + 1].any():
                raise ValueError('Every dialogue example must have masked prompt and assistant targets')
            offset += length + 1
        if (offset != len(stream) or len(windows[split]) != meta['documents'][split]
                or int(masks[split].sum()) != meta['supervised_tokens'][split]):
            raise ValueError('Dialogue metadata does not match conversation streams')
    return masks, windows


class ShuffledExamples:
    def __init__(self, windows, seed):
        self.windows = windows
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor, self.passes = [], 0, 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.windows)).tolist()
                self.cursor = 0
                self.passes += 1
            end = min(len(self.order), self.cursor + count - len(result))
            result.extend(self.windows[index] for index in self.order[self.cursor:end])
            self.cursor = end
        return result

    def state_dict(self):
        return dict(examples=len(self.windows), order=self.order, cursor=self.cursor, passes=self.passes,
                    rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        if state['examples'] != len(self.windows):
            raise ValueError('Dialogue examples changed')
        self.order, self.cursor, self.passes = state['order'], state['cursor'], state['passes']
        self.rng.bit_generator.state = state['rng']
