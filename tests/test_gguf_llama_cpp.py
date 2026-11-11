"""Optional actual llama.cpp inference comparison; random weights only, no training."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scratchllm.dialogue import prompt_tokens
from scratchllm.gguf_export import export_gguf
from gguf_fixture import checkpoint, llama_server


@unittest.skipUnless(os.environ.get('LLAMA_SERVER'), 'Set LLAMA_SERVER to a locally built llama-server binary')
class LlamaCppExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_tokenizer_chat_and_prediction_probabilities_match_both_architectures_and_precisions(self):
        for architecture in ('modern', 'gpt'):
            for precision in ('f32', 'f16'):
                with self.subTest(architecture=architecture, precision=precision), tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
                    root = Path(directory)
                    path, model, tokenizer = checkpoint(root, architecture)
                    out = root/'model.gguf'
                    export_gguf(path, out, precision)
                    with llama_server(os.environ['LLAMA_SERVER'], out, root) as request:
                        for text in ("Hello! I'm reading about the planet.", '  spaces\t\tline\r\n\n',
                                     'Räksmörgås 日本語 שלום 🐱👨‍👩‍👧‍👦', '1234567890... !!!', '<|bos|>Hello<|eos|>'):
                            tokens = request('/tokenize', dict(content=text, add_special=False, parse_special=True))['tokens']
                            self.assertEqual(tokens, tokenizer.encode(text).ids, text)
                        self.assertEqual(request('/tokenize', dict(content='Hello', add_special=True))['tokens'], [1, *tokenizer.encode('Hello').ids])
                        messages = [dict(role='user', content='Hello!'), dict(role='assistant', content='Hello there.'),
                                    dict(role='user', content='Can you explain the rings?')]
                        formatted = request('/apply-template', dict(messages=messages))['prompt']
                        # llama.cpp removes template BOS when its tokenizer will add it.
                        tokens = request('/tokenize', dict(content=formatted, add_special=True, parse_special=True))['tokens']
                        self.assertEqual(tokens, prompt_tokens(tokenizer, messages))
                        chat = request('/v1/chat/completions', dict(messages=messages, max_tokens=1, temperature=0))
                        self.assertEqual(chat['usage']['prompt_tokens'], len(tokens))
                        for prompt in ('The blue planet', 'Rings contain ice.\nWhy?', 'Räksmörgås 日本語'):
                            ids = [1, *tokenizer.encode(prompt).ids]
                            with torch.inference_mode():
                                reference = model(torch.tensor([ids]))[0][0, -1].log_softmax(-1).numpy()
                            result = request('/completion', dict(prompt=ids, n_predict=1, temperature=0, top_k=0,
                                top_p=1, min_p=0, repeat_penalty=1, n_probs=len(reference), return_tokens=True,
                                cache_prompt=False, post_sampling_probs=False))
                            probabilities = result.get('completion_probabilities', result.get('probs'))[0]
                            values = {entry['id']: entry['logprob'] for entry in probabilities['top_logprobs']}
                            self.assertEqual(len(values), len(reference))
                            actual = np.array([values[index] for index in range(len(reference))])
                            tolerance = 0.002 if precision == 'f16' else 0.0003 if architecture == 'gpt' else 0.00003
                            np.testing.assert_allclose(actual, reference, rtol=0, atol=tolerance)
                            self.assertEqual(probabilities['id'], int(reference.argmax()))


if __name__ == '__main__':
    unittest.main()
