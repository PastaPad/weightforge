"""Deterministic sampling math and URL-free output, with fixed logits; no fitting."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scratchllm.generate import Generator
from scratchllm.jobs import TrainRequest
from scratchllm.model import GPT
from scratchllm.serve import CompletionRequest


class FixedModel:
    generate = GPT.generate
    training = False
    cfg = SimpleNamespace(context=8)

    def __init__(self, logits):
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def __call__(self, ids):
        return self.logits.expand(ids.shape[0], 1, -1).clone(), None


class ResponseSamplingTests(unittest.TestCase):
    def test_nucleus_keeps_boundary_token_and_always_at_least_one(self):
        model = FixedModel(torch.tensor([0.5, 0.25, 0.15, 0.1]).log().tolist())
        seen = []
        def choose(probabilities, num_samples):
            seen.append(probabilities)
            return probabilities.argmax(-1, keepdim=True)
        with patch('torch.multinomial', side_effect=choose):
            model.generate(torch.tensor([[3]]), 1, 1, 0, top_p=0.6)
            model.generate(torch.tensor([[3]]), 1, 1, 0, top_p=0.1)
            model.generate(torch.tensor([[3]]), 1, 1, 2, top_p=1)
        torch.testing.assert_close(seen[0], torch.tensor([[2/3, 1/3, 0, 0]]))
        torch.testing.assert_close(seen[1], torch.tensor([[1., 0, 0, 0]]))
        torch.testing.assert_close(seen[2], seen[0])

    def test_penalty_affects_recent_generation_not_prompt_and_handles_negative_logits(self):
        for logits, penalty in (([2, 1.8, -1], 1.2), ([-0.1, -0.11, -1], 2)):
            result = FixedModel(logits).generate(torch.tensor([[0]]), 2, 0, repetition_penalty=penalty)
            self.assertEqual(result.tolist(), [[0, 0, 1]])
        result = FixedModel([10, 9, 1]).generate(torch.tensor([[0]]), 1, 0, forbidden_ids=(0, 1))
        self.assertEqual(result.tolist(), [[0, 2]])

    def test_generator_removes_emitted_urls_with_existing_checkpoint_interface(self):
        generator = Generator.__new__(Generator)
        generator.device = 'cpu'
        generator.model = FixedModel([0, 0, 1, 0])
        generator.tokenizer = SimpleNamespace(token_to_id=lambda token: {'<|pad|>': 0, '<|bos|>': 1, '<|eos|>': 3}[token],
            encode=lambda text: SimpleNamespace(ids=[2]), decode=lambda ids: ' A useful explanation. See https://example.org/page')
        result = generator.complete('Topic:', max_new_tokens=2, temperature=0)
        self.assertEqual(result['completion'], ' A useful explanation. See')
        self.assertTrue(result['text'].startswith('Topic: '))
        self.assertNotIn('example.org', result['text'])
        self.assertEqual(result['generated_tokens'], 2)

    def test_response_defaults_match_config_and_invalid_sampling_is_rejected(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/response.json').read_text())
        request = TrainRequest().model_dump(exclude={'data', 'out', 'resume'})
        self.assertEqual({key: request[key] for key in config}, config)
        self.assertEqual({key: request[key] for key in ('architecture', 'gradient_checkpointing', 'optimizer_foreach', 'precision')},
                         dict(architecture='gpt', gradient_checkpointing=False, optimizer_foreach=None, precision='auto'))
        self.assertEqual(request['context'] * request['batch_size'] * request['accumulation'], 8192)
        self.assertEqual(CompletionRequest().top_p, 0.9)
        for kwargs in ({'top_p': 0}, {'top_p': float('nan')}, {'repetition_penalty': 0.5}):
            with self.assertRaises(ValueError):
                FixedModel([1, 2]).generate(torch.tensor([[0]]), **kwargs)
            with self.assertRaises(ValueError):
                CompletionRequest(**kwargs)


if __name__ == '__main__':
    unittest.main()
