"""Memory persistence, isolation, bounded context and API contracts; no training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from scratchllm.dialogue import CHAT_FORMAT, prompt_tokens
from scratchllm.generate import Generator
from scratchllm.memory import MemoryStore
from scratchllm.serve import create_app
from scratchllm.workspace import assemble, chat
from test_dialogue import fixture_base


class MemoryWorkspaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def generator(self, root, context=512):
        checkpoint, _, _ = fixture_base(root, context=context)
        state = torch.load(checkpoint, weights_only=True)
        state['chat_format'] = CHAT_FORMAT
        torch.save(state, checkpoint)
        return Generator(checkpoint, 'cpu'), checkpoint

    def test_disk_retrieval_session_isolation_feedback_capacity_and_stale_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'memory.sqlite3'
            store = MemoryStore(path, max_episodes=3)
            self.assertFalse(path.exists())
            first = store.record('alice', 'Tell me about Saturn https://example.org', 'Saturn is a planet.', 'base.pt', {}, 'astronomy', 0)
            store.record('bob', 'Saturn private', 'Other session', 'other.pt', {}, '', 0)
            restored = MemoryStore(path, max_episodes=3)
            self.assertEqual(restored.retrieve('alice', 'Saturn')[0]['id'], first)
            self.assertEqual(restored.retrieve('bob', 'Saturn')[0]['assistant_text'], 'Other session')
            self.assertNotIn('https://', restored.snapshot('alice')['recent'][0]['user_text'])
            self.assertEqual(restored.snapshot('alice')['goal'], 'astronomy')
            self.assertEqual(restored.retrieve('alice', 'Saturn', exclude=(first,)), [])
            restored.feedback('alice', first, False, 'User correction')
            self.assertEqual(restored.retrieve('alice', 'Saturn'), [])
            with self.assertRaises(ValueError):
                restored.feedback('bob', first, True)
            with self.assertRaises(ValueError):
                restored.record('alice', 'x', 'y', 'base', {}, '', 1)
            for revision in range(2, 6):
                restored.record('alice', f'Saturn {revision}', f'Answer {revision}', 'base', {}, '', revision)
            self.assertEqual(restored.snapshot('alice')['feedback']['total'], 3)
            self.assertEqual(len(restored.retrieve('alice', '"Saturn" OR * NOT -- :')), 3)
            restored.forget('alice')
            self.assertEqual(restored.retrieve('alice', 'Saturn'), [])
            self.assertEqual(restored.snapshot('alice')['feedback']['total'], 0)
            with self.assertRaises(ValueError):
                restored.record('alice', 'x', 'y', 'base', {}, '', 6)
            self.assertEqual(restored.snapshot('bob')['feedback']['total'], 1)

    def test_workspace_budgets_context_and_review_and_excludes_unhelpful_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator, checkpoint = self.generator(root)
            store = MemoryStore(root/'memory.sqlite3')
            first = store.record('test', 'Saturn rings', 'Made of ice', str(checkpoint), {}, '', 0)
            bad = store.record('test', 'bad', 'wrong answer', str(checkpoint), {}, '', 1)
            store.feedback('test', bad, False)
            for revision in range(3, 8):
                store.record('test', f'Other turn {revision}', 'Short reply', str(checkpoint), {}, '', revision)
            calls = []
            def reply(prompt, history=None, **kwargs):
                self.assertTrue(kwargs['collect_stats'])
                messages = [*(history or []), dict(role='user', content=prompt)]
                self.assertLessEqual(len(prompt_tokens(generator.tokenizer, messages)), generator.model.cfg.context)
                self.assertNotIn('wrong answer', str(messages))
                self.assertIn('What are Saturn rings?', prompt)
                calls.append(prompt)
                return dict(completion='Ice and dust.', text='Ice and dust.', generated_tokens=4,
                            prompt_tokens_used=len(prompt_tokens(generator.tokenizer, messages)),
                            history_messages_used=len(history or []), diagnostics=dict(probability_is_factual_confidence=False))
            with patch.object(generator, 'chat', side_effect=reply):
                result = chat(generator, store, str(checkpoint), 'test', 'What are Saturn rings?', goal='Astronomy', review=True, max_new_tokens=32)
            self.assertEqual(len(calls), 2)
            self.assertIn('Draft to review: Ice and dust.', calls[1])
            self.assertTrue(result['workspace']['review']['performed'])
            self.assertEqual(result['total_generated_tokens'], 8)
            self.assertEqual(store.snapshot('test')['recent'][-1]['user_text'], 'What are Saturn rings?')
            self.assertEqual(store.snapshot('test')['goal'], 'Astronomy')
            self.assertEqual(result['workspace']['self_model']['tools'], [])
            self.assertFalse(result['workspace']['self_model']['factual_confidence_calibrated'])
            text, turns, used = assemble(generator, 'Question?', [], [(first, 'Saturn ' * 1000)], 64)
            self.assertIn(first, used)
            self.assertLessEqual(len(prompt_tokens(generator.tokenizer, [dict(role='user', content=text)])), 448)
            with self.assertRaises(ValueError):
                assemble(generator, 'Long question ' * 200, [], [], 64)

    def test_api_persists_across_restart_and_respects_gpu_reservation_without_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator, checkpoint = self.generator(root)
            with patch('scratchllm.serve.Generator', return_value=generator), patch.object(generator, '_generate', return_value={
                    'completion': 'Saved answer', 'generated_tokens': 3, 'prompt_tokens_used': 30,
                    'diagnostics': dict(probability_is_factual_confidence=False)}), \
                    patch('scratchllm.jobs.subprocess.Popen', side_effect=AssertionError('No worker may start')):
                with TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                    self.assertFalse((root/'runs/memory.sqlite3').exists())
                    self.assertTrue(client.get('/health').json()['features']['memory_workspace'])
                    preset = client.get('/architecture/presets').json()['presets'][0]
                    self.assertEqual(preset['parameters'], 41955840)
                    request = dict(prompt='Remember Saturn', session_id='persistent', goal='Learn astronomy', max_new_tokens=4)
                    response = client.post('/workspace/chat', json=request)
                    self.assertEqual(response.status_code, 200, response.text)
                    entry = response.json()['workspace']['episode_id']
                    self.assertIsNone(client.get('/jobs').json()['job'])
                    with patch('scratchllm.serve.Jobs.training', return_value=True):
                        self.assertEqual(client.post('/workspace/chat', json=request).status_code, 409)
                    self.assertEqual(client.post('/workspace/chat', json=dict(request, session_id='../bad')).status_code, 422)
                    generator.chat_format = None
                    self.assertEqual(client.post('/workspace/chat', json=request).status_code, 400)
                    generator.chat_format = CHAT_FORMAT
                with TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                    saved = client.get('/workspace/persistent').json()
                    self.assertEqual(saved['goal'], 'Learn astronomy')
                    self.assertEqual(saved['recent'][0]['assistant_text'], 'Saved answer')
                    response = client.post('/workspace/persistent/feedback', json=dict(episode_id=entry, helpful=False))
                    self.assertEqual(response.json()['feedback']['unhelpful'], 1)
                    self.assertEqual(client.post('/workspace/other/feedback', json=dict(episode_id=entry, helpful=True)).status_code, 400)
                    self.assertEqual(client.delete('/workspace/persistent').json()['feedback']['total'], 0)


if __name__ == '__main__':
    unittest.main()
