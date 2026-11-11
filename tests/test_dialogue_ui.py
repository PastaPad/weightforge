"""Real browser/API integration with intercepted jobs and a fake generator; no training."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.dialogue import CHAT_FORMAT
from scratchllm.jobs import Jobs, PipelineRequest
from scratchllm.serve import create_app
import test_dialogue
from test_serving_selection import FakeGenerator
from kiwix_fixture import make_archive
from scratchllm.kiwix import DEFAULT_PATH


class FakeChatGenerator(FakeGenerator):
    chat_format = CHAT_FORMAT

    def chat(self, prompt, history, **kwargs):
        return dict(text='Fixture answer.', completion='Fixture answer.', generated_tokens=3,
                    prompt_tokens_used=10, history_messages_used=len(history))


class DialogueUITests(unittest.TestCase):
    def test_dialogue_forms_five_stage_pipeline_and_multiturn_chat(self):
        asyncio.run(self.exercise_browser())

    async def exercise_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            checkpoint, _, _, meta = test_dialogue.DialogueTests().prepared(root)
            jobs = Jobs(root, 'cpu')
            state, calls, errors = {'job': None}, [], []
            with patch('scratchllm.serve.Generator', FakeChatGenerator), \
                    patch.object(Jobs, 'start', side_effect=AssertionError('Unexpected training launch')) as spawn, \
                    TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(12000)
                    page.on('pageerror', lambda error: errors.append(str(error)))

                    async def route(route):
                        incoming = route.request
                        path = urlsplit(incoming.url).path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        if incoming.method == 'POST':
                            calls.append((path, payload))
                        if path == '/jobs':
                            await route.fulfill(json=state)
                            return
                        if path == '/jobs/pipeline':
                            resolved = jobs._pipeline_request(PipelineRequest.model_validate(payload), 'ui-dialogue')
                            outputs = dict(corpus=resolved.crawl.output, dataset=resolved.prepare.output,
                                           base_run=resolved.train.out, dialogue_dataset=resolved.dialogue.output,
                                           run=resolved.finetune.out, checkpoint=resolved.finetune.out + '/best.pt')
                            state['job'] = dict(id='ui-dialogue', kind='pipeline', status='running', output=outputs['run'],
                                                log_path='fixture.log', log='Dialogue fixture running', settings=resolved.model_dump(),
                                                pipeline=dict(stage='finetune', stage_index=5, outputs=outputs,
                                                    stages=[dict(kind=kind, status='running' if kind == 'finetune' else 'succeeded') for kind, _ in resolved.stages()],
                                                    dialogue_report=meta['dialogue_report']))
                            await route.fulfill(json=state)
                            return
                        if path == '/jobs/stop':
                            state['job']['status'] = 'stopped'
                            state['job']['pipeline']['stages'][-1]['status'] = 'stopped'
                            await route.fulfill(json=state)
                            return
                        if path.startswith('/jobs/'):
                            await route.fulfill(status=400, json={'detail': 'Job intercepted by test'})
                            return
                        response = client.request(incoming.method, path, json=payload)
                        await route.fulfill(status=response.status_code, body=response.content,
                                            content_type=response.headers.get('content-type', 'application/json'))

                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#prompt').fill('First question?')
                    await page.locator('#generate').click()
                    await page.wait_for_function("document.getElementById('result').textContent.includes('Fixture answer.')")
                    await page.locator('#prompt').fill('Follow-up?')
                    await page.locator('#generate').click()
                    await page.wait_for_function("document.getElementById('result').textContent.includes('Follow-up?')")
                    chats = [payload for path, payload in calls if path == '/chat']
                    self.assertEqual(len(chats), 2)
                    self.assertEqual(chats[0]['history'], [])
                    self.assertEqual([message['role'] for message in chats[1]['history']], ['user', 'assistant'])
                    await page.locator('#clear').click()
                    await page.locator('#train-tab').click()
                    await page.locator('#dialogue-use-serving').click()
                    self.assertEqual(await page.locator('#dialogue-base').input_value(), str(checkpoint))
                    await page.locator('#dialogue-output').fill('data/dialogue')
                    await page.locator('#dialogue-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('error').textContent.includes('intercepted')")
                    payload = next(payload for path, payload in calls if path == '/jobs/dialogue')
                    self.assertEqual(payload['sources'], ['oasst1', 'dolly15k'])
                    self.assertEqual(payload['checkpoint'], str(checkpoint))
                    await page.locator('#preview-finetuning').click()
                    await page.wait_for_function("document.getElementById('finetuning-plan').textContent.includes('256 context')")
                    self.assertIn('assistant-only', await page.locator('#finetuning-plan').inner_text())
                    await page.locator('#finetune-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('error').textContent.includes('intercepted')")
                    payload = next(payload for path, payload in calls if path == '/jobs/finetune')
                    self.assertEqual(payload['learning_rate'], 0.00005)
                    self.assertEqual(payload['max_epochs'], 3)
                    self.assertEqual(payload['data'], 'data/dialogue')
                    await page.locator('#resume-finetune').click()
                    await page.wait_for_function("document.getElementById('error').textContent.includes('intercepted')")
                    self.assertTrue([payload for path, payload in calls if path == '/jobs/finetune'][-1]['resume'])
                    self.assertTrue(await page.locator('#pipeline-dialogue').is_checked())
                    await page.locator('#hosts input[value="wikipedia.org"]').check()
                    await page.locator('#start-pipeline').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stage 5/5: finetune')")
                    self.assertIn('frozen base vocabulary', await page.locator('#dialogue-report').inner_text())
                    payload = next(payload for path, payload in calls if path == '/jobs/pipeline')
                    self.assertIn('wikipedia.org', payload['crawl']['hosts'])
                    self.assertEqual(payload['dialogue']['sources'], ['oasst1', 'dolly15k'])
                    self.assertEqual(await page.locator('#finetune-data').input_value(), 'data/dialogue-pipeline-ui-dialogue')
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stopped')")
                    self.assertFalse(errors, errors)
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
