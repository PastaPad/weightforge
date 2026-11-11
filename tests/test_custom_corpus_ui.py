"""Browser upload → real CPU preparation → preview; training starts are intercepted."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.jobs import FineTuneRequest, Jobs
from scratchllm.serve import create_app
from test_custom_corpus import fixture_corpus
from test_serving_selection import FakeGenerator


class CustomCorpusUITests(unittest.TestCase):
    def test_upload_prepare_reopen_preview_and_explicit_start_stop_resume(self):
        asyncio.run(self.browser())

    async def browser(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            checkpoint, source, _, _ = fixture_corpus(root, chat=True)
            calls, errors = [], []
            state = dict(training=None, old=False)
            start = Jobs.start
            def prepare_only(self, kind, request):
                if kind != 'custom-corpus':
                    raise AssertionError('Model training is not permitted in this test')
                return start(self, kind, request)
            with patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', prepare_only), \
                    TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(20000)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    async def route(route):
                        incoming = route.request
                        url = urlsplit(incoming.url)
                        path = url.path
                        if path == '/corpus/upload':
                            calls.append((path, None))
                            response = client.post(path + '?' + url.query, content=incoming.post_data_buffer,
                                                   headers={'Content-Type': 'application/octet-stream'})
                        else:
                            payload = json.loads(incoming.post_data) if incoming.post_data else None
                            if incoming.method == 'POST':
                                calls.append((path, payload))
                            if path == '/jobs/finetune':
                                parsed = FineTuneRequest.model_validate(payload)
                                state['training'] = dict(id='fixture-finetuning', kind='finetune', status='running',
                                    output=parsed.out, settings=parsed.model_dump(), log_path='fixture.log', log='Training intercepted', metrics=[])
                                await route.fulfill(json=dict(job=state['training']))
                                return
                            if path == '/jobs/stop' and state['training']:
                                state['training']['status'] = 'stopped'
                                await route.fulfill(json=dict(job=state['training']))
                                return
                            if path == '/jobs' and state['training']:
                                await route.fulfill(json=dict(job=state['training']))
                                return
                            if path == '/health' and state['old']:
                                data = client.get(path).json()
                                data['features'].pop('custom_corpus')
                                await route.fulfill(json=data)
                                return
                            if state['old'] and path in {'/corpus/uploads', '/corpus/upload', '/jobs/custom-corpus'}:
                                raise AssertionError('Old backend received a custom corpus request')
                            response = client.request(incoming.method, path, json=payload)
                        await route.fulfill(status=response.status_code, body=response.content,
                                            content_type=response.headers.get('content-type', 'application/json'))
                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#custom-use-serving').click()
                    self.assertEqual(await page.locator('#custom-base').input_value(), str(checkpoint))
                    await page.locator('#custom-file').set_input_files(source)
                    await page.locator('#custom-output').fill('data/custom')
                    await page.locator('#custom-val').fill('0.25')
                    await page.locator('#prepare-custom').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('CUSTOM-CORPUS · succeeded')")
                    await page.wait_for_function("!document.getElementById('use-custom-dataset').hidden")
                    self.assertFalse(any(path == '/jobs/finetune' for path, _ in calls))
                    self.assertEqual(client.get('/jobs').json()['job']['kind'], 'custom-corpus')
                    uploaded = await page.locator('#custom-input').input_value()
                    self.assertEqual((root/uploaded).read_bytes(), source.read_bytes())
                    self.assertEqual(await page.locator('#custom-uploads option').count(), 1)
                    self.assertEqual(await page.locator('#finetune-base').input_value(), str(checkpoint))
                    self.assertEqual(await page.locator('#finetune-data').input_value(), 'data/custom')
                    self.assertIn('all-text next-token loss', await page.locator('#custom-report').inner_text())
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#use-custom-dataset').click()
                    self.assertEqual(await page.locator('#finetune-data').input_value(), 'data/custom')
                    self.assertEqual(await page.locator('#finetune-base').input_value(), str(checkpoint))
                    await page.locator('#finetune-passes').fill('1.5')
                    await page.locator('#finetune-batch').fill('1')
                    await page.locator('#finetune-accum').fill('2')
                    await page.locator('#preview-finetuning').click()
                    await page.wait_for_function("document.getElementById('finetuning-plan').textContent.includes('256 context')")
                    self.assertIn('all-text next-token loss', await page.locator('#finetuning-plan').inner_text())
                    self.assertFalse(any(path == '/jobs/finetune' for path, _ in calls))
                    await page.locator('#finetune-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('FINETUNE · running') && !document.getElementById('stop-job').disabled")
                    payload = next(payload for path, payload in calls if path == '/jobs/finetune')
                    self.assertEqual(payload['checkpoint'], str(checkpoint))
                    self.assertEqual(payload['data'], 'data/custom')
                    self.assertEqual(payload['max_epochs'], 1.5)
                    self.assertEqual(payload['accumulation'], 2)
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    self.assertTrue(await page.locator('#prepare-custom').is_disabled())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stopped') && !document.getElementById('resume-finetune').disabled")
                    await page.locator('#resume-finetune').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('running')")
                    self.assertTrue([payload for path, payload in calls if path == '/jobs/finetune'][-1]['resume'])
                    state.update(old=True, training=None)
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertTrue(await page.locator('#prepare-custom').is_disabled())
                    self.assertIn('staged', await page.locator('#custom-availability').inner_text())
                    self.assertFalse(errors, errors)


if __name__ == '__main__':
    unittest.main()
