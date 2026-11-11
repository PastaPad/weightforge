"""Separate HF tab, shared scraper controls and user-started jobs; no model fitting."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.hf_requests import HFImportRequest, HFTrainRequest, HFExportRequest
from scratchllm.jobs import Jobs
from scratchllm.serve import create_app
from test_huggingface import fixture, legacy_splits
from test_serving_selection import FakeGenerator


class HuggingFaceUITests(unittest.TestCase):
    def test_import_prepare_preview_and_explicit_start_stop_resume(self):
        asyncio.run(self.browser())

    async def browser(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            fixture(root)
            legacy_splits(root/'data/raw.jsonl', root/'data/previous')
            calls, errors = [], []
            state = dict(job=None, old=False)
            start = Jobs.start
            def prepare_only(self, kind, request):
                if kind != 'hf-prepare':
                    raise AssertionError('This browser test may only run data preparation')
                return start(self, kind, request)
            with patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', prepare_only), \
                    TestClient(create_app(root/'missing.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(30000)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    async def route(route):
                        incoming = route.request
                        url = urlsplit(incoming.url)
                        path = url.path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        if incoming.method == 'POST':
                            calls.append((path, payload))
                        if state['old'] and ('huggingface' in path or '/jobs/hf-' in path):
                            raise AssertionError('Old backend received a Hugging Face request')
                        if path == '/huggingface/repository':
                            await route.fulfill(json=dict(repo='test/repo', revision='a' * 40, license='mit', files=[dict(
                                filename='tiny.gguf', bytes=100, parts=[dict(filename='tiny.gguf', bytes=100)])]))
                            return
                        if path == '/jobs/hf-import':
                            parsed = HFImportRequest.model_validate(payload)
                            state['job'] = dict(id='fixture-import', kind='hf-import', status='succeeded',
                                output='models/base', hf_ready=True, settings=parsed.model_dump(), log='Fixture import intercepted')
                        elif path == '/jobs/crawl':
                            state['job'] = dict(id='fixture-crawl', kind='crawl', status='succeeded', output=payload['output'],
                                settings=payload, log='Collection intercepted')
                        elif path == '/jobs/hf-prepare':
                            state['job'] = None
                        elif path == '/jobs/hf-finetune':
                            parsed = HFTrainRequest.model_validate(payload)
                            state['job'] = dict(id='fixture-training', kind='hf-finetune', status='running',
                                output=parsed.out, settings=parsed.model_dump(), log='Training intercepted', metrics=[])
                        elif path == '/jobs/stop' and state['job']:
                            state['job']['status'] = 'stopped'
                        elif path == '/jobs/hf-merge':
                            state['job'] = dict(id='fixture-merge', kind='hf-merge', status='succeeded',
                                output=payload['output'], settings=payload, hf_ready=True, log='Merge intercepted')
                        elif path == '/jobs/hf-export':
                            parsed = HFExportRequest.model_validate(payload)
                            state['job'] = dict(id='fixture-export', kind='hf-export', status='running',
                                output=parsed.output, settings=parsed.model_dump(), log='Export intercepted', export_ready=False)
                        if state['job'] and path in {'/jobs', '/jobs/stop', '/jobs/hf-import', '/jobs/hf-finetune', '/jobs/crawl', '/jobs/hf-merge', '/jobs/hf-export'}:
                            await route.fulfill(json=dict(job=state['job']))
                            return
                        response = client.request(incoming.method, path + ('?' + url.query if url.query else ''), json=payload)
                        if path == '/health' and state['old']:
                            data = response.json()
                            data['features'].pop('huggingface')
                            await route.fulfill(json=data)
                        else:
                            await route.fulfill(status=response.status_code, body=response.content,
                                                content_type=response.headers.get('content-type', 'application/json'))
                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    self.assertIn('ONLINE', await page.locator('#connection').inner_text())
                    self.assertEqual(await page.locator('#loaded-checkpoint').inner_text(), 'No native checkpoint loaded')
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    await page.locator('#hf-tab').click()
                    self.assertFalse(await page.locator('#huggingface-panel').is_hidden())
                    self.assertTrue(await page.locator('#training-panel').is_hidden())
                    await page.locator('#hf-repo').fill('test/repo')
                    await page.locator('#hf-browse').click()
                    await page.wait_for_function("document.getElementById('hf-file').value === 'tiny.gguf'")
                    await page.locator('#hf-import-output').fill('models/base')
                    await page.locator('#hf-import-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('HF-IMPORT · succeeded')")
                    self.assertEqual(next(body for path, body in calls if path == '/jobs/hf-import')['revision'], 'a' * 40)
                    await page.locator('#hf-use-job-output').click()
                    self.assertEqual(await page.locator('#hf-model').input_value(), 'models/base')
                    await page.locator('#hf-hosts input[value="soyjak.st"]').check()
                    self.assertTrue(await page.locator('#hosts input[value="soyjak.st"]').is_checked())
                    await page.locator('#hf-corpus').fill('data/raw.jsonl')
                    await page.locator('#hf-collect').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('CRAWL · succeeded')")
                    collected = next(body for path, body in calls if path == '/jobs/crawl')
                    self.assertIn('soyjak.st', collected['hosts'])
                    self.assertTrue(collected['ignore_robots'])
                    self.assertEqual(collected['output'], 'data/raw.jsonl')
                    await page.locator('#hf-use-job-output').click()
                    self.assertEqual(await page.locator('#hf-prepare-input').input_value(), 'data/raw.jsonl')
                    # Section 03 selects an existing prepared file independently of collection.
                    self.assertEqual(await page.locator('#hf-corpora option[value="data/previous/train.jsonl"]').count(), 1)
                    await page.locator('#hf-corpus').fill('data/unselected-collection.jsonl')
                    await page.locator('#hf-prepare-input').fill(str(root/'data/previous/train.jsonl'))
                    await page.locator('#hf-refresh').click()
                    await page.locator('#hf-data-output').fill('data/ready')
                    await page.locator('#hf-context').fill('32')
                    await page.locator('#hf-prepare-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('HF-PREPARE · succeeded')")
                    self.assertEqual(next(body for path, body in calls if path == '/jobs/hf-prepare')['input'], str(root/'data/previous/train.jsonl'))
                    self.assertFalse(any(path == '/jobs/hf-finetune' for path, _ in calls))
                    await page.locator('#hf-use-job-output').click()
                    self.assertEqual(await page.locator('#hf-data').input_value(), 'data/ready')
                    self.assertEqual(await page.locator('#hf-train-model').input_value(), 'models/base')
                    await page.locator('#hf-device').select_option('cpu')
                    self.assertEqual(await page.locator('#hf-mode').input_value(), 'lora')
                    await page.locator('#hf-preview').click()
                    await page.wait_for_function("document.getElementById('hf-plan').textContent.includes('32 context')")
                    self.assertIn('soyjak.st', await page.locator('#hf-plan').inner_text())
                    self.assertFalse(any(path == '/jobs/hf-finetune' for path, _ in calls))
                    await page.locator('#hf-train-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('HF-FINETUNE · running') && !document.getElementById('hf-stop').disabled")
                    self.assertTrue(await page.locator('#hf-collect').is_disabled())
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    await page.locator('#hf-stop').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('stopped') && !document.getElementById('hf-resume').disabled")
                    await page.locator('#hf-resume').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('running')")
                    self.assertEqual([body for path, body in calls if path == '/jobs/hf-finetune'][-1], dict(out='runs/huggingface', resume=True))
                    await page.locator('#hf-stop').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('stopped')")
                    await page.locator('#hf-result-run').fill('runs/huggingface')
                    await page.locator('#hf-merge-output').fill('models/merged')
                    await page.locator('#hf-merge-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('HF-MERGE · succeeded')")
                    await page.locator('#hf-use-job-output').click()
                    self.assertEqual(await page.locator('#hf-export-model').input_value(), 'models/merged')
                    await page.locator('#hf-export-precision').select_option('q8_0')
                    await page.locator('#hf-export-output').fill('exports/merged-q8.gguf')
                    await page.locator('#hf-export-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('hf-job-summary').textContent.includes('HF-EXPORT · running')")
                    self.assertTrue(await page.locator('#hf-gguf-download').is_hidden())
                    self.assertEqual(next(body for path, body in calls if path == '/jobs/hf-export'), dict(model='models/merged', output='exports/merged-q8.gguf', precision='q8_0'))
                    state['job'].update(status='succeeded', export_ready=True)
                    await page.locator('#hf-tab').click()
                    await page.wait_for_function("!document.getElementById('hf-gguf-download').hidden")
                    self.assertEqual(await page.locator('#hf-gguf-download').get_attribute('href'), '/huggingface/gguf?job_id=fixture-export')
                    state.update(old=True, job=None)
                    await page.reload(wait_until='networkidle')
                    await page.locator('#hf-tab').click()
                    self.assertTrue(await page.locator('#hf-browse').is_disabled())
                    self.assertTrue(await page.locator('#hf-train-form button[type=submit]').is_disabled())
                    self.assertTrue(await page.locator('#hf-export-form button[type=submit]').is_disabled())
                    self.assertIn('staged', await page.locator('#hf-status').inner_text())
                    await page.locator('#train-tab').click()
                    self.assertFalse(await page.locator('#training-panel').is_hidden())
                    self.assertFalse(errors, errors)


if __name__ == '__main__':
    unittest.main()
