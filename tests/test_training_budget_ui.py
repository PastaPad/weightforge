"""Real-browser budget regression test; HTTP is routed to a fake-model API, training is intercepted."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.serve import create_app
from scratchllm.jobs import Jobs
from test_serving_selection import FakeGenerator


class TrainingBudgetUITests(unittest.TestCase):
    def test_dataset_selection_auto_budget_and_stale_responses(self):
        asyncio.run(self.exercise_browser())

    async def exercise_browser(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', side_effect=AssertionError('Unexpected worker launch')) as spawn:
            root = Path(directory)
            (root/'model').mkdir()
            (root/'model/best.pt').write_text('fake')
            (root/'model/web-job.json').write_text(json.dumps({'data': 'data/old'}))

            def dataset(name, tokens, modified):
                folder = root/'data'/name
                folder.mkdir(parents=True)
                meta = dict(vocab_size=16384, tokens=dict(train=tokens, val=16000), documents=dict(train=937, val=106))
                (folder/'tokenizer.json').write_text('{}')
                for split, count in meta['tokens'].items():
                    with (folder/f'{split}.bin').open('wb') as stream:
                        stream.truncate(4 * count)
                (folder/'meta.json').write_text(json.dumps(meta))
                os.utime(folder/'meta.json', (modified, modified))

            dataset('old', 101000, 10)
            dataset('larger', 1587273, 20)
            stale_entered, release_stale, stale_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
            delay_old = False
            job, submitted, errors = {'job': None}, [], []
            with TestClient(create_app(root/'model/best.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.on('pageerror', lambda error: errors.append(str(error)))

                    async def route(request):
                        nonlocal delay_old
                        incoming = request.request
                        path = urlsplit(incoming.url).path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        if path == '/jobs':
                            await request.fulfill(json=job)
                            return
                        if path.startswith('/jobs/'):
                            self.assertEqual(path, '/jobs/train')
                            submitted.append(payload)
                            await request.fulfill(status=400, json={'detail': 'Training intercepted by test'})
                            return
                        response = client.request(incoming.method, path, json=payload)
                        delayed = delay_old and path == '/training/plan' and payload['data'] == 'data/old'
                        if delayed:
                            delay_old = False
                            stale_entered.set()
                            await asyncio.wait_for(release_stale.wait(), 10)
                        await request.fulfill(status=response.status_code, body=response.content,
                                              content_type=response.headers.get('content-type', 'application/json'))
                        if delayed:
                            stale_finished.set()

                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('training-plan').textContent.includes('25 requested passes')")
                    self.assertEqual(await page.locator('#train-data').input_value(), 'data/larger')
                    self.assertEqual(await page.locator('#loaded-dataset').inner_text(), 'data/old')

                    await page.locator('#train-budget-mode').select_option('passes')
                    await page.locator('#train-epochs').fill('20')

                    async def updates(number):
                        await page.wait_for_function('(value) => document.getElementById("train-steps").value === value', arg=str(number))

                    await updates(3876)
                    await page.locator('#train-accum').fill('2')
                    await updates(15501)
                    await page.locator('#train-data').fill('data/old')
                    await updates(987)
                    await page.locator('#train-data').fill('data/larger')
                    await updates(15501)
                    await page.locator('#train-accum').fill('8')
                    await updates(3876)

                    # A response for the old dataset must not replace the newer calculation.
                    delay_old = True
                    await page.locator('#train-data').fill('data/old')
                    await asyncio.wait_for(stale_entered.wait(), 5)
                    await page.locator('#train-data').fill('data/larger')
                    await updates(3876)
                    release_stale.set()
                    await asyncio.wait_for(stale_finished.wait(), 5)
                    await page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                    self.assertEqual(await page.locator('#train-steps').input_value(), '3876')

                    # Fixed mode stays fixed, with an explanation instead of implicit resizing.
                    await page.locator('#train-budget-mode').select_option('updates')
                    await page.locator('#train-epochs').fill('0')
                    await page.locator('#train-steps').fill('987')
                    await page.locator('#train-data').fill('data/old')
                    await page.wait_for_function("document.getElementById('training-plan').textContent.includes('fixed update limit;')")
                    await page.locator('#train-data').fill('data/larger')
                    await page.wait_for_function("document.getElementById('training-plan').textContent.startsWith('data/larger\\n') && document.getElementById('training-plan').textContent.includes('987 planned updates')")
                    await page.locator('#train-budget-mode').select_option('passes')
                    await page.locator('#train-epochs').fill('20')
                    await updates(3876)

                    # A newly completed preparation is selected only after it succeeds.
                    job['job'] = dict(id='prepare-new', kind='prepare', status='running', output='data/newest', log_path='fixture.log', log='Preparing')
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('running')")
                    self.assertEqual(await page.locator('#train-data').input_value(), 'data/larger')
                    dataset('newest', 3000001, 30)
                    job['job']['status'] = 'succeeded'
                    await updates(7325)
                    self.assertEqual(await page.locator('#train-data').input_value(), 'data/newest')

                    # Verify submission carries the displayed dataset/count, without launching it.
                    await page.locator('#train-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('error').textContent.includes('Training intercepted by test')")
                    self.assertEqual(len(submitted), 1)
                    self.assertEqual(submitted[0]['budget_mode'], 'passes')
                    self.assertEqual(submitted[0]['data'], 'data/newest')
                    self.assertEqual(submitted[0]['steps'], 7325)
                    self.assertFalse(errors, errors)
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
