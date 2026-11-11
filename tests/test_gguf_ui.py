"""Independent export controls during an active training reservation; no training."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.jobs import ExportRequest, Jobs
from scratchllm.serve import create_app
from test_serving_selection import FakeGenerator


class GGUFUITests(unittest.TestCase):
    def test_export_is_enabled_during_training_and_has_separate_stop_and_download(self):
        asyncio.run(self.browser())

    async def browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'runs/chosen').mkdir(parents=True)
            (root/'runs/chosen/best.pt').write_text('fixture checkpoint')
            (root/'runs/chosen/last.pt').write_text('fixture resume state')
            state = dict(export=None, old_server=False)
            errors, calls = [], []
            training = dict(job=dict(id='training-fixture', kind='train', status='running', output='runs/active',
                                    log_path='fixture.log', log='step 10/100 objective=4.0', metrics=[]))
            with patch('scratchllm.serve.Generator', FakeGenerator), \
                    patch.object(Jobs, 'start', side_effect=AssertionError('No real workers in browser test')), \
                    TestClient(create_app(root/'runs/chosen/best.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(10000)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    async def route(route):
                        incoming = route.request
                        path = urlsplit(incoming.url).path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        calls.append((incoming.method, path, payload))
                        if path == '/jobs':
                            await route.fulfill(json=training)
                            return
                        if path == '/health':
                            data = client.get(path).json()
                            data['training'] = True
                            if state['old_server']:
                                data['features'].pop('gguf_export')
                            await route.fulfill(json=data)
                            return
                        if path == '/exports' and incoming.method == 'POST':
                            settings = ExportRequest.model_validate(payload).model_dump()
                            state['export'] = dict(id='export-fixture', kind='export', status='running', settings=settings,
                                output=settings['output'], log='Exporting on CPU', progress=dict(phase='converting',
                                    completed=1, budget=20, architecture='llama', precision=settings['precision'], detail='Writing tensors'))
                        if path == '/exports/stop':
                            state['export']['status'] = 'stopped'
                        if path in {'/exports', '/exports/stop'}:
                            await route.fulfill(json={'job': state['export']})
                            return
                        if path.startswith('/jobs/'):
                            raise AssertionError('Export touched the training worker')
                        response = client.request(incoming.method, path, json=payload)
                        await route.fulfill(status=response.status_code, body=response.content,
                                            content_type=response.headers.get('content-type', 'application/json'))
                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    self.assertFalse(await page.locator('#start-export').is_disabled())
                    self.assertEqual(await page.locator('#export-checkpoints option').count(), 1)
                    await page.locator('#load-checkpoint').fill('runs/chosen/best.pt')
                    await page.locator('#export-use-selection').click()
                    await page.locator('#export-precision').select_option('f32')
                    await page.locator('#export-output').fill('exports/chosen.gguf')
                    await page.locator('#start-export').click()
                    await page.wait_for_function("document.getElementById('export-summary').textContent.includes('1/20 tensors') && !document.getElementById('stop-export').disabled")
                    self.assertIn('TRAIN · running', await page.locator('#job-summary').inner_text())
                    self.assertFalse(await page.locator('#stop-export').is_disabled())
                    payload = next(body for method, path, body in calls if method == 'POST' and path == '/exports')
                    self.assertEqual(payload, dict(checkpoint='runs/chosen/best.pt', output='exports/chosen.gguf', precision='f32'))
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertEqual(await page.locator('#export-output').input_value(), 'exports/chosen.gguf')
                    await page.locator('#stop-export').click()
                    await page.wait_for_function("document.getElementById('export-summary').textContent.includes('STOPPED')")
                    self.assertFalse(any(path == '/jobs/stop' for _, path, _ in calls))
                    await page.locator('#start-export').click()
                    await page.wait_for_function("document.getElementById('export-summary').textContent.includes('RUNNING')")
                    state['export'].update(status='succeeded', export_ready=True)
                    state['export']['progress'].update(phase='completed', completed=20, bytes=10000, sha256='fixture-hash')
                    await page.wait_for_function("!document.getElementById('export-download').hidden")
                    self.assertEqual(await page.locator('#export-download').get_attribute('href'), '/exports/file?job_id=export-fixture')
                    self.assertIn('TRAIN · running', await page.locator('#job-summary').inner_text())
                    state['old_server'] = True
                    calls.clear()
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertTrue(await page.locator('#start-export').is_disabled())
                    self.assertIn('active job finishes', await page.locator('#export-availability').inner_text())
                    self.assertFalse(any(path.startswith('/exports') for _, path, _ in calls))
                    self.assertFalse(errors, errors)


if __name__ == '__main__':
    unittest.main()
