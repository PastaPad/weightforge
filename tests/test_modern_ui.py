"""Hardware preset and persistent-memory browser flow; all training launches forbidden."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.jobs import Jobs
from scratchllm.serve import create_app
import test_memory_workspace


class ModernUITests(unittest.TestCase):
    def test_hardware_preset_and_memory_reopen_feedback_and_old_server_gating(self):
        asyncio.run(self.browser())

    async def browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = test_memory_workspace.MemoryWorkspaceTests()
            helper.setUp()
            generator, checkpoint = helper.generator(root)
            data = root/'data/prepared'
            data.mkdir(parents=True)
            (data/'tokenizer.json').write_text('{}')
            (data/'meta.json').write_text(json.dumps(dict(vocab_size=8192, documents=dict(train=10, val=2),
                                                         tokens=dict(train=10001, val=2001))))
            for split, count in [('train', 10001), ('val', 2001)]:
                (data/f'{split}.bin').write_bytes(b'\0' * count * 4)
            errors, calls, state = [], [], {'old_server': False}
            with patch('scratchllm.serve.Generator', return_value=generator), \
                    patch.object(generator, '_generate', return_value=dict(completion='Saturn has rings.', text='Saturn has rings.',
                        prompt_tokens_used=40, generated_tokens=5, diagnostics=dict(probability_is_factual_confidence=False))), \
                    patch.object(Jobs, 'start', side_effect=AssertionError('No job may start')) as launch, \
                    TestClient(create_app(checkpoint, 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(10000)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    async def route(route):
                        incoming = route.request
                        path = urlsplit(incoming.url).path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        calls.append((incoming.method, path, payload))
                        response = client.request(incoming.method, path, json=payload)
                        if state['old_server'] and path == '/health':
                            value = response.json()
                            value.pop('features')
                            await route.fulfill(json=value)
                        else:
                            await route.fulfill(status=response.status_code, body=response.content,
                                                content_type=response.headers.get('content-type', 'application/json'))
                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#hardware-preset').click()
                    await page.wait_for_function("document.getElementById('training-plan').textContent.includes('41,955,840')")
                    self.assertEqual(await page.locator('#train-architecture').input_value(), 'modern')
                    self.assertEqual(await page.locator('#train-batch').input_value(), '1')
                    self.assertEqual(await page.locator('#train-accum').input_value(), '16')
                    self.assertEqual(await page.locator('#train-layers').input_value(), '12')
                    self.assertTrue(await page.locator('#train-checkpointing').is_checked())
                    self.assertTrue(await page.locator('#finetune-memory-saving').is_checked())
                    # An older automatic preview can arrive after the preset request.
                    payload = [body for _, path, body in calls if path == '/training/plan' and body.get('architecture') == 'modern'][-1]
                    self.assertEqual(payload['architecture'], 'modern')
                    self.assertEqual(payload['kv_heads'], 4)
                    self.assertFalse(payload['optimizer_foreach'])
                    self.assertEqual(payload['precision'], 'fp32')
                    self.assertEqual(payload['max_epochs'], 3)
                    self.assertFalse(any(path.startswith('/jobs/') for _, path, _ in calls))
                    await page.locator('#play-tab').click()
                    await page.locator('#generation-mode').select_option('workspace')
                    await page.wait_for_function("document.getElementById('workspace-info').textContent.includes('saved exchanges')")
                    session = await page.locator('#workspace-session').input_value()
                    await page.locator('#workspace-goal').fill('Learn astronomy')
                    await page.locator('#workspace-review').check()
                    await page.locator('#prompt').fill('What are Saturn rings?')
                    await page.locator('#generate').click()
                    await page.wait_for_function("document.getElementById('workspace-info').textContent.includes('revision 1')")
                    self.assertIn('"performed": true', await page.locator('#workspace-details').text_content())
                    self.assertIn('probability_is_factual_confidence', await page.locator('#workspace-details').text_content())
                    await page.locator('#workspace-inspector summary').click()
                    await page.locator('#feedback-unhelpful').click()
                    await page.wait_for_function("document.getElementById('workspace-info').textContent.includes('excluded')")
                    await page.reload(wait_until='networkidle')
                    await page.locator('#generation-mode').select_option('workspace')
                    await page.wait_for_function("document.getElementById('result').textContent.includes('Saturn has rings.')")
                    self.assertEqual(await page.locator('#workspace-session').input_value(), session)
                    self.assertEqual(await page.locator('#workspace-goal').input_value(), 'Learn astronomy')
                    self.assertIn('1 unhelpful', await page.locator('#workspace-info').inner_text())
                    await page.locator('#workspace-forget').click()
                    await page.wait_for_function("document.getElementById('workspace-info').textContent.includes('0 saved exchanges')")
                    self.assertEqual(await page.locator('#result').inner_text(), '')
                    state['old_server'] = True
                    calls.clear()
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertTrue(await page.locator('#hardware-preset').is_hidden())
                    self.assertTrue(await page.locator('#workspace-option').is_disabled())
                    self.assertIn('active job finishes', await page.locator('#hardware-preset-info').inner_text())
                    payload = [body for _, path, body in calls if path == '/training/plan'][-1]
                    self.assertNotIn('architecture', payload)
                    self.assertFalse(any(path == '/architecture/presets' for _, path, _ in calls))
                    self.assertFalse(errors, errors)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
