"""Browser checks for ZIM selection/download/progress and collection; workers intercepted."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.jobs import Jobs, CrawlRequest, KiwixRequest, DatasetRequest
from scratchllm.kiwix import DEFAULT_PATH, BASE_URL, DEFAULT_FILE
from scratchllm.serve import create_app
from kiwix_fixture import make_archive
from test_serving_selection import FakeGenerator


class KiwixUITests(unittest.TestCase):
    def test_download_monitor_stop_resume_and_select_archive_for_collection(self):
        asyncio.run(self.browser())

    async def browser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            (root/'model.pt').write_text('fake')
            state, calls, errors = {'job': None}, [], []
            resolver = Jobs(root, 'cpu')
            with patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', side_effect=AssertionError('Unexpected worker launch')) as spawn, \
                    TestClient(create_app(root/'model.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(10000)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    async def route(route):
                        request = route.request
                        parsed = urlsplit(request.url)
                        path = parsed.path
                        if path == '/jobs':
                            await route.fulfill(json=state)
                            return
                        if path == '/wikipedia/catalog':
                            await route.fulfill(json={'archives': [dict(name=DEFAULT_FILE, url=BASE_URL + DEFAULT_FILE, size='2.6G')]})
                            return
                        if path.startswith('/jobs/'):
                            payload = json.loads(request.post_data or '{}')
                            calls.append((path, payload))
                            if path == '/jobs/kiwix':
                                KiwixRequest.model_validate(payload)
                                state['job'] = dict(id='download', kind='kiwix', status='running', output=payload['output'],
                                    log_path='download.log', log='Downloading archive', settings=payload,
                                    progress=dict(phase='downloading', heartbeat_at=1, detail='Downloading archive', downloaded_bytes=1024**3, total_bytes=2*1024**3))
                            elif path == '/jobs/stop':
                                state['job']['status'] = 'stopped'
                                if state['job']['kind'] == 'dataset':
                                    state['job']['resumable_dataset'] = True
                            elif path == '/jobs/crawl':
                                settings = CrawlRequest.model_validate(payload).model_dump()
                                state['job'] = dict(id='collect', kind='crawl', status='running', output=payload['output'],
                                    log_path='collect.log', log='Reading offline Wikipedia', settings=settings,
                                    progress=dict(phase='importing_zim', mode='kiwix', heartbeat_at=1, detail='Reading offline Wikipedia',
                                        completed=2, budget=10, saved=2, skipped=0, failed=0, queued=16, archive_cursor=2, archive={'entries': 18}))
                            elif path == '/jobs/zim-dataset':
                                request = DatasetRequest.model_validate(payload)
                                settings = request.model_dump() if request.crawl.resume else resolver._dataset_request(request, 'whole-fixture').model_dump()
                                state['job'] = dict(id='whole-fixture-resumed' if request.crawl.resume else 'whole-fixture', kind='dataset', status='running', stage='crawl',
                                    settings=settings, output=settings['prepare']['output'], log_path='dataset.log', log='Whole archive import',
                                    pipeline=dict(stage='crawl', stage_index=1, outputs=dict(corpus=settings['crawl']['output'], dataset=settings['prepare']['output']),
                                        stages=[dict(kind='crawl', status='running'), dict(kind='prepare', status='pending')]),
                                    progress=dict(phase='importing_zim', mode='kiwix', wikipedia_all=True, heartbeat_at=1, detail='Whole archive import',
                                        completed=4, completed_units=10, budget=19, saved=4, skipped=0, failed=0, queued=9, archive_cursor=10, archive={'entries': 19}))
                            else:
                                raise AssertionError(path)
                            await route.fulfill(json=state)
                            return
                        response = client.request(request.method, path + ('?' + parsed.query if parsed.query else ''),
                                                  json=json.loads(request.post_data) if request.post_data else None)
                        await route.fulfill(status=response.status_code, body=response.content, content_type=response.headers.get('content-type'))
                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#hosts input[value="soyjakwiki.org"]').uncheck()
                    await page.locator('#hosts input[value="wikipedia.org"]').check()
                    self.assertIn('Offline fixture', await page.locator('#wikipedia-info').inner_text())
                    await page.locator('#wikipedia-settings summary').click()
                    await page.locator('#kiwix-catalog-button').click()
                    await page.wait_for_function("document.getElementById('kiwix-catalog-info').textContent.includes('Select an archive')")
                    await page.locator('#download-kiwix').click()
                    await page.wait_for_function("document.getElementById('job-metrics').textContent.includes('1.00 / 2.00 GiB')")
                    self.assertFalse(await page.locator('#generate').is_disabled())
                    self.assertTrue(await page.locator('#start-pipeline').is_disabled())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("!document.getElementById('download-kiwix').disabled")
                    await page.locator('#download-kiwix').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('KIWIX · running')")
                    downloads = [payload for path, payload in calls if path == '/jobs/kiwix']
                    self.assertEqual(downloads, [dict(url=BASE_URL + DEFAULT_FILE, output=DEFAULT_PATH)] * 2)
                    state['job']['status'] = 'succeeded'
                    state['job']['progress'].update(phase='completed', downloaded_bytes=2*1024**3)
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("!document.getElementById('download-kiwix').disabled")
                    await page.locator('#wikipedia-seed').fill('9')
                    await page.locator('#crawl-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('job-metrics').textContent.includes('Offline Wikipedia')")
                    payload = [payload for path, payload in calls if path == '/jobs/crawl'][-1]
                    self.assertEqual(payload['hosts'], ['wikipedia.org'])
                    self.assertEqual(payload['wikipedia_zim'], DEFAULT_PATH)
                    self.assertEqual(payload['wikipedia_seed'], 9)
                    self.assertTrue(payload['wikipedia_all'])
                    self.assertTrue(await page.locator('#browser-snapshot').is_hidden())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("!document.getElementById('prepare-zim-dataset').disabled")
                    await page.locator('#wikipedia-all').uncheck()
                    await page.locator('#crawl-pages').fill('1')
                    await page.locator('#prepare-zim-dataset').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stage 1/2: crawl') && document.getElementById('wikipedia-all').checked")
                    payload = [payload for path, payload in calls if path == '/jobs/zim-dataset'][-1]
                    self.assertEqual(set(payload), {'crawl', 'prepare', 'auto_paths'})
                    self.assertEqual(payload['crawl']['hosts'], ['wikipedia.org'])
                    self.assertTrue(payload['crawl']['wikipedia_all'])
                    self.assertTrue(await page.locator('#wikipedia-all').is_checked())
                    self.assertEqual(await page.locator('#prepare-output').input_value(), 'data/prepared-zim-whole-fixture')
                    self.assertIn('Whole ZIM', await page.locator('#job-metrics').inner_text())
                    self.assertEqual(await page.locator('#job-progress').get_attribute('max'), '19')
                    self.assertEqual(await page.locator('#job-progress').get_attribute('value'), '10')
                    self.assertNotIn('undefined', await page.locator('#pipeline-detail').inner_text())
                    self.assertFalse(await page.locator('#generate').is_disabled())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("!document.getElementById('resume-dataset').hidden && !document.getElementById('resume-dataset').disabled")
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#resume-dataset').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('DATASET · running')")
                    payload = [payload for path, payload in calls if path == '/jobs/zim-dataset'][-1]
                    self.assertFalse(payload['auto_paths'])
                    self.assertTrue(payload['crawl']['resume'])
                    self.assertTrue(payload['crawl']['wikipedia_all'])
                    self.assertEqual(payload['prepare']['output'], 'data/prepared-zim-whole-fixture')
                    output = root/payload['prepare']['output']
                    output.mkdir(parents=True)
                    (output/'tokenizer.json').write_text('{}')
                    (output/'meta.json').write_text(json.dumps(dict(documents=dict(train=7, val=1), tokens=dict(train=10000, val=1000), vocab_size=300)))
                    for split, count in [('train', 10000), ('val', 1000)]:
                        (output/f'{split}.bin').write_bytes(b'\0' * (4 * count))
                    state['job'].update(status='succeeded', stage='prepare', progress=None)
                    state['job']['pipeline'].update(stage='prepare', stage_index=2)
                    for stage in state['job']['pipeline']['stages']:
                        stage['status'] = 'succeeded'
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('train-data').value === 'data/prepared-zim-whole-fixture'")
                    self.assertTrue(await page.locator('#select-job-checkpoint').is_hidden())
                    self.assertFalse(errors, errors)
                spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
