"""One-click UI checks against a fake pipeline API; no workers, training, or external sites."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.jobs import Jobs, PipelineRequest
from scratchllm.serve import create_app
from test_serving_selection import FakeGenerator
from kiwix_fixture import make_archive
from scratchllm.kiwix import DEFAULT_PATH


class AutomatedPipelineUITests(unittest.TestCase):
    def test_one_click_settings_monitor_reopen_stop_and_checkpoint_selection(self):
        asyncio.run(self.exercise_browser())

    async def exercise_browser(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', side_effect=AssertionError('Unexpected worker launch')) as spawn:
            root = Path(directory)
            make_archive(root/DEFAULT_PATH)
            (root/'old.pt').write_text('fake model')
            resolver = Jobs(root, 'cpu')
            state, submitted, errors, started = {'job': None}, [], [], {}
            with TestClient(create_app(root/'old.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(10000)
                    page.on('pageerror', lambda error: errors.append(str(error)))

                    async def route(request):
                        incoming = request.request
                        path = urlsplit(incoming.url).path
                        if path == '/jobs':
                            await request.fulfill(json=state)
                            return
                        if path.startswith('/jobs/'):
                            payload = json.loads(incoming.post_data or '{}')
                            submitted.append((path, payload))
                            if path == '/jobs/pipeline':
                                settings = resolver._pipeline_request(PipelineRequest.model_validate(payload), 'ui-fixture').model_dump()
                                state['job'] = dict(id='ui-fixture', kind='pipeline', status='running', stage='crawl',
                                    settings=settings, output=settings['train']['out'], log_path='fixture.log', log='Pipeline started',
                                    pipeline=dict(phase='running', stage='crawl', stage_index=1, detail='Collecting selected sources',
                                        outputs=dict(corpus=settings['crawl']['output'], dataset=settings['prepare']['output'], run=settings['train']['out']),
                                        stages=[dict(kind=kind, status='running' if kind == 'crawl' else 'pending') for kind in ['crawl', 'prepare', 'train']]),
                                    progress=dict(phase='fetching', heartbeat_at=1, completed=1, budget=10, saved=1, skipped=0, failed=0, queued=2, detail='Fetching a post'))
                                # The real start response can precede the coordinator's first snapshot.
                                started['job'] = copy.deepcopy(state['job'])
                                state['job'].update(pipeline={}, stage=None, progress=None)
                            elif path == '/jobs/stop':
                                state['job']['status'] = 'stopped'
                                state['job']['pipeline'].update(phase='stopped', detail='Remaining stages cancelled')
                                for stage in state['job']['pipeline']['stages']:
                                    if stage['status'] == 'running':
                                        stage['status'] = 'stopped'
                                    elif stage['status'] == 'pending':
                                        stage['status'] = 'cancelled'
                            else:
                                raise AssertionError(f'Unexpected job request {path}')
                            await request.fulfill(json=state)
                            return
                        response = client.request(incoming.method, path, json=json.loads(incoming.post_data) if incoming.post_data else None)
                        await request.fulfill(status=response.status_code, body=response.content,
                                              content_type=response.headers.get('content-type', 'application/json'))

                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#hosts input[value="soyjakwiki.org"]').uncheck()
                    await page.locator('#hosts input[value="soyjak.st"]').check()
                    await page.locator('#hosts input[value="swedishwin.com"]').check()
                    await page.locator('#hosts input[value="wikipedia.org"]').check()
                    self.assertTrue(await page.locator('#wikipedia-settings').is_visible())
                    await page.locator('#wikipedia-seed').fill('123')
                    await page.locator('#pipeline-dialogue').uncheck()
                    await page.locator('#crawl-pages').fill('5')
                    await page.locator('#vocab-size').fill('4096')
                    await page.locator('#train-budget-mode').select_option('passes')
                    await page.locator('#train-epochs').fill('50')
                    # Auto-path mode works before a dataset exists, even with empty obsolete paths.
                    for field in ['corpus-output', 'prepare-input', 'prepare-output', 'train-data', 'train-out']:
                        await page.locator('#' + field).fill('')
                    await page.locator('#start-pipeline').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('PIPELINE · running') && !document.getElementById('stop-job').disabled")
                    self.assertEqual(await page.locator('#error').inner_text(), '')
                    self.assertIn('Waiting for pipeline stage information', await page.locator('#pipeline-detail').inner_text())
                    self.assertIsNone(await page.locator('#job-progress').get_attribute('value'))
                    self.assertEqual(await page.locator('#train-out').input_value(), state['job']['settings']['train']['out'])
                    self.assertEqual(len(submitted), 1)
                    payload = submitted[0][1]
                    self.assertEqual(payload['crawl']['hosts'], ['soyjak.st', 'swedishwin.com', 'wikipedia.org'])
                    self.assertEqual(payload['crawl']['pages_per_site'], 5)
                    self.assertEqual(payload['crawl']['wikipedia_zim'], DEFAULT_PATH)
                    self.assertEqual(payload['crawl']['wikipedia_seed'], 123)
                    self.assertTrue(payload['crawl']['wikipedia_all'])
                    self.assertEqual(payload['prepare']['vocab_size'], 4096)
                    self.assertEqual(payload['train']['max_epochs'], 50)
                    self.assertEqual(payload['train']['context'], 512)
                    self.assertEqual(payload['train']['batch_size'], 2)
                    self.assertEqual(payload['train']['sampling'], 'shuffled')
                    self.assertTrue(payload['auto_paths'])
                    self.assertTrue(await page.locator('#start-pipeline').is_disabled())
                    self.assertTrue(await page.locator('#generate').is_disabled())
                    self.assertEqual(await page.locator('#stop-job').inner_text(), 'Stop entire pipeline')

                    async def render_snapshot(label):
                        state['job']['log'] = label
                        await page.locator('#train-tab').click()
                        await page.wait_for_function('(message) => document.getElementById("job-log").textContent === message', arg=label)
                        self.assertNotIn('Status unavailable', await page.locator('#job-summary').inner_text())

                    # Missing/null/empty stages, including a partial stage label, must render safely.
                    for index, pipeline in enumerate((None, {'stage': 'crawl', 'stage_index': 1},
                                     {'stage': 'crawl', 'stage_index': 1, 'stages': None}, {'stages': []})):
                        state['job']['pipeline'] = pipeline
                        await render_snapshot(f'Incomplete snapshot {index}')
                        self.assertNotIn('undefined', await page.locator('#job-summary').inner_text())
                        self.assertIsNone(await page.locator('#job-progress').get_attribute('value'))
                        self.assertFalse(await page.locator('#stop-job').is_disabled())
                    state['job'].pop('pipeline')
                    await render_snapshot('No pipeline snapshot yet')
                    self.assertIn('Waiting for pipeline stage information', await page.locator('#pipeline-detail').inner_text())
                    state['job']['status'] = 'failed'
                    await render_snapshot('Failed before first snapshot')
                    self.assertIn('PIPELINE · failed', await page.locator('#job-summary').inner_text())
                    self.assertEqual(await page.locator('#job-progress').get_attribute('value'), '0')
                    self.assertTrue(await page.locator('#stop-job').is_disabled())

                    state['job'] = started['job']
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stage 1/3: crawl')")
                    self.assertTrue(await page.locator('#crawl-detail').is_visible())

                    job = state['job']
                    job.update(stage='prepare', progress=None)
                    job['pipeline'].update(stage='prepare', stage_index=2, detail='Preparing the collected corpus')
                    job['pipeline']['stages'][0]['status'] = 'succeeded'
                    job['pipeline']['stages'][1]['status'] = 'running'
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('stage 2/3: prepare')")
                    self.assertIn('stage 2/3: prepare', await page.locator('#job-summary').inner_text())
                    self.assertTrue(await page.locator('#crawl-detail').is_hidden())

                    # Reload during the pipeline: server state, not page callbacks, drives the monitor.
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    self.assertIn('prepare: running', await page.locator('#pipeline-detail').inner_text())
                    self.assertTrue(await page.locator('#start-pipeline').is_disabled())
                    await page.locator('#stop-job').click()
                    await page.wait_for_function("document.getElementById('pipeline-detail').textContent.includes('train: cancelled')")
                    await page.wait_for_function("!document.getElementById('start-pipeline').disabled")
                    self.assertEqual([path for path, _ in submitted], ['/jobs/pipeline', '/jobs/stop'])
                    self.assertFalse(await page.locator('#start-pipeline').is_disabled())

                    # On returning to a completed run, its best checkpoint is explicitly selectable.
                    completed = copy.deepcopy(job)
                    completed.update(id='finished-fixture', status='succeeded', stage='train', checkpoint=job['output'] + '/best.pt', metrics=[dict(
                        step=3, train_loss=1.5, train_eval_loss=1.5, val_loss=2.0, perplexity=7.39, generalization_gap=0.5,
                        epochs=0.1, best_val_loss=1.9, best_step=2, selected_weights='raw', planned_steps=100,
                        stop_reason='early_stopping')])
                    completed['pipeline'].update(phase='completed', stage='train', stage_index=3, detail='Best checkpoint saved')
                    for stage in completed['pipeline']['stages']:
                        stage['status'] = 'succeeded'
                    state['job'] = completed
                    await page.reload(wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('PIPELINE · succeeded')")
                    self.assertIn('Finished: early stopping', await page.locator('#job-metrics').inner_text())
                    self.assertEqual(await page.locator('#load-checkpoint').input_value(), str(root/'old.pt'))
                    await page.locator('#select-job-checkpoint').click()
                    self.assertEqual(await page.locator('#load-checkpoint').input_value(), completed['checkpoint'])
                    self.assertEqual(await page.locator('#loaded-checkpoint').inner_text(), str(root/'old.pt'))
                    self.assertFalse(errors, errors)
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
