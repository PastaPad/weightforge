"""Offline browser checks for inspection, text preparation and resume; no model training."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from unittest.mock import patch

from camoufox.async_api import AsyncCamoufox
from fastapi.testclient import TestClient

from scratchllm.corpus import corpus_plan
from scratchllm.jobs import CrawlRequest, Jobs
from scratchllm.serve import create_app
from test_data_quality import write_corpus
from test_serving_selection import FakeGenerator


class DataQualityUITests(unittest.TestCase):
    def test_short_text_upload_and_preparation_have_independent_length_controls(self):
        asyncio.run(self.exercise_text_upload())

    async def exercise_text_upload(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)/'workspace'
            (root/'data').mkdir(parents=True)
            text = Path(directory)/'short.txt'
            raw = 'Räksmörgås i solen.\r\nEn andra mening.\r\nEn tredje text.\r\n'.encode('utf-8')
            text.write_bytes(raw)
            write_corpus(root/'data/short.jsonl', [dict(text=line) for line in raw.decode().splitlines()])
            calls, errors = [], []
            start = Jobs.start

            def prepare_only(self, kind, request):
                if kind != 'prepare':
                    raise AssertionError('Only fixture dataset preparation is permitted')
                return start(self, kind, request)

            with patch.object(Jobs, 'start', prepare_only), TestClient(create_app(root/'missing.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.set_default_timeout(20000)
                    page.on('pageerror', lambda error: errors.append(str(error)))

                    async def route(route):
                        incoming = route.request
                        url = urlsplit(incoming.url)
                        if url.path == '/corpus/upload':
                            calls.append((url.path, None))
                            response = client.post(url.path + '?' + url.query, content=incoming.post_data_buffer,
                                                   headers={'Content-Type': 'application/octet-stream'})
                        else:
                            payload = json.loads(incoming.post_data) if incoming.post_data else None
                            if incoming.method == 'POST':
                                calls.append((url.path, payload))
                            response = client.request(incoming.method, url.path, json=payload)
                        await route.fulfill(status=response.status_code, body=response.content,
                                            content_type=response.headers.get('content-type', 'application/json'))

                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.locator('#min-article-chars').fill('999')
                    await page.locator('#min-post-chars').fill('80')
                    await page.locator('#prepare-input').fill('data/short.jsonl')
                    await page.locator('#inspect-corpus').click()
                    await page.wait_for_function("document.getElementById('corpus-report').textContent.includes('All text was removed')")
                    self.assertEqual(calls[-1][1]['min_chars'], 300)
                    await page.locator('#prepare-short-text').click()
                    await page.locator('#inspect-corpus').click()
                    await page.wait_for_function("document.getElementById('corpus-report').textContent.includes('Ready for preparation.')")
                    self.assertIn('3 unique text records', await page.locator('#corpus-report').inner_text())
                    self.assertEqual(await page.locator('#min-article-chars').input_value(), '999')
                    self.assertEqual(await page.locator('#min-post-chars').input_value(), '80')
                    self.assertIsNone(client.get('/jobs').json()['job'])

                    await page.locator('#prepare-min-chars').fill('300')
                    await page.locator('#prepare-file').set_input_files(text)
                    self.assertEqual(await page.locator('#prepare-min-chars').input_value(), '1')
                    self.assertEqual(await page.locator('#prepare-input').input_value(), '')
                    await page.locator('#inspect-corpus').click()
                    await page.wait_for_function("document.getElementById('corpus-report').textContent.includes('Ready for preparation.')")
                    uploaded = await page.locator('#prepare-input').input_value()
                    self.assertEqual((root/uploaded).read_bytes(), raw)
                    self.assertIsNone(client.get('/jobs').json()['job'])

                    # Submitting a freshly selected file also uploads it before starting preparation.
                    await page.locator('#prepare-file').set_input_files(text)
                    await page.locator('#prepare-output').fill('data/ready')
                    await page.locator('#vocab-size').fill('259')
                    await page.locator('#prepare-form button[type=submit]').click()
                    await page.wait_for_function("document.getElementById('job-summary').textContent.includes('PREPARE · succeeded')")
                    submitted = next(body for path, body in calls if path == '/jobs/prepare')
                    self.assertEqual(submitted['min_chars'], 1)
                    self.assertEqual(submitted['min_post_chars'], 1)
                    self.assertEqual((root/submitted['input']).read_bytes(), raw)
                    self.assertEqual(text.read_bytes(), raw)
                    self.assertEqual(len([path for path, _ in calls if path == '/corpus/upload']), 2)
                    self.assertEqual([path for path, _ in calls if path.startswith('/jobs/')], ['/jobs/prepare'])
                    report = json.loads((root/'data/ready/report.json').read_text())
                    self.assertEqual(report['accepted_records'], 3)
                    self.assertEqual(report['text_unit'], 'lines')
                    self.assertFalse(errors, errors)

    def test_inspection_samples_stale_results_reports_and_resume_settings(self):
        asyncio.run(self.exercise_browser())

    async def exercise_browser(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', side_effect=AssertionError('Unexpected worker launch')) as start:
            root = Path(directory)
            (root/'fake.pt').write_text('fake')
            (root/'data').mkdir()
            corpus = root/'data/corpus.jsonl'
            write_corpus(corpus)
            _, report = corpus_plan(corpus)
            dataset = root/'data/prepared'
            dataset.mkdir()
            (dataset/'tokenizer.json').write_text('{}')
            for source in report['sources'].values():
                source['tokens'] = {split: count * 800 for split, count in source['documents'].items()}
            tokens = {split: sum(source['tokens'][split] for source in report['sources'].values()) for split in ('train', 'val')}
            for split, count in tokens.items():
                with (dataset/f'{split}.bin').open('wb') as stream:
                    stream.truncate(4 * count)
            (dataset/'meta.json').write_text(json.dumps(dict(vocab_size=259, tokens=tokens, documents=report['documents'], corpus_report=report)))
            settings = CrawlRequest(hosts=['swedishwin.com', 'soyjakwiki.org'], output='data/corpus.jsonl',
                                    pages_per_site=100, depth=2, delay=5, headed=False, min_post_chars=60).model_dump()
            state = {'job': dict(id='saved-crawl', kind='crawl', status='stopped', output=settings['output'],
                                 log_path='fixture.log', log='Stopped', settings=settings,
                                 resumable_crawl=dict(output=settings['output'], pending=3, settings=settings))}
            submitted, errors = [], []
            entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
            delay_inspection = False
            with TestClient(create_app(root/'fake.pt', 'cpu', workspace=root)) as client:
                async with AsyncCamoufox(headless=True) as browser:
                    page = await browser.new_page()
                    page.on('pageerror', lambda error: errors.append(str(error)))

                    async def route(request):
                        nonlocal delay_inspection
                        incoming = request.request
                        path = urlsplit(incoming.url).path
                        payload = json.loads(incoming.post_data) if incoming.post_data else None
                        if path == '/jobs':
                            await request.fulfill(json=state)
                            return
                        if path.startswith('/jobs/'):
                            self.assertEqual(path, '/jobs/crawl')
                            self.assertTrue(payload['resume'])
                            submitted.append(payload)
                            restored = dict(settings, output=payload['output'], pages_per_site=max(settings['pages_per_site'], payload['pages_per_site']), resume=True)
                            state['job'] = dict(state['job'], id=f'resume-{len(submitted)}', status='running', settings=restored)
                            await request.fulfill(json=state)
                            return
                        response = client.request(incoming.method, path, json=payload)
                        delayed = delay_inspection and path == '/corpus/inspect'
                        if delayed:
                            delay_inspection = False
                            entered.set()
                            await asyncio.wait_for(release.wait(), 10)
                        await request.fulfill(status=response.status_code, body=response.content,
                                              content_type=response.headers.get('content-type', 'application/json'))
                        if delayed:
                            finished.set()

                    await page.route('http://scratch.test/**', route)
                    await page.goto('http://scratch.test/', wait_until='networkidle')
                    await page.locator('#train-tab').click()
                    await page.wait_for_function("document.getElementById('training-plan').textContent.includes('swedishwin.com: 3 records')")
                    self.assertIn('soyjakwiki.org', await page.locator('#dataset-info').inner_text())
                    await page.locator('#prepare-input').fill('data/corpus.jsonl')
                    await page.locator('#inspect-corpus').click()
                    await page.wait_for_function("document.getElementById('corpus-report').textContent.includes('Ready for preparation.')")
                    summary = await page.locator('#corpus-report').inner_text()
                    self.assertIn('4 unique text records', summary)
                    self.assertIn('1 duplicates removed', summary)
                    self.assertIn('1 too short', summary)
                    await page.locator('#corpus-samples summary').click()
                    self.assertIn('local history', await page.locator('#corpus-sample-list').inner_text())

                    # A completed inspection for a previous input must not overwrite an edit.
                    delay_inspection = True
                    await page.locator('#inspect-corpus').click()
                    await asyncio.wait_for(entered.wait(), 5)
                    await page.locator('#prepare-input').fill('data/different.jsonl')
                    release.set()
                    await asyncio.wait_for(finished.wait(), 5)
                    await page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
                    self.assertEqual(await page.locator('#corpus-report').inner_text(), '')
                    self.assertTrue(await page.locator('#corpus-samples').is_hidden())
                    self.assertFalse(submitted)

                    # The monitor shortcut selects the saved corpus despite a stale output field.
                    await page.locator('#corpus-output').fill('data/wrong.jsonl')
                    await page.locator('#crawl-pages').fill('150')
                    await page.locator('#resume-saved-crawl').click()
                    await page.wait_for_function("document.getElementById('corpus-output').value === 'data/corpus.jsonl'")
                    self.assertEqual(submitted[-1], dict(output='data/corpus.jsonl', resume=True, pages_per_site=150, wikipedia_all=True))
                    self.assertEqual(await page.locator('#prepare-input').input_value(), 'data/corpus.jsonl')
                    self.assertEqual(await page.locator('#crawl-depth').input_value(), '2')
                    self.assertEqual(await page.locator('#min-post-chars').input_value(), '60')
                    self.assertFalse(await page.locator('#headed-browser').is_checked())
                    self.assertTrue(await page.locator('#inspect-corpus').is_disabled())
                    self.assertEqual(set(await page.locator('#hosts input:checked').evaluate_all('(inputs) => inputs.map(input => input.value)')), {'swedishwin.com', 'soyjakwiki.org'})

                    # The form resume button can target a different saved corpus by path.
                    state['job'] = dict(state['job'], status='stopped')
                    await page.wait_for_function("!document.getElementById('resume-crawl').disabled")
                    await page.locator('#corpus-output').fill('data/other-saved.jsonl')
                    await page.locator('#resume-crawl').click()
                    await page.wait_for_function("document.getElementById('prepare-input').value === 'data/other-saved.jsonl'")
                    self.assertEqual(submitted[-1]['output'], 'data/other-saved.jsonl')
                    self.assertEqual(len(submitted), 2)
                    self.assertFalse(errors, errors)
            start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
