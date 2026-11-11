"""No training. Exercises real browser completion/stop and detached process cleanup."""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import psutil

from scratchllm.crawl import EXTRACTION_VERSION, extract
from scratchllm.jobs import CrawlRequest, Jobs


class CrawlLifecycleTests(unittest.TestCase):
    def wait_for(self, jobs, predicate, timeout=35):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = jobs.status()["job"]
            if predicate(status):
                return status
            time.sleep(0.05)
        self.fail("Job timed out: " + json.dumps(jobs.status()))

    def browser_job(self, jobs, mode, request):
        real_popen = subprocess.Popen
        fixture = Path(__file__).with_name('crawl_fixture_worker.py')
        def spawn(command, **kwargs):
            return real_popen([sys.executable, '-u', str(fixture), mode, *command[4:]], **kwargs)
        with patch('scratchllm.jobs.subprocess.Popen', side_effect=spawn):
            return jobs.start('crawl', request)

    def assert_closed(self, jobs):
        self.assertFalse(jobs.active())
        self.assertEqual(jobs.job['remaining_pids'], [])
        self.assertFalse(any(jobs._live_process(row) for row in jobs.job['processes']))

    def test_duplicates_and_empty_pages_report_and_browser_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "An actual article paragraph with enough written text for extraction. " * 8
            # Match the real extracted text, including whitespace normalization.
            text = extract(f'<div id="mw-content-text"><p>{text}</p></div>')[1]
            (root/'corpus.jsonl').write_text(json.dumps(dict(url='https://soyjakwiki.org/',
                text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), extraction=EXTRACTION_VERSION))+'\n')
            jobs = Jobs(root, 'cpu')
            try:
                self.browser_job(jobs, 'complete', CrawlRequest(output='corpus.jsonl', pages_per_site=4, depth=1, delay=0.5, headed=False))
                status = self.wait_for(jobs, lambda j: j['status'] not in Jobs.LIVE)
                self.assertEqual(status['status'], 'succeeded', status['log'])
                progress = status['progress']
                self.assertEqual(progress['completed'], 4)
                self.assertEqual(progress['saved'], 1)
                self.assertEqual(progress['skipped'], 3)
                for message in ['FETCH [4/4]', 'URL already saved', 'no article/post text', 'duplicate text', 'Browser closed']:
                    self.assertIn(message, status['log'])
                self.assert_closed(jobs)
                restored = Jobs(root, 'cpu').status()['job']
                self.assertEqual(restored['id'], status['id'])
                self.assertEqual(restored['status'], 'succeeded')
            finally:
                jobs.close()

    def test_stop_during_verification_closes_real_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                self.browser_job(jobs, 'wait', CrawlRequest(output='corpus.jsonl', pages_per_site=2, headed=False))
                self.wait_for(jobs, lambda j: (j.get('progress') or {}).get('phase') == 'waiting_verification')
                # Check that long waits produce an actual heartbeat, not a frozen log.
                self.wait_for(jobs, lambda j: 'ACTIVE waiting_verification' in j['log'], timeout=15)
                jobs.stop()
                status = self.wait_for(jobs, lambda j: j['status'] == 'stopped')
                self.assertEqual(status['returncode'], 0, status['log'])
                self.assertFalse(status['forced_stop'])
                self.assertIn('STOPPED', status['log'])
                self.assert_closed(jobs)
            finally:
                jobs.close()

    def test_real_browser_adopts_verified_tab_and_saves_content(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                self.browser_job(jobs, 'other-tab', CrawlRequest(output='corpus.jsonl', pages_per_site=1, headed=False))
                status = self.wait_for(jobs, lambda j: j['status'] not in Jobs.LIVE)
                self.assertEqual(status['status'], 'succeeded', status['log'])
                self.assertEqual(status['progress']['saved'], 1)
                row = json.loads((Path(directory) / 'corpus.jsonl').read_text())
                self.assertEqual(row['url'], 'https://soyjakwiki.org/verified')
                self.assertNotIn('Manual verification', row['text'])
                self.assertIn('READY wiki document detected', status['log'])
                self.assertTrue(status['browser_snapshot'])
                self.assert_closed(jobs)
            finally:
                jobs.close()

    def test_manual_window_close_stops_crawl(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            try:
                self.browser_job(jobs, 'close-window', CrawlRequest(output='corpus.jsonl', pages_per_site=2, headed=False))
                status = self.wait_for(jobs, lambda j: j['status'] not in Jobs.LIVE)
                self.assertEqual(status['status'], 'stopped', status['log'])
                self.assertIn('closed by user', status['log'])
                self.assert_closed(jobs)
            finally:
                jobs.close()

    def test_worker_crash_cleans_child_in_separate_session(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(directory, 'cpu')
            real_popen = subprocess.Popen
            child_code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)'
            worker = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child_code!r}],start_new_session=True); print("child started",flush=True); time.sleep(120)'
            def spawn(command, **kwargs):
                return real_popen([sys.executable, '-u', '-c', worker], **kwargs)
            try:
                with patch('scratchllm.jobs.subprocess.Popen', side_effect=spawn):
                    jobs.start('crawl', CrawlRequest(output='corpus.jsonl', headed=False))
                self.wait_for(jobs, lambda j: len(jobs.job['processes']) >= 2)
                jobs.process.kill()
                status = self.wait_for(jobs, lambda j: j['status'] not in Jobs.LIVE)
                self.assertEqual(status['status'], 'failed')
                self.assert_closed(jobs)
            finally:
                jobs.close()

    def test_restart_recovers_interrupted_job_and_closes_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / 'runs/jobs'
            logs.mkdir(parents=True)
            (logs/'old.log').write_text('Previous server log\n')
            code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"],start_new_session=True); time.sleep(120)'
            process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
            parent = psutil.Process(process.pid)
            try:
                deadline = time.monotonic() + 5
                while not parent.children() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(parent.children())
                state = dict(id='old', kind='crawl', status='running', output='corpus.jsonl',
                             stop_requested=False, log_path='runs/jobs/old.log',
                             processes=[Jobs._identity(parent)])
                (logs/'latest.json').write_text(json.dumps(state))
                jobs = Jobs(root, 'cpu')
                status = jobs.status()['job']
                self.assertEqual(status['status'], 'interrupted')
                self.assertIn('RECOVER', status['log'])
                self.assertFalse(any(jobs._live_process(row) for row in jobs.job['processes']))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
