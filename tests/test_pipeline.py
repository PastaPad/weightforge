"""Offline checks for causality, learning, extraction, and the complete local pipeline."""
import json
import asyncio
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from scratchllm.crawl import canonical, extract, robots, fetch_page, is_challenge, wait_for_wiki
from scratchllm.model import GPT, ModelConfig
from scratchllm.serve import create_app


class PipelineTests(unittest.TestCase):
    def test_ignore_robots_does_not_fetch_policy(self):
        with patch("scratchllm.crawl.urlopen") as fetch:
            policy = robots("https://swedishwin.com", ignore=True)
            self.assertTrue(policy.can_fetch("ScratchLLMCorpusBot", "https://swedishwin.com/"))
            fetch.assert_not_called()

    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_causal_mask_and_learning(self):
        model = GPT(ModelConfig(32, context=16, width=32, layers=2, heads=4, dropout=0)).eval()
        x = torch.randint(0, 32, (2, 16))
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 32
        original_logits, _ = model(x, x)
        changed_logits, _ = model(changed, x)
        torch.testing.assert_close(original_logits[:, :8], changed_logits[:, :8])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.tensor([[1, 2, 3, 4] * 4])
        targets = torch.tensor([[2, 3, 4, 1] * 4])
        initial = model(inputs, targets)[1].item()
        model.train()
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(inputs, targets)[1]
            loss.backward()
            optimizer.step()
        model.eval()
        self.assertLess(model(inputs, targets)[1].item(), initial / 3)
        generated = model.generate(inputs[:, :4], max_new_tokens=4, temperature=0)
        self.assertEqual(generated[0, 4:].tolist(), [1, 2, 3, 4])

    def test_extraction_and_scope(self):
        title, text, links = extract('<title>Board</title><body><nav>menu</nav><form><div class="post"><div class="postInfo">Anonymous 2026 Reply</div><div class="body">A real post</div></div><input value="delete"></form><script>bad</script><a href="/thread/1">Thread</a></body>')
        self.assertIn("A real post", text)
        self.assertNotIn("menu", text)
        self.assertNotIn("bad", text)
        self.assertEqual(text, "A real post")
        self.assertEqual(title, "Board")
        self.assertEqual(links, ["/thread/1"])
        self.assertIsNone(canonical("https://evil.example/a", "soyjak.st"))
        self.assertIsNone(canonical("https://soyjak.st/a.jpg", "soyjak.st"))
        self.assertEqual(canonical("https://soyjakwiki.org/index.php?title=Test#x", "soyjakwiki.org"), "https://soyjakwiki.org/index.php?title=Test")

    def test_wiki_content_and_fallback(self):
        html = '<title>Article</title><nav>Site links</nav><div id="mw-content-text"><div class="mw-parser-output"><div class="infobox">Metadata</div><p>Actual article text.</p><span class="mw-editsection">edit</span><div class="navbox">Other pages</div></div><div class="printfooter">Footer</div></div>'
        self.assertEqual(extract(html)[1], "Actual article text.")
        self.assertEqual(extract('<body>Site directory <a href="/a">article</a></body>')[1], "")
        challenge = '<title>Just a moment...</title><form id="challenge-form">Wait</form>'
        self.assertTrue(is_challenge(challenge))
        page = AsyncMock()
        page.on = MagicMock()
        page.remove_listener = MagicMock()
        page.is_closed = MagicMock(return_value=False)
        page.goto.side_effect = RuntimeError("Cloudflare blocked browser")
        with patch("scratchllm.crawl.curl_fetch", return_value=(200, "https://soyjakwiki.org/Test", html)) as fetch:
            final, content, transport = asyncio.run(fetch_page(page, "https://soyjakwiki.org/Test", "soyjakwiki.org", 0))
            self.assertEqual(transport, "curl_cffi")
            self.assertEqual(extract(content)[1], "Actual article text.")
            fetch.assert_called_once()
        with patch("scratchllm.crawl.curl_fetch", return_value=(200, "https://soyjakwiki.org/Test", challenge)):
            with self.assertRaises(RuntimeError):
                asyncio.run(fetch_page(page, "https://soyjakwiki.org/Test", "soyjakwiki.org", 0))

    def test_cloudflare_navigation_after_manual_verification(self):
        page = AsyncMock()
        page.url = "https://soyjakwiki.org/Test"
        page.is_closed = MagicMock(return_value=False)
        page.remove_listener = MagicMock()
        response = MagicMock(status=403, headers={'content-type': 'text/html'})
        response.request.is_navigation_request.return_value = True
        response.request.frame = page.main_frame
        page.goto.return_value = response
        html = '<div id="mw-content-text"><p>Actual article after verification.</p></div>'
        page.content.side_effect = ['<title>Just a moment...</title>', html]
        page.context.pages = [page]
        page.on = MagicMock()
        # No second navigation event: the original response remains HTTP 403.
        with patch("scratchllm.crawl.curl_fetch") as fallback:
            final, content, transport = asyncio.run(fetch_page(page, page.url, "soyjakwiki.org", 0, 60))
            self.assertEqual(transport, "camoufox")
            self.assertEqual(content, html)
            fallback.assert_not_called()
            page.remove_listener.assert_called_once()

    def test_verification_in_another_wiki_tab(self):
        blocked, verified = AsyncMock(), AsyncMock()
        for page in (blocked, verified):
            page.is_closed = MagicMock(return_value=False)
            page.on = MagicMock()
            page.remove_listener = MagicMock()
        blocked.url = 'https://soyjakwiki.org/'
        blocked.goto.return_value = MagicMock(status=403, headers={'content-type': 'text/html'})
        blocked.content.return_value = '<title>Just a moment...</title><form id="challenge-form"></form>'
        verified.url = 'https://soyjakwiki.org/Article'
        verified.content.return_value = '<title>Article</title><div class="mw-parser-output"><p>Actual article prose.</p></div>'
        blocked.context.pages = [blocked, verified]
        select = MagicMock()
        with patch('scratchllm.crawl.curl_fetch') as fallback:
            url, html, transport = asyncio.run(fetch_page(blocked, blocked.url, 'soyjakwiki.org', 0, 1, select_page=select))
            self.assertEqual(url, verified.url)
            self.assertEqual(transport, 'camoufox')
            self.assertEqual(extract(html)[1], 'Actual article prose.')
            select.assert_called_once_with(verified)
            fallback.assert_not_called()

    def test_readiness_rejects_unrelated_tab(self):
        page = AsyncMock()
        page.url = 'https://example.test/Article'
        page.is_closed = MagicMock(return_value=False)
        page.context.pages = [page]
        page.content.return_value = '<div id="mw-content-text">Not the requested wiki</div>'
        with self.assertRaises(RuntimeError):
            asyncio.run(wait_for_wiki(page, 0))
        page.content.assert_not_called()

    def test_strict_article_and_post_text(self):
        html = '''<body class="mediawiki ns-0 page-Article"><nav>Login</nav>
          <div id="mw-content-text"><div class="mw-parser-output">
          <div class="template-soybox noexcerpt"><p>Update this page</p></div>
          <table class="sidebar"><tr><td>Series navigation</td></tr></table>
          <div>Unclassified widget text</div>
          <p>An <a href="/Article">actual</a> sentence.</p>
          <h2>History<span class="mw-editsection">edit</span></h2>
          <ul><li>A written fact.</li><li>Another fact.</li></ul>
          <h2>See also</h2><ul><li>Related article link</li></ul>
          </div></div></body>'''
        text = extract(html)[1]
        self.assertEqual(text, 'An actual sentence.\n\nHistory\n\nA written fact.\nAnother fact.')
        self.assertEqual(extract(html.replace('ns-0 page-Article', 'ns--1 page-Special_Statistics'))[1], '')
        self.assertEqual(extract(html.replace('page-Article', 'page-Main_Page'))[1], '')
        post = '''<main><label>Captcha</label><div class="post-container"><div class="post-info">Anonymous [Quote]</div>
          <pre class="post-message">Actual <a href="#">post</a> text.<br>Second line.</pre>
          <div class="post-quoters">Replies: 123</div></div><div>Delete posts</div></main>'''
        self.assertEqual(extract(post)[1], 'Actual post text.\nSecond line.')
        self.assertEqual(extract('<main>Unknown layout with buttons and labels</main>')[1], '')

    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, prepared, run = root / "raw.jsonl", root / "prepared", root / "run"
            rows = [{"text": (f"Document {i}. The little model learns patterns in text. Åäö שלום. " * 12), "url": f"https://example.test/{i}"} for i in range(60)]
            raw.write_text("\n".join(json.dumps(row) for row in rows + rows[:2]), encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
            def command(*args):
                result = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            command("scratchllm.prepare", "--input", str(raw), "--output", str(prepared), "--vocab-size", "300", "--validation-fraction", "0.2")
            meta = json.loads((prepared / "meta.json").read_text())
            self.assertEqual(sum(meta["documents"].values()), 60)
            cfg = dict(context=16, width=32, layers=2, heads=4, dropout=0.1, batch_size=2,
                       accumulation=2, steps=2, learning_rate=0.001, min_learning_rate=0.0001,
                       warmup_steps=1, weight_decay=0.1, eval_every=1, eval_batches=2, seed=42)
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            options = ["--data", str(prepared), "--config", str(config), "--out", str(run), "--device", "cpu"]
            command("scratchllm.train", *options)
            command("scratchllm.train", *options, "--resume")
            self.assertEqual(len((run / "metrics.jsonl").read_text().splitlines()), 2)
            app = create_app(run / "best.pt", "cpu", workspace=root)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                response = client.post("/complete", json={"prompt": "The", "max_new_tokens": 4, "temperature": 0})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()["text"].startswith("The"))
                self.assertLessEqual(response.json()["generated_tokens"], 4)
                self.assertEqual(client.post("/complete", json={"max_new_tokens": -1}).status_code, 422)
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.post("/model/load", json={"checkpoint": "../outside.pt"}).status_code, 400)
                # Exercise actual child training, graceful stop/save, and resume through the API.
                payload = dict(cfg, steps=1000, data="prepared", out="gui-run", eval_every=1000)
                started = client.post("/jobs/train", json=payload)
                self.assertEqual(started.status_code, 200, started.text)
                self.assertEqual(client.post("/jobs/train", json=payload).status_code, 400)
                self.assertEqual(client.post("/complete", json={"prompt": "test"}).status_code, 409)
                def wait_for(predicate):
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        state = client.get("/jobs").json()["job"]
                        if predicate(state):
                            return state
                        if state["status"] == "failed":
                            self.fail(state["log"])
                        time.sleep(0.05)
                    self.fail("Job timed out: " + state["log"])
                wait_for(lambda job: "step 1/" in job["log"])
                client.post("/jobs/stop")
                stopped = wait_for(lambda job: job["status"] == "stopped")
                self.assertIn("Stopped after saving checkpoint", stopped["log"])
                first_step = stopped["metrics"][-1]["step"]
                resumed = client.post("/jobs/train", json={"out": "gui-run", "resume": True})
                self.assertEqual(resumed.status_code, 200, resumed.text)
                wait_for(lambda job: f"step {first_step + 1}/" in job["log"])
                client.post("/jobs/stop")
                stopped = wait_for(lambda job: job["status"] == "stopped")
                self.assertGreater(stopped["metrics"][-1]["step"], first_step)
                loaded = client.post("/model/load", json={"checkpoint": "gui-run/best.pt"})
                self.assertEqual(loaded.status_code, 200, loaded.text)
                self.assertEqual(client.get("/health").json()["checkpoint"], "gui-run/best.pt")


if __name__ == "__main__":
    unittest.main()
