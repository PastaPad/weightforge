"""Corpus quality, grouped splits, and reports. Uses a fake tokenizer; no training."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from scratchllm import prepare
from scratchllm.corpus import corpus_plan, records
from scratchllm.crawl import extract_documents
from scratchllm.frontier import Frontier
from scratchllm.jobs import Jobs
from scratchllm.serve import create_app
from scratchllm.text_policy import EXTRACTION_VERSION
from test_serving_selection import FakeGenerator


def corpus_rows():
    rows = []
    for post, thread, text in [(11, 10, 'First discussion: a clear sentence about local history and its context. ' * 2),
                               (12, 10, 'A reply adds another written explanation and answers the earlier post. ' * 2),
                               (21, 20, 'Another independent discussion about a different topic and its details. ' * 2)]:
        rows.append(dict(text=text, url=f'https://swedishwin.com/g/thread/{thread}.html#{post}',
                         group=f'swedishwin.com/g/thread/{thread}', kind='post', host='swedishwin.com', extraction=EXTRACTION_VERSION))
    rows.append(dict(text='An independent wiki article contains complete paragraphs and useful context. ' * 12,
                     url='https://soyjakwiki.org/History', group='https://soyjakwiki.org/History', kind='article',
                     host='soyjakwiki.org', extraction=EXTRACTION_VERSION))
    rows += [dict(rows[0], text=rows[0]['text'].replace(' ', '  ')), dict(rows[2], text='short')]
    return rows


def write_corpus(path, rows=None):
    path.write_text(''.join(json.dumps(row) + '\n' for row in (corpus_rows() if rows is None else rows)))


class FakeTokenizer:
    fitted_texts = None

    def __init__(self, *_):
        pass

    def train_from_iterator(self, texts, trainer):
        # Record the input only; no tokenizer/model fitting occurs.
        type(self).fitted_texts = list(texts)

    def token_to_id(self, value):
        return prepare.SPECIALS.index(value)

    def encode(self, text):
        return SimpleNamespace(ids=[byte + 3 for byte in text.encode()])

    def save(self, path):
        Path(path).write_text('{"fixture": true}')

    def get_vocab_size(self):
        return 259


class DataQualityTests(unittest.TestCase):
    def test_plain_text_preparation_preserves_source_and_splits_before_tokenizing(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            source, output = root/'sentences.txt', root/'prepared'
            raw = ('\ufeffRäksmörgås i solen. https://example.org/one\r\n'
                   'En kort andra mening.\r\n'
                   'Räksmörgås  i solen. https://example.org/two\r\n'
                   'En tredje text.\r\n\r\n').encode('utf-8')
            source.write_bytes(raw)
            self.assertEqual(corpus_plan(source)[1]['too_short_removed'], 4)
            with patch('scratchllm.prepare.Tokenizer', FakeTokenizer), contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare_atomic(SimpleNamespace(input=source, output=output, vocab_size=259,
                    validation_fraction=0.25, min_chars=1, min_post_chars=40))
            splits = {split: list(map(json.loads, (output/f'{split}.jsonl').read_text().splitlines()))
                      for split in ('train', 'val')}
            self.assertTrue(all(splits.values()))
            rows = splits['train'] + splits['val']
            self.assertEqual({row['text'] for row in rows},
                {'Räksmörgås i solen.', 'En kort andra mening.', 'En tredje text.'})
            self.assertFalse({row['group'] for row in splits['train']} & {row['group'] for row in splits['val']})
            self.assertEqual(FakeTokenizer.fitted_texts, [row['text'] for row in splits['train']])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['text_unit'], 'lines')
            self.assertEqual(report['duplicates_removed'], 1)
            self.assertEqual(report['source'], dict(path=str(source), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            self.assertEqual({row['source']['start_line'] for row in rows}, {1, 2, 4})
            self.assertTrue(all(row['source']['start_line'] == row['source']['end_line'] for row in rows))
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(corpus_plan(source, 0.25, 1)[0], {row['group'] for row in splits['val']})

    def test_plain_text_paragraphs_stay_whole_and_invalid_text_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            source = Path(directory)/'paragraphs.TXT'
            source.write_text('\nFirst paragraph.\nIts second line.\n \n'
                              'Another paragraph.\nIts last line.\n\n', encoding='utf-8')
            rows = list(records(source, min_chars=1))
            self.assertEqual([row['text'] for row in rows],
                ['First paragraph.\nIts second line.', 'Another paragraph.\nIts last line.'])
            self.assertEqual([(row['source']['start_line'], row['source']['end_line']) for row in rows], [(2, 3), (5, 6)])
            validation, report = corpus_plan(source, min_chars=1)
            self.assertEqual(report['text_unit'], 'paragraphs')
            self.assertEqual(report['documents'], dict(train=1, val=1))
            self.assertEqual(len(validation), 1)
            for raw, message in ((b'Invalid\xffUTF-8', 'UTF-8'), (b'Binary\x00text', 'control')):
                source.write_bytes(raw)
                with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
                    list(records(source, min_chars=1))

    def test_post_records_use_metadata_permalink_and_skip_truncated_previews(self):
        html = '''<div class="thread" id="thread_100">
          <div class="post op" id="op_100"><div class="post-info"><a href="/g/thread/100.html#100">No.100</a></div>
            <div class="post-message">The actual opening post links to <a href="/q/thread/200.html">another discussion</a>.
            <span class="noselect">[Expand]</span></div></div>
          <div class="post reply" id="reply_101"><div class="post-info"><a href="/g/res/100.html#101">No.101</a></div>
            <div class="post-message">A complete written reply to the opening post with its own details.</div></div>
          <div class="post reply" id="reply_102"><div class="post-message">An incomplete preview
            <div class="cb">Message too long. <a class="viewfulltext" href="/g/thread/100.html#102">View the full text</a></div></div></div>
        </div>'''
        _, documents, links = extract_documents(html, 'https://www.swedishwin.com/overboard.html')
        self.assertEqual(len(documents), 2)
        self.assertEqual({row['group'] for row in documents}, {'swedishwin.com/g/thread/100'})
        self.assertEqual(documents[0]['url'], 'https://www.swedishwin.com/g/thread/100.html#100')
        self.assertEqual([row['post_id'] for row in documents], ['100', '101'])
        self.assertIn('/g/thread/100.html#102', links)
        self.assertNotIn('Expand', documents[0]['text'])
        self.assertNotIn('preview', ' '.join(row['text'] for row in documents))
        self.assertTrue(all(row['kind'] == 'post' for row in documents))

    def test_grouped_split_is_repeatable_and_removes_duplicates_before_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'corpus.jsonl'
            write_corpus(path)
            validation, report = corpus_plan(path, validation_fraction=0.2)
            accepted = list(records(path))
            self.assertEqual(len(accepted), 4)
            self.assertEqual(report['groups'], 3)
            self.assertEqual(report['duplicates_removed'], 1)
            self.assertEqual(report['too_short_removed'], 1)
            self.assertEqual(report['sources']['swedishwin.com']['accepted_records'], 3)
            self.assertTrue(report['ready'])
            train = [row for row in accepted if row['group'] not in validation]
            val = [row for row in accepted if row['group'] in validation]
            self.assertTrue(train and val)
            self.assertFalse({row['group'] for row in train} & {row['group'] for row in val})
            self.assertFalse({row['sha256'] for row in train} & {row['sha256'] for row in val})
            same_thread = [row for row in accepted if '/thread/10' in row['group']]
            self.assertEqual(len({row['group'] in validation for row in same_thread}), 1)
            write_corpus(path, list(reversed(corpus_rows())))
            other, other_report = corpus_plan(path, validation_fraction=0.2)
            self.assertEqual(validation, other)
            self.assertEqual(report['documents'], other_report['documents'])

    def test_single_thread_is_not_split_into_a_leaky_validation_set(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'one-thread.jsonl', Path(directory)/'prepared'
            write_corpus(source, corpus_rows()[:2])
            self.assertFalse(corpus_plan(source)[1]['ready'])
            with patch('sys.argv', ['prepare', '--input', str(source), '--output', str(output)]), patch('scratchllm.prepare.Tokenizer') as tokenizer, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    prepare.main()
            tokenizer.assert_not_called()
            self.assertFalse(output.exists())

    def test_prepare_uses_only_training_text_and_persists_source_token_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'corpus.jsonl', root/'prepared'
            write_corpus(source)
            queue = Frontier(source, dict(hosts=['swedishwin.com', 'soyjakwiki.org', 'soyjak.st'], pages_per_site=10))
            queue.close()
            with patch('scratchllm.prepare.Tokenizer', FakeTokenizer), patch('sys.argv', [
                    'prepare', '--input', str(source), '--output', str(output), '--validation-fraction', '0.2']), contextlib.redirect_stdout(io.StringIO()):
                prepare.main()
            splits = {split: [json.loads(line) for line in (output/f'{split}.jsonl').read_text().splitlines()] for split in ('train', 'val')}
            self.assertEqual(FakeTokenizer.fitted_texts, [row['text'] for row in splits['train']])
            self.assertFalse(set(FakeTokenizer.fitted_texts) & {row['text'] for row in splits['val']})
            meta = json.loads((output/'meta.json').read_text())
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(meta['corpus_report'], report)
            self.assertEqual(report['sources']['soyjak.st']['accepted_records'], 0)
            self.assertEqual(report['sources']['soyjak.st']['tokens'], dict(train=0, val=0))
            for split in splits:
                self.assertEqual(meta['tokens'][split], sum(host['tokens'][split] for host in report['sources'].values()))
                self.assertEqual((output/f'{split}.bin').stat().st_size, 4 * meta['tokens'][split])
                self.assertEqual(meta['documents'][split], len(splits[split]))

    def test_prepared_splits_reuse_report_without_losing_group_or_extraction_identity(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            root = Path(directory)
            source, output = root/'raw.jsonl', root/'prepared'
            write_corpus(source)
            with patch('scratchllm.prepare.Tokenizer', FakeTokenizer), contextlib.redirect_stdout(io.StringIO()):
                prepare.prepare(SimpleNamespace(input=source, output=output, min_chars=300,
                    min_post_chars=40, validation_fraction=0.2, vocab_size=259))
            path = output/'train.jsonl'
            original = list(map(json.loads, path.read_text().splitlines()))
            copied = root/'copied.jsonl'
            copied.write_bytes(path.read_bytes())
            prepare.validate_corpus(copied)
            self.assertEqual(list(records(copied)), original)
            # Earlier prepared files retained host/group/hash, but omitted row versions.
            legacy = [{key: value for key, value in row.items() if key not in {'extraction', 'text_policy'}} for row in original]
            path.write_text(''.join(json.dumps(row) + '\n' for row in legacy))
            legacy_bytes = path.read_bytes()
            prepare.validate_corpus(path)
            self.assertEqual(list(records(path)), original)
            self.assertEqual(path.read_bytes(), legacy_bytes)
            copied.write_bytes(legacy_bytes)
            with self.assertRaisesRegex(ValueError, 'unversioned'):
                prepare.validate_corpus(copied)
            # An explicit obsolete row is rejected even alongside a current report.
            path.write_text(json.dumps(dict(legacy[0], extraction='article-post-text-v3')) + '\n')
            with self.assertRaisesRegex(ValueError, 'older'):
                list(records(path))
            path.write_bytes(legacy_bytes)
            meta_path = output/'meta.json'
            meta = json.loads(meta_path.read_text())
            meta['corpus_report']['extraction'] = 'article-post-text-v3'
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'older'):
                prepare.validate_corpus(path)

    def test_inspection_api_is_read_only_and_reports_actual_sources(self):
        with tempfile.TemporaryDirectory() as directory, patch('scratchllm.serve.Generator', FakeGenerator), patch.object(Jobs, 'start', side_effect=AssertionError('No jobs expected')) as start:
            root = Path(directory)
            write_corpus(root/'corpus.jsonl')
            (root/'fake.pt').write_text('fake')
            before = set(root.rglob('*'))
            with TestClient(create_app(root/'fake.pt', 'cpu', workspace=root)) as client:
                response = client.post('/corpus/inspect', json={'input': 'corpus.jsonl'})
                self.assertEqual(response.status_code, 200, response.text)
                report = response.json()
                self.assertEqual(report['accepted_records'], 4)
                self.assertEqual(set(report['sources']), {'swedishwin.com', 'soyjakwiki.org'})
                self.assertTrue(report['sources']['swedishwin.com']['samples'])
                self.assertIsNone(client.get('/jobs').json()['job'])
                self.assertEqual(client.post('/corpus/inspect', json={'input': '../outside.jsonl'}).status_code, 400)
            self.assertEqual(before, set(root.rglob('*')))
            start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
