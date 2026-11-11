"""URL-free text and English Wikipedia scope; offline fixtures, no training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scratchllm.corpus import corpus_plan, records
from scratchllm.crawl import canonical, extract_documents
from scratchllm.jobs import Jobs, TrainRequest
from scratchllm.links import TEXT_POLICY_VERSION, strip_links
from scratchllm.sources import seed_url
from scratchllm.text_policy import EXTRACTION_VERSION


class LinkPolicyTests(unittest.TestCase):
    def test_addresses_removed_but_prose_labels_numbers_and_punctuation_survive(self):
        text = ('Read [the explanation](https://example.org/a), then https://example.org/Test_(topic).\n'
                'Contact person@example.net or www.example.com/path and example.xyz/path.\n'
                '>>12345\n>>>/g/567\nNumbers 1.25, version v2.0, e.g. ordinary prose.')
        cleaned = strip_links(text)
        for value in ('https', 'example.', '@', 'www.', '12345', '/g/567'):
            self.assertNotIn(value, cleaned)
        for value in ('the explanation', 'Numbers 1.25', 'version v2.0', 'e.g. ordinary prose.'):
            self.assertIn(value, cleaned)
        self.assertEqual(strip_links(cleaned), cleaned)
        _, rows, links = extract_documents('''<div class="post-message">The <a href="https://example.org">written explanation</a>
             includes https://example.net/path and <a href="https://example.com">https://example.com</a>.</div>
             <a href="/g/thread/123.html">Full thread</a>''', 'https://swedishwin.com/g/')
        self.assertIn('written explanation', rows[0]['text'])
        self.assertNotIn('example.', rows[0]['text'])
        self.assertIn('/g/thread/123.html', links)  # Discovery and provenance are not training text.

    def test_existing_v4_is_cleaned_before_dedupe_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old-v4.jsonl'
            original = [dict(text=('An actual written explanation with sufficient source prose. ' * 8) + url,
                             host='swedishwin.com', extraction=EXTRACTION_VERSION, url=f'https://swedishwin.com/g/{i}')
                        for i, url in enumerate(('https://first.example.org', 'https://second.example.org'))]
            original.append(dict(text='Independent article prose about a different subject. ' * 10))
            encoded = ''.join(json.dumps(row) + '\n' for row in original)
            path.write_text(encoded)
            _, report = corpus_plan(path)
            self.assertEqual(report['duplicates_removed'], 1)
            self.assertEqual(report['text_policy'], TEXT_POLICY_VERSION)
            self.assertTrue(report['ready'])
            self.assertFalse(any('https://' in row['text'] for row in records(path)))
            self.assertEqual(path.read_text(), encoded)

    def test_new_training_rejects_unclean_dataset_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            (root/'data/meta.json').write_text(json.dumps(dict(vocab_size=259, tokens=dict(train=2000, val=1000))))
            jobs = Jobs(root, 'cpu')
            with patch('scratchllm.jobs.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'predates URL removal'):
                    jobs.start('train', TrainRequest(data='data', out='run'))
            spawn.assert_not_called()
            self.assertFalse((root/'run').exists())


class WikipediaTests(unittest.TestCase):
    def test_canonical_english_articles_and_namespace_media_restrictions(self):
        self.assertEqual(seed_url('wikipedia.org'), 'https://en.wikipedia.org/wiki/Main_Page')
        for url in ('https://en.wikipedia.org/wiki/Language#History',
                    'https://en.m.wikipedia.org/wiki/Language',
                    'https://en.wikipedia.org/w/index.php?title=Language&action=view'):
            self.assertEqual(canonical(url, 'wikipedia.org'), 'https://en.wikipedia.org/wiki/Language')
        self.assertIsNotNone(canonical('https://en.wikipedia.org/wiki/Star_Trek:_Voyager', 'wikipedia.org'))
        for title in ('Audio', 'Image', 'Video'):
            self.assertIsNotNone(canonical('https://en.wikipedia.org/wiki/' + title, 'wikipedia.org'))
        _, _, links = extract_documents('<a href="/wiki/Audio">Audio article</a><a href="/wiki/File:Song.mp3">File</a>', seed_url('wikipedia.org'))
        self.assertEqual(links, ['/wiki/Audio'])
        for url in ('https://en.wikipedia.org/wiki/Talk:Language', 'https://en.wikipedia.org/wiki/Category:Languages',
                    'https://en.wikipedia.org/wiki/User:Example', 'https://en.wikipedia.org/wiki/File:Image.jpg',
                    'https://en.wikipedia.org/wiki/Special:Random', 'https://en.wikipedia.org/wiki/Help:Contents',
                    'https://en.wikipedia.org/w/index.php?title=Language&action=edit',
                    'https://en.wikipedia.org/w/index.php?title=Language&oldid=12',
                    'https://en.wikipedia.org/w/api.php', 'https://en.wikipedia.org/wiki/Language?diff=123',
                    'https://fr.wikipedia.org/wiki/Langue', 'https://upload.wikimedia.org/a.jpg',
                    'https://en.wikipedia.org.evil.test/wiki/Language'):
            self.assertIsNone(canonical(url, 'wikipedia.org'), url)

if __name__ == '__main__':
    unittest.main()
