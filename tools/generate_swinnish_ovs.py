"""Reproducible synthetic Swinnish 1.5 corpus; no model training or dependencies."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

SEED = 20261031
# Vocabulary manually selected from the supplied 2026-10-31 reference.
# Only semantically compatible verb/object/modifier combinations are expanded.
FRAMES = {
    'edza': [('breu', 'arian teep'), ('3ebl', 'soet'), ('zaa', 'teep'), ('blub', 'kald'), ('embi', 'kald')],
    'skir': [('keel', 'nu'), ('norti', 'nu'), ('repi', 'nu'), ('haxem', ''), ('tsemtcata', 'nu')],
    'qes': [('hema', 'nu sto'), ('rad', 'nu viid'), ('voravoz', 'nu sto'), ('sqe', 'nu sto'), ('taol', 'nu pedi')],
    'heelm': [('hema', 'nu'), ('kaka', ''), ('fonoim', ''), ('edza', ''), ('keelboad', '')],
    'tek': [('keel', 'nu'), ('mulo', 'nu'), ('cata', 'hvit'), ('panu', 'hvit'), ('breu', 'teep')],
    'zdotek': [('keel', 'nu'), ('mulo', ''), ('cata', 'hvit'), ('seni', ''), ('tuxcata', '')],
    'zi': [('fugl', 'hvit'), ('olf', ''), ('voravoz', 'sto'), ('3eie', 'sto'), ('plolix', '')],
    'lys': [('fugl', ''), ('bist', ''), ('deiu', ''), ('voravoz', ''), ('bol', '')],
    'xidza': [('vora', ''), ('edza', ''), ('keel', ''), ('rad', ''), ('mulo', '')],
    'gna': [('haxem', ''), ('lelo', ''), ('rad', ''), ('fono', '')],
    'loha': [('mama', ''), ('papa', ''), ('hema', ''), ('fonoim', '')],
    'gerdo': [('dura', ''), ('keel', ''), ('sqe', '')],
    'mulo': [('mulo', 'nu'), ('voz', 'nu'), ('keel', ''), ('knuu', '')],
    'kasta': [('bol', ''), ('tson', '')],
    'liid': [('liid', 'nu arian')],
    'ana': [('mama', ''), ('papa', ''), ('oot', ''), ('doktos', '')],
    'aux': [('fono', ''), ('kaka', ''), ('doktos', ''), ('dorufono', '')],
    'snope': [('tuxcata', ''), ('keel', ''), ('rad', ''), ('mulo', '')],
    'pebie': [('keel', ''), ('rad', ''), ('mulo', ''), ('panu', '')],
    'meet': [('keel', 'nu'), ('hema', ''), ('mulo', ''), ('sqe', '')],
    'syia': [('panu', ''), ('toog', '')],
}
SUBJECTS = ['ego', 'su', 'kie', 'xe', 'egoim', 'ze fono', 'ze kaka',
            'ze mama', 'ze papa', 'ze doktos', 'ze dorufono', 'ze fonoim',
            'ze kakaim', 'ze oot', 'ze moi', 'ze foi', 'ze arian fono',
            'ze soet fono', 'ze multid dorufono', 'ze voravozqesere']
TENSES = ['e', 'i', 'vil', 'ed']
PREDICATES = [('hema', 'sto'), ('rad', 'viid'), ('boad', 'suarti'),
              ('breu', 'teep'), ('3ebl', 'soet'), ('cata', 'hvit'),
              ('mulo', 'nu'), ('tson', 'tugu'), ('vora', 'tebi')]


def sentence(text):
    return text[0].upper() + text[1:] + '.'


def family_split(family):
    # All inflections, subjects and article variants of a verb/object pair
    # stay together. This is stronger than a random sentence-level split.
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
    return 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'


def generate():
    rows, seen = [], set()

    def add(text, family, pattern, **details):
        text = sentence(text)
        if text in seen:
            return
        seen.add(text)
        rows.append(dict(text=text, group='swinnish-1.5:' + family,
                         split=family_split(family), pattern=pattern,
                         status='synthetic', language='Swinnish',
                         reference_revision='2026-10-31', word_order='OVS',
                         **details))

    for verb, objects in FRAMES.items():
        for noun, modifiers in objects:
            family = f'{verb}:{noun}'
            for modifier in ['', *modifiers.split()]:
                obj = ' '.join(filter(None, (modifier, noun)))
                for article in ['', 'zea ']:
                    for subject in SUBJECTS:
                        # Continuous only for dynamic events, not knowledge,
                        # possession or enduring affection.
                        for tense in TENSES:
                            if tense == 'ed' and verb in {'gna', 'meet', 'loha'}:
                                continue
                            for negation in ['', 'eo ', 'nie ']:
                                add(f'{article}{obj} {negation}{tense} {verb} {subject}',
                                    family, 'transitive', verb=verb, object=noun,
                                    subject=subject, tense=tense, negation=negation.strip())

    # Keep the explicitly recorded copula-free predicate construction.
    for noun, adjective in PREDICATES:
        add(f'ze {noun} {adjective}', f'predicate:{noun}', 'adjective_predicate')
        for owner in ['mama', 'papa', 'oot', 'doktos']:
            add(f'zei {owner} {noun} {adjective}', f'predicate:{noun}', 'genitive_predicate')

    # Coherent short paragraphs: each sequence shares a topic and protagonist.
    # Dedicated verb/object families keep constituent sentences out of the
    # single-sentence examples, including across train/validation/test.
    scenes = [
        ('garden', [('si', 'vac'), ('zdotek', 'vora'), ('zi', 'seni')]),
        ('library', [('xidza', 'norti'), ('tek', 'norti'), ('skir', 'okuu')]),
        ('boatwork', [('snope', 'doru'), ('tek', 'doru'), ('qes', 'voravozsqement')]),
        ('music', [('si', 'sau'), ('pebie', 'liid'), ('skir', 'liid')]),
        ('records', [('xidza', 'tuxcata'), ('tek', 'tuxcata'), ('zdotek', 'tsemtcata')]),
        ('meal', [('pebie', 'edza'), ('tek', 'edza'), ('edza', 'edza')]),
        ('cloth', [('pebie', 'toog'), ('tek', 'toog'), ('zdotek', 'panu')]),
        ('carving', [('pebie', 'doru'), ('zdotek', 'doru'), ('qes', 'ment')]),
    ]
    for topic, actions in scenes:
        assert not any(noun == n for v, noun in actions for n, _ in FRAMES.get(v, []))
        for subject in SUBJECTS:
            for tense in ['e', 'i', 'vil', 'ed']:
                text = ' '.join(sentence(f'zea {noun} {tense} {verb} {subject}') for verb, noun in actions)
                add(text[:-1], f'scene:{topic}', 'three_sentence_scene', topic=topic)
    random.Random(SEED).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = generate()
    # Refuse to overwrite an existing corpus.
    args.output.mkdir(parents=True, exist_ok=False)
    counts, patterns, hashes = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        selected = [row for row in rows if row['split'] == split]
        assert selected
        counts[split] = len(selected)
        patterns[split] = dict(collections.Counter(row['pattern'] for row in selected))
        for suffix in ['sw.txt', 'jsonl']:
            path = args.output/f'{split}.{suffix}'
            with path.open('w', encoding='utf-8', newline='\n') as stream:
                for row in selected:
                    stream.write((row['text'] if suffix == 'sw.txt' else json.dumps(row, ensure_ascii=False)) + '\n')
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(language='Swinnish', reference_revision='2026-10-31',
                    grammar='1.5 OVS, explicitly confirmed by owner', seed=SEED,
                    synthetic=True, total_records=len(rows), split_counts=counts,
                    patterns=patterns, hashes=hashes,
                    split_method='SHA-256 lexical verb/object families; target 90/5/5, not exact',
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
