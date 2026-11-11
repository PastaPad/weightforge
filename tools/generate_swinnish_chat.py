"""Create synthetic, role-labelled Swinnish conversations without training a model."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from generate_swinnish_ovs import FRAMES, SUBJECTS, family_split, sentence


def generate():
    rows, seen = [], set()

    def add(turns, family, task):
        messages = [dict(role='user' if i % 2 == 0 else 'assistant', content=text)
                    for i, text in enumerate(turns)]
        identity = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if identity in seen:
            return
        seen.add(identity)
        rows.append(dict(id=hashlib.sha256(identity.encode()).hexdigest(),
                         messages=messages, group='swinnish-chat:' + family,
                         split=family_split(family), task=task, synthetic=True,
                         language='Swinnish', reference_revision='2026-10-31',
                         word_order='OVS', source='owner-reference-template-generation'))

    for verb, objects in FRAMES.items():
        for noun, modifiers in objects:
            family = f'{verb}:{noun}'
            for subject in SUBJECTS[5:15]:
                for tense in ['e', 'i', 'vil']:
                    fact = sentence(f'zea {noun} {tense} {verb} {subject}')
                    what = f'Hva {tense} {verb} {subject}?'
                    who = f'Zea {noun} {tense} {verb} hvafono?'
                    add([fact + ' ' + what, fact, who, fact], family, 'fact_questions')
                    add([f'To mo skir su: {fact}', fact, 'Meetnu.', fact],
                        family, 'repeat_requested_text')
                    # A personal fact is answered with speaker-relative pronouns.
                    # Only once per verb/object/tense, deduplicated across subjects.
                    personal = sentence(f'zea {noun} {tense} {verb} ego')
                    answer = sentence(f'zea {noun} {tense} {verb} su')
                    add([personal + f' Hva {tense} {verb} ego?', answer,
                         f'Zea {noun} {tense} {verb} hvafono?', answer],
                        family, 'speaker_perspective')
                    # Contrastive context updates: preserve negation in the reply.
                    negative = sentence(f'zea {noun} eo {tense} {verb} {subject}')
                    add([f'To mo skir su: {fact}', fact,
                         f'To mo skir su: {negative}', negative], family, 'negation_update')
                    if modifiers:
                        modifier = modifiers.split()[0]
                        detail = sentence(f'zea {modifier} {noun} {tense} {verb} {subject}')
                        add([detail + ' ' + what, detail, who, detail],
                            family, 'retain_modifier')
            # Context accumulation with two different actors, followed by retrieval.
            for first, second in [('ze mama', 'ze papa'), ('ze fono', 'ze doktos')]:
                past = sentence(f'zea {noun} i {verb} {first}')
                future = sentence(f'zea {noun} vil {verb} {second}')
                add([past + f' Zea {noun} i {verb} hvafono?', past,
                     future + f' Zea {noun} vil {verb} hvafono?', future,
                     f'Zea {noun} i {verb} hvafono?', past], family, 'recall_earlier_turn')
    random.Random(20261101).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = generate()
    args.output.mkdir(parents=True, exist_ok=False)
    counts, tasks, hashes = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        selected = [row for row in rows if row['split'] == split]
        assert selected
        counts[split] = len(selected)
        tasks[split] = dict(collections.Counter(row['task'] for row in selected))
        # Standard messages-only export for trainers that require this schema.
        for filename, values in [
            (f'{split}.jsonl', selected),
            (f'{split}.messages.jsonl', [dict(messages=row['messages']) for row in selected]),
            (f'{split}.transcripts.jsonl', [dict(text='\n'.join(
                ('User:' if message['role'] == 'user' else 'Assistant:') + '\n' + message['content']
                for message in row['messages']), group=row['group'], synthetic=True) for row in selected]),
        ]:
            path = args.output/filename
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values), encoding='utf-8')
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(synthetic=True, word_order='OVS', total_conversations=len(rows),
                    split_counts=counts, tasks=tasks, sha256=hashes,
                    split_method='Whole verb/object families, same assignment as the OVS corpus',
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    vocabulary_generator_sha256=hashlib.sha256(Path(__file__).with_name('generate_swinnish_ovs.py').read_bytes()).hexdigest())
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(total=len(rows), counts=counts, tasks=tasks), indent=2))


if __name__ == '__main__':
    main()
