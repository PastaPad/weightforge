"""Shared corpus normalization, post deduplication, grouped splits, and source reports."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

from .text_policy import EXTRACTION_VERSION
from .frontier import read_state, state_path
from .links import TEXT_POLICY_VERSION, strip_links
from .sources import WIKIPEDIA_HOSTS


def clean(text):
    text = unicodedata.normalize('NFKC', text)
    return strip_links('\n'.join(' '.join(line.split()) for line in text.splitlines()).strip())


def content_hash(text):
    return hashlib.sha256(re.sub(r'\s+', ' ', clean(text)).encode()).hexdigest()


def source_host(row):
    host = (urlsplit(row.get('url') or '').hostname or row.get('host') or '(local)').removeprefix('www.')
    return 'wikipedia.org' if host in WIKIPEDIA_HOSTS or host == 'simple.wikipedia.org' else host


def prepared_source_report(path):
    """Recognize older prepared splits whose extraction version lives in a sidecar."""
    path = Path(path).resolve()
    if path.name not in {'train.jsonl', 'val.jsonl'}:
        return None
    for name in ('meta.json', 'hf-meta.json', 'report.json'):
        metadata = path.parent/name
        if not metadata.is_file():
            continue
        try:
            data = json.loads(metadata.read_text(encoding='utf-8'))
            report = data if name == 'report.json' else data.get('corpus_report', {})
            if report.get('protocol') != 'grouped-text-v1':
                continue
            count = report.get('documents', {}).get(path.stem)
            if report.get('extraction') != EXTRACTION_VERSION or type(count) is not int or count < 1:
                return None
            return report
        except (OSError, ValueError, AttributeError):
            return None
    return None


def validate_extraction(row, prepared=False):
    if not isinstance(row, dict):
        raise ValueError('each JSONL record must be an object')
    if ('extraction' not in row and 'host' not in row) or row.get('extraction') == EXTRACTION_VERSION:
        return
    # Only normalized prepared records can inherit a verified dataset report.
    # An explicit obsolete extraction marker must never be overridden by it.
    if (prepared and 'extraction' not in row and isinstance(row.get('text'), str)
            and all(isinstance(row.get(key), str) for key in ('host', 'group', 'kind', 'sha256'))
            and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
        return
    raise ValueError(f'older or unversioned crawler extraction; recollect with {EXTRACTION_VERSION}, '
                     'or keep prepared train.jsonl/val.jsonl beside its matching dataset metadata/report')


def text_units(path):
    """Keep blank-line-delimited paragraphs together; otherwise use nonempty lines."""
    seen_text = separator = False
    with open(path, encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                if separator:
                    return 'paragraphs'
                seen_text = True
            elif seen_text:
                separator = True
    return 'lines'


def source_records(path):
    path = Path(path)
    try:
        if path.suffix.lower() != '.txt':
            with path.open(encoding='utf-8') as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        yield number, json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f'Corpus line {number}: {exc}') from exc
            return
        paragraphs = text_units(path) == 'paragraphs'
        origin = str(path.resolve())
        pieces, start, end = [], None, None
        def row():
            return dict(text='\n'.join(pieces), source=dict(path=origin, start_line=start, end_line=end))
        with path.open(encoding='utf-8-sig') as stream:
            for number, line in enumerate(stream, 1):
                if re.search(r'[\x00-\x08\x0b\x0e-\x1f]', line):
                    raise ValueError(f'Corpus line {number}: plain text contains binary/control characters')
                text = line.rstrip('\r\n')
                if text.strip():
                    if start is None:
                        start = number
                    end = number
                    pieces.append(text)
                if pieces and (not paragraphs or not text.strip()):
                    yield start, row()
                    pieces, start, end = [], None, None
            if pieces:
                yield start, row()
    except UnicodeDecodeError as exc:
        raise ValueError('Save the corpus as UTF-8 text') from exc


def records(path, min_chars=300, min_post_chars=40, report=None):
    seen = set()
    prepared = prepared_source_report(path) is not None
    for number, row in source_records(path):
        try:
            if not isinstance(row.get('text'), str):
                raise ValueError('text must be a string')
            validate_extraction(row, prepared)
            text = clean(row['text'])
            host = source_host(row)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError(f'Corpus line {number}: {exc}') from exc
        source = None
        if report is not None:
            source = report['sources'].setdefault(host, dict(input_records=0, accepted_records=0, duplicates_removed=0,
                too_short_removed=0, characters=0, text_bytes=0, documents=dict(train=0, val=0), samples=[]))
            report['input_records'] += 1
            source['input_records'] += 1
        threshold = min_post_chars if row.get('kind') == 'post' else min_chars
        reason = 'too_short_removed' if len(text) < threshold else None
        digest = content_hash(text)
        if reason is None and digest in seen:
            reason = 'duplicates_removed'
        if reason:
            if report is not None:
                report[reason] += 1
                source[reason] += 1
            continue
        seen.add(digest)
        # New post records share a thread group. Older/custom documents use
        # content identity, retaining whole-document deduplication support.
        group = str(row['group']) if row.get('group') else 'document:' + digest
        if row.get('group') and not prepared and not group.startswith(host + ':'):
            group = f'{host}:{group}'
        result = dict(text=text, url=row.get('url'), sha256=digest, host=host,
                      group=group, kind=row.get('kind', 'document'),
                      extraction=EXTRACTION_VERSION, text_policy=TEXT_POLICY_VERSION)
        if isinstance(row.get('source'), dict):
            result['source'] = row['source']
        if report is not None:
            report['accepted_records'] += 1
            source['accepted_records'] += 1
            source['characters'] += len(text)
            source['text_bytes'] += len(text.encode('utf-8'))
            source['samples'].append(dict(url=result['url'], text=text[:500], sha256=digest))
            source['samples'] = sorted(source['samples'], key=lambda sample: sample['sha256'])[:2]
        yield result


def corpus_plan(path, validation_fraction=0.1, min_chars=300, min_post_chars=40):
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one')
    if min(min_chars, min_post_chars) < 1:
        raise ValueError('Minimum article/post lengths must be positive')
    report = dict(protocol='grouped-text-v1', extraction=EXTRACTION_VERSION, text_policy=TEXT_POLICY_VERSION, input_records=0, accepted_records=0,
                  duplicates_removed=0, too_short_removed=0, sources={}, documents=dict(train=0, val=0))
    groups = {}
    for row in records(path, min_chars, min_post_chars, report):
        group = groups.setdefault(row['group'], dict(host=row['host'], characters=0, records=0))
        group['characters'] += len(row['text'])
        group['records'] += 1
    # Split whole threads/articles before fitting the tokenizer. Character weighting
    # limits skew from very large threads; stable hashes make tie-breaking repeatable.
    target = sum(group['characters'] for group in groups.values()) * validation_fraction
    ordered = sorted(groups, key=lambda key: (-groups[key]['characters'], hashlib.sha256(key.encode()).hexdigest()))
    validation, size = set(), 0
    for key in ordered:
        weight = groups[key]['characters']
        if len(validation) + 1 < len(groups) and abs(target - size - weight) < abs(target - size):
            validation.add(key)
            size += weight
    if not validation and len(groups) > 1:
        validation.add(ordered[-1])
    for key, group in groups.items():
        split = 'val' if key in validation else 'train'
        report['documents'][split] += group['records']
        report['sources'][group['host']]['documents'][split] += group['records']
    report.update(groups=len(groups), validation_groups=len(validation), requested_validation_fraction=validation_fraction,
                  validation_character_fraction=sum(groups[key]['characters'] for key in validation) / max(1, sum(g['characters'] for g in groups.values())),
                  ready=bool(report['documents']['train'] and report['documents']['val']))
    if Path(path).suffix.lower() == '.txt':
        with open(path, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report.update(input_format='text', text_unit=text_units(path),
                      source=dict(path=str(Path(path).resolve()), bytes=Path(path).stat().st_size, sha256=digest))
    if state_path(path).is_file():
        saved = read_state(path)
        report['collection'] = saved['stats'].get('sources', {})
        report['requested_sources'] = saved['settings']['hosts']
        for host in report['requested_sources']:
            report['sources'].setdefault(host, dict(input_records=0, accepted_records=0, duplicates_removed=0,
                too_short_removed=0, characters=0, text_bytes=0, documents=dict(train=0, val=0), samples=[]))
    return validation, report
