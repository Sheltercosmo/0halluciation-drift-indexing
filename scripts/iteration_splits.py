"""Prepare document-disjoint improvement splits and guard one-time test access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import build_document, read_json, sha

SEED = 20261005
SOURCES = [
    {'name': 'qasper-train-dev.tgz', 'url': 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz',
     'bytes': 10835856, 'sha256': 'a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a'},
    {'name': 'quality-train.jsonl', 'url': 'https://raw.githubusercontent.com/nyu-mll/quality/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train',
     'bytes': 10881962, 'sha256': '4011e9952d5395beb8ff7637b963481a400630c1bbe2f40dc0d83f5d59f926ed'}]


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def load_jsonl(path):
    # str.splitlines also splits U+2028 inside valid JSON strings in QuALITY train.
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').split('\n') if line.strip()]


def resolve_groups(documents, cases, assignment):
    """Move all detectable duplicates toward the more exposed partition."""
    parents = {key: key for key in documents}
    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    identities = {}
    for key, doc in sorted(documents.items()):
        fields = [('text', hashlib.sha256(normalized(doc['text']).encode()).hexdigest())]
        title = normalized(doc['title'])
        if len(title) >= 20:
            fields.append(('title', title))
        for identity in fields:
            if identity in identities:
                union(key, identities[identity])
            else:
                identities[identity] = key
    # Identical generic questions are common across unrelated QASPER papers.
    # Group question+options only when document identity already matches; the
    # document union above is the leakage unit, not question wording alone.
    priority = {'development': 0, 'validation': 1, 'test': 2}
    groups = defaultdict(list)
    for key in documents:
        groups[root(key)].append(key)
    result, moved = dict(assignment), []
    for group in groups.values():
        target = min((assignment[k] for k in group), key=priority.get)
        for key in group:
            if assignment[key] != target:
                moved.append({'doc_id': key, 'from': assignment[key], 'to': target, 'group': sorted(group)})
            result[key] = target
    return result, moved


def prepare(data, output, public):
    if (output / 'manifest.json').exists():
        raise ValueError('Split registry already exists; do not silently replace it')
    sources = SOURCES + [r for r in read_json(ROOT / 'evals/frontier/sources.json')['files']
                         if r['name'] in {'qasper-test.tgz', 'quality-dev.jsonl'}]
    for row in sources:
        path = data / row['name']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Input hash mismatch: ' + row['name'])
    exposed = {r['doc_id'] for r in read_json(ROOT / 'evals/results/bounded-v1/selection.json')}
    docs, cases, gold, assignment, upstream = {}, [], {}, {}, {}
    def qasper(raw, split):
        for doc_id, paper in sorted(raw.items()):
            key = 'qasper/' + doc_id
            native = [('Abstract', [paper['abstract']])]
            native += [(s['section_name'], s['paragraphs']) for s in paper['full_text']]
            native += [('Figure and table captions', [c['caption'] for c in paper['figures_and_tables']])]
            docs[key] = build_document(key, paper['title'], native)
            upstream[key] = 'qasper/' + split
            assignment[key] = 'development' if split == 'train' or key in exposed else 'validation' if split == 'dev' else 'test'
            for q in paper['qas']:
                qid = 'qasper/' + q['question_id']
                cases.append({'id': qid, 'doc_id': key, 'dataset': 'qasper', 'question': q['question'], 'options': []})
                gold[qid] = {'answers': [a['answer'] for a in q['answers']]}
    with tarfile.open(data / 'qasper-train-dev.tgz') as archive:
        for split in ('train', 'dev'):
            qasper(json.load(archive.extractfile('qasper-' + split + '-v0.3.json')), split)
    with tarfile.open(data / 'qasper-test.tgz') as archive:
        qasper(json.load(archive.extractfile('qasper-test-v0.3.json')), 'test')
        evaluator = archive.extractfile('qasper_evaluator.py').read()
    train = load_jsonl(data / 'quality-train.jsonl')
    ids = sorted({d['article_id'] for d in train}, key=lambda x: signature({'seed': SEED, 'doc_id': 'quality/' + x}))
    validation_ids = set(ids[:50])
    for split, raw in [('train', train), ('dev', load_jsonl(data / 'quality-dev.jsonl'))]:
        for paper in raw:
            key = 'quality/' + paper['article_id']
            doc = build_document(key, paper['title'], [('', re.split(r'\n[ \t]*\n', paper['article']))])
            if key in docs and docs[key] != doc:
                raise ValueError('Same article ID contains different text: ' + key)
            docs[key], upstream[key] = doc, 'quality/' + split
            assignment[key] = 'development' if split == 'dev' or key in exposed else 'validation' if paper['article_id'] in validation_ids else 'test'
            for q in paper['questions']:
                if q['difficult']:
                    qid = 'quality/' + q['question_unique_id']
                    cases.append({'id': qid, 'doc_id': key, 'dataset': 'quality', 'question': q['question'], 'options': q['options']})
                    gold[qid] = {'label': 'ABCD'[int(q['gold_label']) - 1]}
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate question ID')
    assignment, moved = resolve_groups(docs, cases, assignment)
    if any(assignment[key] != 'development' for key in exposed):
        raise ValueError('Exposed document entered held-out data')
    registry, counts, hashes = [], {}, {}
    for partition in ('development', 'validation', 'test'):
        selected = sorted([c for c in cases if assignment[c['doc_id']] == partition], key=lambda c: c['id'])
        selected_docs = {c['doc_id'] for c in selected}
        folder = output / partition
        save(folder / 'cases.json', selected)
        save(folder / 'documents.json', {k: docs[k] for k in sorted(selected_docs)})
        save(folder / 'gold.json', {c['id']: gold[c['id']] for c in selected})
        (folder / 'qasper_evaluator.py').write_bytes(evaluator)
        for name in ('cases.json', 'documents.json', 'gold.json', 'qasper_evaluator.py'):
            hashes[partition + '/' + name] = sha(folder / name)
        counts[partition] = {dataset: {'questions': sum(c['dataset'] == dataset for c in selected),
            'documents': len({c['doc_id'] for c in selected if c['dataset'] == dataset})}
            for dataset in ('qasper', 'quality')}
        registry += [{'id': c['id'], 'doc_id': c['doc_id'], 'dataset': c['dataset'], 'partition': partition,
            'upstream_split': upstream[c['doc_id']], 'question_sha256': signature([c['question'], c['options']]),
            'document_sha256': signature(docs[c['doc_id']]['text'])} for c in selected]
    old_cases = read_json(ROOT / 'evals/results/bounded-v1/selection.json')
    screening = []
    for dataset in ('qasper', 'quality'):
        screening += sorted([c['id'] for c in old_cases if c['dataset'] == dataset],
                            key=lambda q: signature({'seed': SEED, 'screening': q}))[:64]
    manifest = {'version': 1, 'seed': SEED, 'status': 'test_locked', 'counts': counts,
        'exposed_source': 'bounded-v1', 'duplicate_moves': moved, 'sources': sources,
        'data_hashes': hashes, 'registry_sha256': signature(registry),
        'development_screening_ids': sorted(screening),
        'protocol_sha256': sha(ROOT / 'evals/ITERATION_PROTOCOL.md'), 'split_code_sha256': sha(Path(__file__))}
    save(output / 'registry.json', registry)
    save(output / 'manifest.json', manifest)
    save(public / 'registry.json', registry)
    save(public / 'manifest.json', manifest)
    print(json.dumps({'counts': counts, 'duplicate_moves': len(moved), 'test_status': 'locked',
                      'registry_sha256': manifest['registry_sha256']}))


def open_test(output, frozen):
    """Register irreversible exposure; changed configs may never reuse this test."""
    from scripts.validation_gate import verify_winner, write_once
    manifest = read_json(output / 'manifest.json')
    config = verify_winner(output, frozen)
    record = {'frozen_configuration_sha256': sha(frozen), 'registry_sha256': manifest['registry_sha256'],
              'registration_sha256': config['registration_sha256'],
              'validation_predictions_sha256': config['validation_predictions_sha256'],
              'status': 'test_opened_no_further_tuning'}
    marker = output / 'test-opened.json'
    write_once(marker, record)
    return output / 'test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'open-test'])
    parser.add_argument('--data', type=Path, default=ROOT / 'output/frontier-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/improvement-v1')
    parser.add_argument('--public', type=Path, default=ROOT / 'evals/iterations/v1')
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.data, args.output, args.public)
    else:
        if args.frozen is None:
            parser.error('--frozen is required')
        print(open_test(args.output, args.frozen))
