"""Publish complete live results with exact source spans and byte-hashed provenance."""
from collections import Counter
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.live_tree_eval import OUT, OLD, METHODS, verify
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import read_json, sha


def export(source, destination):
    manifest = verify(source)
    cases, rows = read_json(source / 'cases.json'), read_json(source / 'scores.json')
    expected = {(c['id'], m) for c in cases for m in METHODS}
    if Counter((r['id'], r['method']) for r in rows) != Counter({k: 1 for k in expected}):
        raise ValueError('Refuse to publish incomplete results')
    old = read_json(OLD / 'manifest.json')
    for name, digest in old['data_hashes'].items():
        if sha(OLD / name) != digest:
            raise ValueError('Official evaluation input changed: ' + name)
    destination.mkdir(parents=True, exist_ok=True)
    retrieval = {c['id']: read_json(source / 'retrieval' / (signature(c['id']) + '.json')) for c in cases}
    for row in rows:
        context = retrieval[row['id']]['methods'][row['method']]
        row['spans'] = context['spans']
        row['context_sha256'] = hashlib.sha256(context['context'].encode()).hexdigest()
    save(destination / 'scores.json', rows)
    for name in ['manifest.json', 'cases.json', 'summary.json', 'analysis.json', 'paired-intervals.json',
                 'failure-comparisons.json', 'routing-analysis.json', 'usage.json', 'partition-reconstruction-audit.json', 'runtime.json',
                 'ranker-normalizations.json', 'reader-identity-normalizations.json', 'common-unblocked-summary.json']:
        (destination / name).write_bytes((source / name).read_bytes())
    for name, digest in manifest['source_hashes'].items():
        path = source / 'source' / name
        if sha(path) != digest:
            raise ValueError('Source archive changed: ' + name)
        target = destination / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    plans = [read_json(source / 'plans' / (signature(c['id']) + '.json')) for c in cases]
    save(destination / 'query-plans.json', plans)
    provenance = {'official_input_hashes': old['data_hashes'], 'runtime_source_hashes': manifest['source_hashes'],
                  'manifest_sha256': sha(source / 'manifest.json'),
                  'retrieval_sha256': {p.name: sha(p) for p in (source / 'retrieval').glob('*.json')},
                  'index_sha256': {p.name: sha(p) for p in (source / 'indexes').glob('*.json')},
                  'audit_sha256': {p.name: sha(p) for p in source.glob('*-audit.jsonl')},
                  'replay_scope': 'Official answer and evidence scores, source spans, context budgets, all aggregates and paired intervals; no model calls.'}
    save(destination / 'provenance.json', provenance)
    save(destination / 'budget.json', read_json(ROOT / 'output/research-budget.json'))
    for path in (source / 'amendments').glob('*.json'):
        target = destination / 'amendments' / path.name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
    print({'exported_predictions': len(rows), 'destination': str(destination)})


if __name__ == '__main__':
    export(OUT, ROOT / 'evals/results/live-tree-v1')
