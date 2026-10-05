"""Analyze the complete isolated-reader development run, never held-out data."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save
from scripts.bounded_eval import paired_interval, read_json, sha
from scripts.isolated_reader_trial import METHODS, verify


def analyze(output):
    manifest = verify(output)
    rows = read_json(output / 'scores.json')
    cases = read_json(output / 'cases.json')
    expected = {(c['id'], m) for c in cases for m in METHODS}
    if Counter((r['id'], r['method']) for r in rows) != Counter({k:1 for k in expected}):
        raise ValueError('Analyze only complete registered prediction coverage')
    jobs = read_json(output / 'requests.json')
    if len(list((output / 'predictions').glob('*.json'))) != len(jobs):
        raise ValueError('Unique-request coverage mismatch')
    keyed = {(r['id'], r['method']):r for r in rows}
    intervals = {}
    for candidate in METHODS:
        if candidate == 'codex_pool_8192':
            continue
        for baseline in ('jev_blocking_rerank', 'codex_pool_8192'):
            if candidate == baseline:
                continue
            for dataset in ('qasper','quality'):
                pair = [{**r, 'method':'jev_blocking_rerank' if r['method']==candidate else 'comparison'}
                        for r in rows if r['dataset']==dataset and r['method'] in (candidate,baseline)]
                intervals[f'{candidate}/{baseline}/{dataset}'] = {
                    metric:paired_interval(pair, metric, 'comparison') for metric in
                    (('answer_score','evidence_recall') if dataset=='qasper' else ('answer_score',))}
    same_context = {}
    for a,b in combinations(METHODS, 2):
        for dataset in ('qasper','quality'):
            pairs = [(keyed[c['id'],a],keyed[c['id'],b]) for c in cases if c['dataset']==dataset]
            matching = [(x,y) for x,y in pairs if x['request_sha256']==y['request_sha256']]
            changes = sum(x['answer']!=y['answer'] or x['answer_score']!=y['answer_score'] for x,y in matching)
            if changes:
                raise ValueError('Identical-context outputs were not reused')
            same_context[f'{a}/{b}/{dataset}'] = {'same_context':len(matching), 'changed_answer_or_score':changes}
    audits = [json.loads(line) for line in (output / 'codex-audit.jsonl').read_text(encoding='utf-8').splitlines()]
    tokens = Counter()
    for row in audits:
        tokens.update({k:v for k,v in row.get('usage',{}).items() if type(v) is int})
    usage = {'attempts':len(audits), 'successful_attempts':sum(r['status']=='ok' for r in audits),
             'failed_attempts':sum(r['status']!='ok' for r in audits),
             'attempts_without_recorded_usage':sum('usage' not in r for r in audits),
             'recorded_tokens':dict(tokens), 'new_gemini_calls':0, 'new_jev_calls':0,
             'limitation':'Structurally rejected responses in the frozen client lack recorded usage; attempts remain reserved.'}
    differences = []
    for c in cases:
        baseline = keyed[c['id'],'codex_pool_8192']
        for method in METHODS:
            if method == 'codex_pool_8192':
                continue
            row = keyed[c['id'],method]
            delta = row['answer_score']-baseline['answer_score']
            if abs(delta) > 1e-12:
                differences.append({'id':c['id'], 'doc_id':c['doc_id'], 'dataset':c['dataset'],
                    'method':method, 'answer_delta_vs_codex':delta,
                    'evidence_recall':row['evidence_recall'], 'baseline_evidence_recall':baseline['evidence_recall'],
                    'answer':row['answer'], 'baseline_answer':baseline['answer']})
    differences.sort(key=lambda r:(r['answer_delta_vs_codex'],r['id'],r['method']))
    analysis = {'status':'complete_exposed_development_not_confirmatory',
                'method_predictions':len(rows), 'unique_reader_requests':len(jobs),
                'failed_predictions':sum(r['status']!='ok' for r in rows),
                'reader_replicates':manifest['replicates'],
                'interval_scope':'Descriptive paired document bootstrap, 10,000 replicates; no multiplicity correction.',
                'identical_context_audit':same_context,
                'source_hashes':{name:sha(ROOT/name) for name in ('scripts/analyze_isolated_reader.py','scripts/bounded_eval.py')},
                'scores_sha256':sha(output/'scores.json')}
    save(output/'paired-intervals.json', intervals)
    save(output/'analysis.json', analysis)
    save(output/'usage.json', usage)
    save(output/'failure-comparisons.json', differences)
    print({'predictions':len(rows),'unique_requests':len(jobs),'failed_predictions':analysis['failed_predictions'],
           'comparisons':len(intervals),'same_context_score_changes':0,'usage':usage})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/isolated-reader-v1')
    args = parser.parse_args()
    analyze(args.output)
