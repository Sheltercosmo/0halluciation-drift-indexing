"""Recompute published tree-study scores using official data, without API calls."""
from collections import Counter
import hashlib
import importlib.util
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_eval import covered_paragraphs, paired_interval, read_json, sha, tokenizer


def check(actual, expected):
    if actual is None or expected is None:
        assert actual == expected
    else:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def replay(result, data):
    manifest, provenance = read_json(result / 'manifest.json'), read_json(result / 'provenance.json')
    assert sha(result / 'manifest.json') == provenance['manifest_sha256']
    for name, digest in provenance['official_input_hashes'].items():
        assert sha(data / name) == digest, name
    for name, digest in manifest['source_hashes'].items():
        assert sha(result / 'source' / name) == digest, name
    cases = {c['id']: c for c in read_json(data / 'cases.json')}
    docs, gold = read_json(data / 'documents.json'), read_json(data / 'gold.json')
    spec = importlib.util.spec_from_file_location('official_replay', data / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'paper': {'qas': [
        {'question_id': c['id'], 'answers': [{'answer': a} for a in gold[c['id']]['answers']]}
        for c in cases.values() if c['dataset'] == 'qasper']}}, False)
    rows, summary = read_json(result / 'scores.json'), read_json(result / 'summary.json')
    expected = {(c, m) for c in cases for m in manifest['methods']}
    assert Counter((r['id'], r['method']) for r in rows) == Counter({k: 1 for k in expected})
    enc = tokenizer()
    for row in rows:
        case = cases[row['id']]; doc = docs[case['doc_id']]
        assert row['doc_id'] == case['doc_id'] and row['dataset'] == case['dataset'] and row['status'] in ('ok', 'failed', 'blocked_by_reader')
        spans = row['spans']; previous = -1
        for a, b in spans:
            assert type(a) is int and type(b) is int and previous < a < b <= len(doc['text'])
            previous = b
        context = ('Title: ' + doc['title'] + '\n' + '\n\n'.join(
            f'[{i+1}]\n{doc["text"][a:b]}' for i, (a, b) in enumerate(spans))) if spans else ''
        assert hashlib.sha256(context.encode()).hexdigest() == row['context_sha256']
        tokens = len(enc.encode(context, disallowed_special=()))
        assert tokens == row['context_tokens'] and tokens <= 2048
        if case['dataset'] == 'quality':
            answer = int(row['status'] == 'ok' and row['answer'] == gold[case['id']]['label'])
            recall = complete = None
        else:
            answer = max(official.token_f1_score(row['answer'], r['answer']) for r in refs[case['id']]) if row['status'] == 'ok' else 0
            evidence = set(covered_paragraphs(doc, spans))
            eligible = [set(r['evidence']) for r in refs[case['id']] if r['evidence']]
            recall = max(len(evidence & ref) / len(ref) for ref in eligible) if eligible else None
            complete = int(any(ref <= evidence for ref in eligible)) if eligible else None
        for key, value in [('answer_score', answer), ('evidence_recall', recall), ('complete_evidence', complete)]:
            check(row[key], value)
    for dataset, methods in summary['methods'].items():
        for method, metrics in methods.items():
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            assert len(subset) == metrics['n']
            for key in ['answer_score', 'evidence_recall', 'complete_evidence', 'context_tokens']:
                values = [r[key] for r in subset if r[key] is not None]
                check(statistics.mean(values) if values else None, metrics[key])
    intervals = read_json(result / 'paired-intervals.json')
    failed_ids = {r['id'] for r in rows if r['status'] != 'ok'}
    for contrast in intervals:
        pair = [{**r, 'method': 'jev_blocking_rerank' if r['method'] == contrast['candidate'] else 'comparison'}
                for r in rows if r['method'] in (contrast['candidate'], contrast['baseline']) and r['dataset'] == contrast['dataset']]
        for metric, recorded in contrast['metrics'].items():
            actual = paired_interval(pair, metric, 'comparison')
            for key in ['difference', 'documents', 'questions', 'replicates']:
                check(actual[key], recorded[key])
            for a, b in zip(actual['ci95'], recorded['ci95']): check(a, b)
        for metric, recorded in contrast.get('common_unblocked_metrics', {}).items():
            actual = paired_interval([r for r in pair if r['id'] not in failed_ids], metric, 'comparison')
            for key in ['difference', 'documents', 'questions', 'replicates']: check(actual[key], recorded[key])
            for a, b in zip(actual['ci95'], recorded['ci95']): check(a, b)
    print({'predictions': len(rows), 'contrasts': len(intervals), 'source_contexts_and_official_metrics': 'passed', 'model_calls': 0})


if __name__ == '__main__':
    replay(ROOT / 'evals/results/live-tree-v1', ROOT / 'output/bounded-v1')
