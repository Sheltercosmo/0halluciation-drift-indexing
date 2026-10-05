"""Verify public development predictions against frozen source data, without APIs."""
import argparse
from collections import Counter
import importlib.util
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_eval import covered_paragraphs, read_json, tokenizer


def replay(data, result):
    cases = {c['id']: c for c in read_json(data / 'cases.json')}
    docs, gold = read_json(data / 'documents.json'), read_json(data / 'gold.json')
    screen = set(read_json(ROOT / 'evals/iterations/v1/manifest.json')['development_screening_ids'])
    spec = importlib.util.spec_from_file_location('qasper_development_replay', data / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'p': {'qas': [{'question_id': c['id'],
        'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases.values() if c['dataset'] == 'qasper']}}, False)
    rows = read_json(result / 'scores.json')
    manifest = read_json(result / 'manifest.json')
    if result.name == 'closed-book-v1':
        expected_ids = {key for key in screen if cases[key]['dataset'] == 'quality'}
        if Counter(r['id'] for r in rows) != Counter({key: 1 for key in expected_ids}):
            raise ValueError('Closed-book coverage mismatch')
        for row in rows:
            expected = int(row['status'] == 'ok' and row['answer'] == gold[row['id']]['label'])
            if row['correct'] != expected:
                raise ValueError('Closed-book score mismatch')
        summary = read_json(result / 'summary.json')
        if summary['questions'] != len(rows) or summary['correct'] != sum(r['correct'] for r in rows):
            raise ValueError('Closed-book summary mismatch')
        if not math.isclose(summary['accuracy'], statistics.mean(r['correct'] for r in rows), abs_tol=1e-12):
            raise ValueError('Closed-book accuracy mismatch')
        return len(rows)
    methods = set(manifest['methods'])
    if result.name == 'pool-screen-v1':
        methods |= {'hybrid_recursive', 'hybrid_semantic', 'hybrid_codex_rerank', 'jev_blocking_rerank'}
    expected = {(key, method) for key in screen for method in methods}
    if Counter((r['id'], r['method']) for r in rows) != Counter({key: 1 for key in expected}):
        raise ValueError('Development prediction coverage mismatch')
    enc = tokenizer()
    def check(value, expected, metric):
        if value is None or expected is None:
            if value != expected:
                raise ValueError('Missing metric: ' + metric)
        elif not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Metric mismatch: ' + metric)
    for row in rows:
        case, doc = cases[row['id']], docs[cases[row['id']]['doc_id']]
        if row['doc_id'] != case['doc_id'] or row['dataset'] != case['dataset']:
            raise ValueError('Case metadata mismatch')
        chunks = []
        last_end = -1
        for start, end in sorted(row['spans']):
            if not 0 <= start < end <= len(doc['text']) or start < last_end:
                raise ValueError('Invalid or overlapping source span')
            last_end = end
            heading = next(u['heading'] for u in doc['units'] if u['start'] <= start < u['end'])
            chunks.append({'heading': heading, 'text': doc['text'][start:end]})
        context = 'Title: ' + doc['title'] + '\n' + '\n\n'.join(f"[{i+1}] {c['heading']}\n{c['text']}" for i,c in enumerate(chunks))
        tokens = len(enc.encode(context, disallowed_special=()))
        if tokens != row['context_tokens'] or tokens > 2048:
            raise ValueError('Rendered source token budget mismatch')
        if case['dataset'] == 'quality':
            answer = int(row['status'] == 'ok' and row['answer'] == gold[row['id']]['label'])
            recall = complete = evidence_f1 = None
        else:
            ref = refs[row['id']]
            answer = max(official.token_f1_score(row['answer'], r['answer']) for r in ref) if row['status'] == 'ok' else 0
            evidence_list = covered_paragraphs(doc, row['spans'])
            evidence = set(evidence_list)
            evidence_f1 = max(official.paragraph_f1_score(evidence_list, r['evidence']) for r in ref)
            eligible = [set(r['evidence']) for r in ref if r['evidence']]
            recall = max(len(evidence & r)/len(r) for r in eligible) if eligible else None
            complete = int(any(r <= evidence for r in eligible)) if eligible else None
        for metric, value in [('answer_score', answer), ('evidence_recall', recall),
                              ('complete_evidence', complete), ('retrieved_evidence_f1', evidence_f1)]:
            check(row[metric], value, metric)
    summary = read_json(result / 'summary.json')
    if summary['questions'] != len(screen) or set(summary['methods']) != methods:
        raise ValueError('Summary coverage mismatch')
    for method in methods:
        for dataset in ('qasper', 'quality'):
            group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
            expected = summary['methods'][method][dataset]
            if expected['questions'] != len(group):
                raise ValueError('Summary denominator mismatch')
            for metric in ('answer_score','evidence_recall','complete_evidence','retrieved_evidence_f1','context_tokens'):
                values = [r[metric] for r in group if r[metric] is not None]
                check(expected[metric], statistics.mean(values) if values else None, metric)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'output/bounded-v1')
    parser.add_argument('--results', nargs='+', type=Path, default=[ROOT / 'evals/results/pool-screen-v1', ROOT / 'evals/results/closed-book-v1'])
    args = parser.parse_args()
    for result in args.results:
        count = replay(args.data, result)
        print(f'PASS: {result.name}: {count} predictions, exact source spans/token budgets, official scores and summaries; no model calls.')
