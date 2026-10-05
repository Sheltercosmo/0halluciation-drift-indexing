"""Diagnose the now-exposed bounded run using its saved retrieval, without APIs.

This is development analysis. It must never be used to inspect a locked test set.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_eval import covered_paragraphs, read_json, signature, tokenizer
from scripts.bounded_clients import save


def evidence_recall(doc, spans, references):
    eligible = [set(a['evidence']) for a in references if a['evidence']]
    if not eligible:
        return None
    actual = set(covered_paragraphs(doc, spans))
    return max(len(actual & ref) / len(ref) for ref in eligible)


def analyze(run):
    cases, docs, gold, scores = [read_json(run / name) for name in
                               ('cases.json', 'documents.json', 'gold.json', 'scores.json')]
    enc = tokenizer()
    by_score = {(r['id'], r['method']): r for r in scores}
    rows, disagreements = [], []
    for case in cases:
        doc = docs[case['doc_id']]
        retrieval = read_json(run / 'retrieval' / (signature(case['id']) + '.json'))
        index = read_json(run / 'indexes' / (signature(case['doc_id']) + '.json'))
        jev = by_score[case['id'], 'jev_blocking_rerank']
        baseline = by_score[case['id'], 'hybrid_codex_rerank']
        if jev['answer_score'] < baseline['answer_score']:
            disagreements.append({'id': case['id'], 'doc_id': case['doc_id'], 'dataset': case['dataset'],
                'question': case['question'], 'options': case['options'],
                'jev_answer': jev['answer'], 'baseline_answer': baseline['answer'],
                'jev_answer_score': jev['answer_score'], 'baseline_answer_score': baseline['answer_score'],
                'jev_evidence_recall': jev['evidence_recall'], 'baseline_evidence_recall': baseline['evidence_recall']})
        for method, chunking in [('jev_blocking_rerank', 'jev'), ('hybrid_codex_rerank', 'recursive')]:
            context = retrieval['methods'][method]
            chunk_map = {c['id']: c for c in index[chunking]}
            pool = [chunk_map[key] for key in context['ranking']]
            row = {'id': case['id'], 'doc_id': case['doc_id'], 'dataset': case['dataset'], 'method': method,
                'candidate_count': len(pool), 'candidate_source_tokens': sum(len(enc.encode(c['text'], disallowed_special=())) for c in pool),
                'context_tokens': context['context_tokens'], 'selected_chunks': len(context['spans']),
                'answer_score': by_score[case['id'], method]['answer_score'],
                'source_recall_ceiling': None, 'candidate_recall': None, 'context_recall': None,
                'evidence_failure_stage': None}
            if case['dataset'] == 'qasper':
                refs = gold[case['id']]['answers']
                row['source_recall_ceiling'] = evidence_recall(doc, [(0, len(doc['text']))], refs)
                row['candidate_recall'] = evidence_recall(doc, [(c['start'], c['end']) for c in pool], refs)
                row['context_recall'] = evidence_recall(doc, context['spans'], refs)
                if row['context_recall'] is not None:
                    row['evidence_failure_stage'] = (
                        'source_not_textually_represented' if row['source_recall_ceiling'] < 1 else
                        'candidate_pool_incomplete' if row['candidate_recall'] < 1 else
                        'ranking_or_packing_lost_evidence' if row['context_recall'] < 1 else
                        'full_reference_recovered')
            rows.append(row)
    summary = {'status': 'development_diagnostic_only', 'source_run': 'bounded-v1', 'model_calls': 0,
               'questions': len(cases), 'methods': {}}
    for method in ('jev_blocking_rerank', 'hybrid_codex_rerank'):
        summary['methods'][method] = {}
        for dataset in ('qasper', 'quality'):
            selected = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
            evidence = [r for r in selected if r['context_recall'] is not None]
            summary['methods'][method][dataset] = {
                'questions': len(selected), 'evidence_questions': len(evidence),
                **{name: statistics.mean(r[name] for r in selected) for name in
                   ('candidate_source_tokens', 'context_tokens', 'selected_chunks', 'answer_score')},
                **{name: statistics.mean(r[name] for r in evidence) if evidence else None for name in
                   ('source_recall_ceiling', 'candidate_recall', 'context_recall')},
                'contexts_below_1536_tokens': sum(r['context_tokens'] < 1536 for r in selected),
                'failure_stage_counts': dict(Counter(r['evidence_failure_stage'] for r in evidence)),
                'answer_score_when_full_evidence': statistics.mean(r['answer_score'] for r in evidence if r['context_recall'] == 1) if evidence else None,
                'answer_score_when_incomplete_evidence': statistics.mean(r['answer_score'] for r in evidence if r['context_recall'] < 1) if evidence else None}
    disagreements.sort(key=lambda r: (r['dataset'], r['jev_answer_score']-r['baseline_answer_score'], r['id']))
    return summary, rows, disagreements


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'output/bounded-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'evals/results/development-v1')
    args = parser.parse_args()
    # Only this already-public, exposed run is authorized for this analysis.
    if read_json(args.run / 'manifest.json')['data_hashes'] != read_json(ROOT / 'evals/results/bounded-v1/manifest.json')['data_hashes']:
        raise ValueError('This script is restricted to the exposed bounded-v1 development run')
    summary, rows, failures = analyze(args.run)
    save(args.output / 'diagnostic-summary.json', summary)
    save(args.output / 'diagnostic-rows.json', rows)
    save(args.output / 'disagreements.json', failures)
    print(json.dumps(summary, indent=2))
