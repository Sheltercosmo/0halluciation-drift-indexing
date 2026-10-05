"""Paired component effects and source-based failure analysis, after inference."""
from collections import Counter
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import paired_interval, read_json, covered_paragraphs
from scripts.live_tree_eval import OUT, OLD, METHODS, TREES, verify


def analyze(output):
    manifest = verify(output)
    cases, scores = read_json(output / 'cases.json'), read_json(output / 'scores.json')
    expected = {(c['id'], m) for c in cases for m in METHODS}
    if Counter((r['id'], r['method']) for r in scores) != Counter({k: 1 for k in expected}):
        raise ValueError('Analysis requires all registered answers')
    pairs = []
    failed_ids = {r['id'] for r in scores if r['status'] != 'ok'}
    for axis, factor in enumerate(['split', 'representative', 'router']):
        for other in itertools.product('EJ', repeat=2):
            values = list(other); values.insert(axis, 'E'); a = ''.join(values)
            values[axis] = 'J'; b = ''.join(values)
            pairs.append((factor, b, a))
    for b in ['EEE', 'dense_recursive', 'rrf_recursive', 'rerank_recursive', 'dense_semantic', 'rrf_semantic', 'rerank_semantic']:
        pairs.append(('system', 'JJJ', b))
    for method in ['hybrid_matched', 'hybrid_full']:
        for b in ['JJJ', 'dense_recursive', 'rerank_recursive', 'rerank_semantic']:
            pairs.append(('hybrid', method, b))
    intervals = []
    for factor, candidate, baseline in pairs:
        for dataset in ['qasper', 'quality']:
            rows = [{**r, 'method': 'jev_blocking_rerank' if r['method'] == candidate else 'comparison'}
                    for r in scores if r['method'] in (candidate, baseline) and r['dataset'] == dataset]
            intervals.append({'factor': factor, 'candidate': candidate, 'baseline': baseline, 'dataset': dataset,
                              'metrics': {k: paired_interval(rows, k, 'comparison') for k in
                                          (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])},
                              'common_unblocked_metrics': {k: paired_interval([r for r in rows if r['id'] not in failed_ids], k, 'comparison') for k in
                                          (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])}})
    save(output / 'paired-intervals.json', intervals)
    keyed = {(r['id'], r['method']): r for r in scores}
    differences, same_context, routing, central = [], Counter(), [], Counter()
    docs, gold = read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    for doc_id in sorted(docs):
        for partition in 'EJ':
            trees = [read_json(output / 'indexes' / (signature(doc_id) + '-' + partition + rep + '.json')) for rep in 'EJ']
            def walk(n):
                yield n
                for child in n['children']:
                    yield from walk(child)
            lookup = [{n['node_id']: n for n in walk(t['root'])} for t in trees]
            assert lookup[0].keys() == lookup[1].keys()
            for node_id, a in lookup[0].items():
                b = lookup[1][node_id]
                assert (a['start'], a['end'], a['kind']) == (b['start'], b['end'], b['kind'])
                if a['kind'] not in ('section', 'paragraph') or not a['central'] or not b['central']:
                    continue
                prefix = partition + '/' + a['kind']
                central[prefix + '/nodes'] += 1
                central[prefix + '/same_sentence'] += int(a['central']['start'] == b['central']['start'])
                assert a['central']['candidate_indices'] == b['central']['candidate_indices']
                central[prefix + '/matched_candidates'] += len(a['central']['candidate_indices'])
    for case in cases:
        result = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))
        for a, b in itertools.combinations(METHODS, 2):
            if result['methods'][a]['context'] == result['methods'][b]['context']:
                same_context['identical_context_pairs'] += 1
                if keyed[case['id'], a]['answer'] != keyed[case['id'], b]['answer']:
                    raise ValueError('Identical input was resampled')
        baseline = keyed[case['id'], 'rerank_recursive']
        for method in ['JJJ', 'hybrid_full', 'hybrid_matched']:
            row = keyed[case['id'], method]
            if row['answer_score'] != baseline['answer_score']:
                differences.append({'id': case['id'], 'dataset': case['dataset'], 'method': method,
                                    'question': case['question'], 'answer': row['answer'], 'baseline_answer': baseline['answer'],
                                    'delta': row['answer_score'] - baseline['answer_score'],
                                    'evidence_recall': row['evidence_recall'], 'baseline_evidence_recall': baseline['evidence_recall'],
                                    'routing_status': row['routing_status']})
        for method in TREES:
            search = result['searches'][method]
            row = {'id': case['id'], 'dataset': case['dataset'], 'method': method, 'status': search['status'],
                   'reached_leaves': len(search['leaves']), 'node_scores': search['node_scores'],
                   'preview_tokens': search['preview_tokens'], 'final_context_tokens': result['methods'][method]['context_tokens']}
            if case['dataset'] == 'qasper':
                refs = [set(a['evidence']) for a in gold[case['id']]['answers'] if a['evidence']]
                if refs:
                    index = read_json(output / 'indexes' / (signature(case['doc_id']) + '-' + method[:2] + '.json'))
                    nodes = {n['node_id']: n for n in walk(index['root'])}
                    stage_spans = {}
                    for step in search['trace']:
                        stage_spans.setdefault(step['depth'], []).extend((nodes[i]['start'], nodes[i]['end']) for i in step['selected_ids'])
                    retained = {}
                    for depth, spans in stage_spans.items():
                        # Sentence choices map back to their source paragraphs, as in the reader's bottom-up step.
                        if depth == 4:
                            spans = [(u['start'], u['end']) for u in docs[case['doc_id']]['units']
                                     if any(u['start'] <= a < b <= u['end'] for a, b in spans)]
                        evidence = set(covered_paragraphs(docs[case['doc_id']], spans))
                        retained[depth] = max(len(evidence & ref) / len(ref) for ref in refs)
                    row['annotated_evidence_retained_by_depth'] = retained
            routing.append(row)
    differences.sort(key=lambda r: (r['delta'], r['id'], r['method']))
    save(output / 'failure-comparisons.json', differences)
    save(output / 'routing-analysis.json', routing)
    audits, usage = {}, {}
    for provider in ['gemini', 'embedding', 'jev']:
        path = output / (provider + '-audit.jsonl')
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        tokens = Counter()
        for row in rows:
            tokens.update({k: v for k, v in row.get('usage', {}).items() if type(v) is int})
            if provider == 'jev':
                tokens.update({k: row[k] for k in ['input_tokens', 'output_tokens'] if type(row.get(k)) is int})
        usage[provider] = {'attempts': len(rows), 'failed_attempts': sum(r['status'] != 'ok' for r in rows),
                           'stages': dict(Counter(r.get('stage') for r in rows)), 'recorded_usage': dict(tokens),
                           'summed_request_seconds': sum(r['seconds'] for r in rows)}
        if provider == 'jev':
            usage[provider]['stage_label_note'] = "The inherited audit label 'reranking' covers representative selection and hierarchical routing; it is not a flat-reranking count."
    save(output / 'usage.json', usage)
    normalizations = [read_json(p) for p in sorted((output / 'ranker-normalizations').glob('*.json'))]
    save(output / 'ranker-normalizations.json', normalizations)
    identity_repairs = [read_json(p) for p in sorted((output / 'reader-identity-normalizations').glob('*.json'))]
    save(output / 'reader-identity-normalizations.json', identity_repairs)
    save(output / 'common-unblocked-summary.json', {dataset: {method: {
        'n': sum(r['dataset'] == dataset and r['method'] == method and r['id'] not in failed_ids for r in scores),
        'answer_score': statistics.mean(r['answer_score'] for r in scores if r['dataset'] == dataset and r['method'] == method and r['id'] not in failed_ids)}
        for method in METHODS} for dataset in ['qasper', 'quality']})
    save(output / 'analysis.json', {'status': 'complete_development_only', 'predictions': len(scores),
        'central_sentence_agreement': dict(central), 'identical_context_audit': dict(same_context),
        'normalized_ranker_responses': len(normalizations),
        'reader_identity_repairs': len(identity_repairs),
        'failure_statuses': dict(Counter(r['status'] for r in scores if r['status'] != 'ok')),
        'questions_with_any_reader_failure': sorted(failed_ids),
        'failed_predictions': sum(r['status'] != 'ok' for r in scores), 'registered_methods': METHODS,
        'intervals': 'Descriptive paired document bootstrap, 10,000 replicates; no confirmatory multiplicity-adjusted claim',
        'unopened_validation_questions': 1425, 'unopened_test_questions': 1559})
    print(json.dumps({'predictions': len(scores), 'contrasts': len(intervals), 'usage': usage}))


if __name__ == '__main__':
    analyze(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
