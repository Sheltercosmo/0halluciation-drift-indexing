"""Export and independently replay the completed factorial and bottom-up studies."""
import argparse
import hashlib
import importlib.util
from collections import Counter
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save
from scripts.bounded_eval import read_json, sha, paired_interval
from scripts.bounded_eval import covered_paragraphs
from scripts.bounded_clients import signature
from scripts.live_tree_eval import OUT as BASE, OLD
from scripts.live_tree_reader_v2 import OUT as READER, verify as verify_reader
from scripts.live_tree_followup import OUT as FOLLOWUP, verify as verify_followup
from scripts.live_tree_ancestor import OUT as ANCESTOR, METHODS as ANCESTOR_METHODS, verify as verify_ancestor
from scripts.replay_live_tree import replay

DEST = ROOT / 'evals/results/live-tree-complete-v2'


def export():
    verify_reader(); verify_followup(); verify_ancestor()
    rows = read_json(READER / 'scores.json') + read_json(FOLLOWUP / 'scores.json')
    methods = read_json(BASE / 'manifest.json')['methods'] + read_json(FOLLOWUP / 'manifest.json')['methods'] + ANCESTOR_METHODS
    cases = read_json(BASE / 'cases.json')
    docs, gold = read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    spec = importlib.util.spec_from_file_location('ancestor_official', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'paper': {'qas': [{'question_id': c['id'], 'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases if c['dataset'] == 'qasper']}}, False)
    for case in cases:
        name = signature(case['id']) + '.json'; context = read_json(ANCESTOR / 'retrieval' / name)
        for method in ANCESTOR_METHODS:
            row = read_json(ANCESTOR / 'predictions' / method / name); retrieved = context['methods'][method]
            row.update({'doc_id': case['doc_id'], 'dataset': case['dataset'], 'spans': retrieved['spans'],
                'context_tokens': retrieved['context_tokens'], 'context_sha256': hashlib.sha256(retrieved['context'].encode()).hexdigest(),
                'evidence_recall': None, 'complete_evidence': None})
            if case['dataset'] == 'quality': row['answer_score'] = int(row['status'] == 'ok' and row['answer'] == gold[row['id']]['label'])
            else:
                row['answer_score'] = max(official.token_f1_score(row['answer'], r['answer']) for r in refs[row['id']]) if row['status'] == 'ok' else 0
                eligible = [set(r['evidence']) for r in refs[row['id']] if r['evidence']]
                if eligible:
                    evidence = set(covered_paragraphs(docs[case['doc_id']], row['spans']))
                    row['evidence_recall'] = max(len(evidence & e)/len(e) for e in eligible)
                    row['complete_evidence'] = int(any(e <= evidence for e in eligible))
            rows.append(row)
    save(ANCESTOR / 'scores.json', [r for r in rows if r['method'] in ANCESTOR_METHODS])
    assert Counter((r['id'], r['method']) for r in rows) == Counter({(c['id'], m): 1 for c in cases for m in methods})
    failed = {r['id'] for r in rows if r['status'] != 'ok'}
    summary = {'questions': len(cases), 'predictions': len(rows), 'documents': 307, 'methods': {},
               'questions_with_any_reader_failure': sorted(failed), 'budget': read_json(ROOT / 'output/research-budget.json')}
    for dataset in ['qasper', 'quality']:
        summary['methods'][dataset] = {}
        for method in methods:
            part = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            summary['methods'][dataset][method] = {'n': len(part),
                **{k: statistics.mean([r[k] for r in part if r[k] is not None]) if any(r[k] is not None for r in part) else None for k in ['answer_score', 'evidence_recall', 'complete_evidence', 'context_tokens']},
                'reader_failures': sum(r['status'] != 'ok' for r in part),
                'empty_contexts': sum(r['context_tokens'] == 0 for r in part),
                'common_unblocked_n': sum(r['id'] not in failed for r in part),
                'common_unblocked_answer': statistics.mean(r['answer_score'] for r in part if r['id'] not in failed)}
    contrasts = read_json(READER / 'paired-intervals.json') + read_json(FOLLOWUP / 'paired-intervals.json')
    extra_pairs = [('EEE_ancestor', 'EEE_bottom_up'), ('JJJ_ancestor', 'JJJ_bottom_up'), ('JJJ_ancestor', 'JJJ'), ('JJJ_ancestor', 'EEE_ancestor'), ('JJJ_ancestor', 'hybrid_full')]
    extra_pairs += [('JJJ_ancestor', m) for m in ['dense_recursive', 'rrf_recursive', 'rerank_recursive', 'dense_semantic', 'rrf_semantic', 'rerank_semantic']]
    extra_pairs += [('JJJ_bottom_up', 'rrf_semantic')]
    for candidate, baseline in extra_pairs:
        for dataset in ['qasper', 'quality']:
            contrasts.append({'factor': 'ancestor-repair' if 'ancestor' in candidate else 'development-repair', 'candidate': candidate, 'baseline': baseline, 'dataset': dataset,
                'metrics': {k: None for k in (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])},
                'common_unblocked_metrics': {k: None for k in (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])}})
    for contrast in contrasts:
        pair = [{**r, 'method': 'jev_blocking_rerank' if r['method'] == contrast['candidate'] else 'comparison'} for r in rows if r['dataset'] == contrast['dataset'] and r['method'] in (contrast['candidate'], contrast['baseline'])]
        for key, subset in [('metrics', pair), ('common_unblocked_metrics', [r for r in pair if r['id'] not in failed])]:
            contrast[key] = {metric: paired_interval(subset, metric, 'comparison') for metric in contrast[key]}
    DEST.mkdir(parents=True, exist_ok=True)
    save(DEST / 'scores.json', rows); save(DEST / 'summary.json', summary); save(DEST / 'paired-intervals.json', contrasts)
    sources = {}
    for run in [BASE, READER, FOLLOWUP, ANCESTOR]:
        manifest = read_json(run / 'manifest.json')
        for name, digest in manifest['source_hashes'].items():
            source = run / 'source' / name
            assert sha(source) == digest, name
            if name in sources: assert sources[name] == digest, 'Conflicting frozen source: ' + name
            sources[name] = digest
            dest = DEST / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(source.read_bytes())
        dest = DEST / 'registrations' / (run.name + '.json'); dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((run / 'manifest.json').read_bytes())
    manifest = {'status': 'completed_development_comparison', 'partition': 'exposed_development_only', 'methods': methods,
        'questions': 384, 'predictions': len(rows), 'source_hashes': sources,
        'registrations': {run.name: sha(run / 'manifest.json') for run in [BASE, READER, FOLLOWUP, ANCESTOR]},
        'primary_policy': 'All sixteen registered retrieval arms; original QASPER answers unchanged, every QuALITY arm uses the uniformly repaired labelled-option reader.',
        'followup_policy': 'Both frozen bottom-up repairs included without selection. Two potential routing attempts have extra compute; final context remains 2048 tokens.',
        'ancestor_policy': 'Both second-iteration arms use the exact follow-up routes, expanding source paragraphs through successive ancestors. No additional routing calls.',
        'original_reader_archive': '../live-tree-v1/scores.json',
        'selection': 'Development-only diagnosis and iteration; no validation/test data used; no default promotion.'}
    save(DEST / 'manifest.json', manifest)
    original_provenance = read_json(ROOT / 'evals/results/live-tree-v1/provenance.json')
    save(DEST / 'provenance.json', {**original_provenance, 'manifest_sha256': sha(DEST / 'manifest.json'),
        'runtime_source_hashes': sources, 'parent_provenance_sha256': sha(ROOT / 'evals/results/live-tree-v1/provenance.json'),
        'followup_retrieval_sha256': {p.name: sha(p) for p in (FOLLOWUP / 'retrieval').glob('*.json')},
        'ancestor_retrieval_sha256': {p.name: sha(p) for p in (ANCESTOR / 'retrieval').glob('*.json')},
        'additional_audit_sha256': {run.name + '/' + p.name: sha(p) for run in [READER, FOLLOWUP, ANCESTOR] for p in run.glob('*-audit.jsonl')}})
    for name in ['cases.json', 'query-plans.json', 'routing-analysis.json', 'partition-reconstruction-audit.json', 'ranker-normalizations.json', 'reader-identity-normalizations.json', 'runtime.json']:
        (DEST / name).write_bytes((ROOT / 'evals/results/live-tree-v1' / name).read_bytes())
    usage = {}
    for run in [BASE, READER, FOLLOWUP, ANCESTOR]:
        usage[run.name] = {}
        for p in run.glob('*-audit.jsonl'):
            records = [__import__('json').loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line]
            native_keys = sorted({k for r in records for k, v in r.get('usage', {}).items() if type(v) is int})
            usage[run.name][p.name] = {'attempts': len(records), 'failed_attempts': sum(r['status'] != 'ok' for r in records),
                'stages': dict(Counter(r['stage'] for r in records)),
                'request_seconds': sum(r.get('seconds', 0) for r in records),
                'provider_usage': {k: sum(r.get('usage', {}).get(k, 0) for r in records) for k in native_keys},
                'root_usage_fields': {k: sum(r.get(k, 0) for r in records) for k in ['input_tokens', 'output_tokens', 'questions', 'inputs'] if any(k in r for r in records)}}
    save(DEST / 'usage.json', usage)
    original = read_json(BASE / 'scores.json')
    save(DEST / 'reader-contract-audit.json', {
        'original_quality_non_letter_answers': dict(Counter(r['method'] for r in original if r['dataset'] == 'quality' and r['status'] == 'ok' and r['answer'] not in ('A', 'B', 'C', 'D'))),
        'corrected_quality_non_letter_answers': sum(r['dataset'] == 'quality' and r['status'] == 'ok' and r['answer'] not in ('A', 'B', 'C', 'D') for r in rows),
        'qaspar_answers_unchanged': all(next(s for s in original if s['id'] == r['id'] and s['method'] == r['method'])['answer'] == r['answer'] for r in rows if r['dataset'] == 'qasper' and r['method'] in methods[:16]),
        'blocked_questions': sorted(failed)})
    keyed = {(r['id'], r['method']): r for r in rows}; failures = []
    for case in cases:
        baseline = keyed[case['id'], 'rerank_recursive']
        for method in ['JJJ', 'JJJ_bottom_up', 'JJJ_ancestor', 'hybrid_full']:
            row = keyed[case['id'], method]
            failures.append({'id': case['id'], 'doc_id': case['doc_id'], 'dataset': case['dataset'], 'method': method,
                'answer_delta_vs_recursive_reranker': row['answer_score'] - baseline['answer_score'],
                'evidence_delta': row['evidence_recall'] - baseline['evidence_recall'] if row['evidence_recall'] is not None else None,
                'context_tokens': row['context_tokens'], 'routing_status': row.get('routing_status'),
                'reader_status': row['status']})
    save(DEST / 'failure-comparisons.json', failures)
    save(DEST / 'retrieval-component-audit.json', read_json(BASE / 'analysis.json'))
    print({'exported_records': len(rows), 'destination': str(DEST)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['export', 'replay']); args = parser.parse_args()
    if args.command == 'export': export()
    else: replay(DEST, OLD)
