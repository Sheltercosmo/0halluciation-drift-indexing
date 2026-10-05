"""Two registered development trials using saved Jev decisions, with no reindexing."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import Codex, save, signature, validate_answers
from scripts.bounded_eval import batches, context_for, covered_paragraphs, parallel_progress, query_text, read_json, sha, tokenizer
from scripts.development_iteration import OLD, research_budget
from zero_index.context_merge import merge_context, topic_parents

BASE = ROOT / 'output/development-pool-v1'
METHODS = ('jev_fusion_025', 'jev_topic_merge')


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Trial already frozen')
    cases, docs, enc = read_json(BASE / 'cases.json'), read_json(OLD / 'documents.json'), tokenizer()
    save(output / 'cases.json', cases)
    inputs = {'cases.json': sha(output / 'cases.json')}
    for case in cases:
        doc = docs[case['doc_id']]
        row = read_json(BASE / 'retrieval' / (signature(case['id']) + '.json'))
        index = read_json(OLD / 'indexes' / (signature(case['doc_id']) + '.json'))
        pool = row['pools']['jev']['candidates']
        lookup = {c['id']: c for c in pool}
        initial = {c['id']: i+1 for i,c in enumerate(pool)}
        decision = {k: i+1 for i,k in enumerate(row['methods']['jev_pool_8192']['ranking'])}
        fused = sorted(pool, key=lambda c: (-.25/(60+initial[c['id']])-.75/(60+decision[c['id']]), initial[c['id']]))
        context, spans, count = context_for(doc, fused, enc)
        merged = merge_context(doc, [lookup[k] for k in decision], topic_parents(doc, index['trace']),
                               lambda text: len(enc.encode(text, disallowed_special=())))
        path = 'retrieval/' + signature(case['id']) + '.json'
        save(output / path, {'id': case['id'], 'methods': {
            'jev_fusion_025': {'context': context, 'spans': spans, 'context_tokens': count},
            'jev_topic_merge': merged}})
        inputs[path] = sha(output / path)
    source = ['scripts/context_trial.py', 'scripts/bounded_clients.py', 'scripts/bounded_eval.py',
              'zero_index/context_merge.py']
    save(output / 'manifest.json', {'status': 'registered_before_reading', 'partition': 'exposed_development',
        'methods': METHODS, 'questions': len(cases), 'context_tokens': 2048, 'new_indexing_calls': 0,
        'configurations': {'jev_fusion_025': {'hybrid_rank_weight': .25, 'decision_rank_weight': .75, 'rrf_constant': 60},
                           'jev_topic_merge': {'coverage': '(1+context_tokens/budget)/3', 'minimum_selected_children': 2}},
        'selection_disclosure': 'Fusion weight .25 preserved development QASPER evidence recall; .5 and .75 reduced it. Topic merge did not improve QASPER recall; test narrative context recovery.',
        'data_hashes': inputs, 'source_hashes': {p: sha(ROOT / p) for p in source}})
    print(json.dumps({'methods': METHODS, 'questions': len(cases)}))


def verify(output):
    manifest = read_json(output / 'manifest.json')
    for name, expected in manifest['data_hashes'].items():
        if sha(output / name) != expected:
            raise ValueError('Changed prepared input: ' + name)
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed trial source: ' + name)
    return manifest


def run(output):
    verify(output)
    cases = read_json(output / 'cases.json')
    client = Codex(output, research_budget())
    client.cache = BASE / 'codex-cache'
    def read(job):
        method, batch = job
        folder = output / 'predictions' / method
        if all((folder / (signature(c['id']) + '.json')).exists() for c in batch):
            return
        request = [{'id': c['id'], 'question': query_text(c), 'context': read_json(output / 'retrieval' /
                    (signature(c['id']) + '.json'))['methods'][method]['context']} for c in batch]
        answers = validate_answers(client.call(request, 'reader'), [c['id'] for c in batch])
        for case in batch:
            answer = answers[case['id']].strip()
            valid = bool(answer) and (not case['options'] or answer in ('A', 'B', 'C', 'D'))
            save(folder / (signature(case['id']) + '.json'), {'id': case['id'], 'method': method, 'answer': answer,
                'status': 'ok' if valid else 'failed', 'request_sha256': signature(request)})
    parallel_progress(read, [(m,b) for b in batches(cases) for m in METHODS], 2, 'context-trial')


def score(output):
    verify(output)
    cases = read_json(output / 'cases.json')
    docs, gold = read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    spec = importlib.util.spec_from_file_location('qasper_context_trial', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'p': {'qas': [{'question_id': c['id'],
        'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases if c['dataset'] == 'qasper']}}, False)
    rows = []
    for case in cases:
        contexts = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))['methods']
        for method in METHODS:
            prediction = read_json(output / 'predictions' / method / (signature(case['id']) + '.json'))
            context = contexts[method]
            row = {**prediction, 'doc_id': case['doc_id'], 'dataset': case['dataset'],
                'context_tokens': context['context_tokens'], 'spans': context['spans'],
                'evidence_recall': None, 'complete_evidence': None, 'retrieved_evidence_f1': None}
            if case['dataset'] == 'quality':
                row['answer_score'] = int(prediction['status'] == 'ok' and prediction['answer'] == gold[case['id']]['label'])
            else:
                ref = refs[case['id']]
                row['answer_score'] = max(official.token_f1_score(prediction['answer'], a['answer']) for a in ref) if prediction['status'] == 'ok' else 0
                evidence = covered_paragraphs(docs[case['doc_id']], context['spans'])
                row['retrieved_evidence_f1'] = max(official.paragraph_f1_score(evidence, a['evidence']) for a in ref)
                eligible = [a for a in ref if a['evidence']]
                if eligible:
                    row['evidence_recall'] = max(len(set(evidence)&set(a['evidence']))/len(set(a['evidence'])) for a in eligible)
                    row['complete_evidence'] = int(any(set(a['evidence']) <= set(evidence) for a in eligible))
            rows.append(row)
    summary = {'status': 'development_only_not_confirmatory', 'questions': len(cases), 'methods': {}}
    for method in METHODS:
        summary['methods'][method] = {}
        for dataset in ('qasper', 'quality'):
            group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
            metrics = {}
            for name in ('answer_score', 'evidence_recall', 'complete_evidence', 'retrieved_evidence_f1', 'context_tokens'):
                values = [r[name] for r in group if r[name] is not None]
                metrics[name] = statistics.mean(values) if values else None
            summary['methods'][method][dataset] = {'questions': len(group), **metrics}
    save(output / 'scores.json', rows)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','score'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output/development-context-v1')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.command](args.output)
