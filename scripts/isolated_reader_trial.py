"""Remove cross-case reader batching from the exposed development comparison."""
import argparse
import importlib.util
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import Codex, save, signature, validate_answers
from scripts.bounded_eval import covered_paragraphs, parallel_progress, query_text, read_json, sha
from scripts.development_iteration import OLD, research_budget

BASE = ROOT / 'output/development-pool-v1'
CONTEXT = ROOT / 'output/development-context-v1'
METHODS = ('jev_blocking_rerank', 'jev_pool_8192', 'codex_pool_8192', 'jev_fusion_025', 'jev_topic_merge')


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Trial already frozen')
    cases = read_json(BASE / 'cases.json')
    jobs, contexts = {}, {}
    for case in cases:
        filename = signature(case['id']) + '.json'
        methods = {'jev_blocking_rerank': read_json(OLD / 'retrieval' / filename)['methods']['jev_blocking_rerank']}
        methods.update(read_json(BASE / 'retrieval' / filename)['methods'])
        methods.update(read_json(CONTEXT / 'retrieval' / filename)['methods'])
        contexts[case['id']] = {}
        for method in METHODS:
            context = methods[method]
            request = {'id': case['id'], 'question': query_text(case), 'context': context['context']}
            key = signature(request)
            jobs[key] = request
            contexts[case['id']][method] = {'request_sha256': key, 'spans': context['spans'],
                                          'context_tokens': context['context_tokens']}
    save(output / 'cases.json', cases)
    save(output / 'requests.json', jobs)
    save(output / 'contexts.json', contexts)
    paths = ('scripts/isolated_reader_trial.py', 'scripts/bounded_clients.py', 'scripts/bounded_eval.py')
    save(output / 'manifest.json', {'status': 'registered_before_inference', 'partition': 'exposed_development',
        'methods': METHODS, 'questions': len(cases), 'expected_predictions': len(cases)*len(METHODS),
        'unique_reader_requests': len(jobs), 'cases_per_reader_call': 1, 'context_token_limit': 2048,
        'reader_model': Codex.model, 'reader_prompt': 'Unchanged bounded_clients.Codex reader instruction.',
        'replicates': 1, 'identical_input_policy': 'One shared prediction per exact case ID, query and context across methods.',
        'data_hashes': {name: sha(output / name) for name in ('cases.json','requests.json','contexts.json')},
        'source_hashes': {name: sha(ROOT / name) for name in paths}})
    print({'questions': len(cases), 'predictions': len(cases)*len(METHODS), 'unique_single_case_calls': len(jobs)})


def verify(output):
    manifest = read_json(output / 'manifest.json')
    for name, expected in manifest['data_hashes'].items():
        if sha(output / name) != expected:
            raise ValueError('Changed reader input: ' + name)
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed reader implementation: ' + name)
    return manifest


def run(output):
    verify(output)
    client = Codex(output, research_budget())
    jobs = read_json(output / 'requests.json')
    def read(key):
        path = output / 'predictions' / (key + '.json')
        if path.exists():
            return
        request = jobs[key]
        answer = validate_answers(client.call([request], 'reader'), [request['id']])[request['id']].strip()
        save(path, {'id': request['id'], 'answer': answer, 'request_sha256': key,
                    'status': 'ok' if answer else 'failed'})
    parallel_progress(read, sorted(jobs), 2, 'isolated-development-reader')


def score(output):
    verify(output)
    cases, contexts = read_json(output / 'cases.json'), read_json(output / 'contexts.json')
    docs, gold = read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    spec = importlib.util.spec_from_file_location('qasper_isolated_reader', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'p': {'qas': [{'question_id': c['id'],
        'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases if c['dataset'] == 'qasper']}}, False)
    rows = []
    for case in cases:
        for method in METHODS:
            context = contexts[case['id']][method]
            prediction = read_json(output / 'predictions' / (context['request_sha256'] + '.json'))
            if prediction['id'] != case['id'] or prediction['request_sha256'] != context['request_sha256']:
                raise ValueError('Prediction identity mismatch')
            row = {**prediction, 'method': method, 'doc_id': case['doc_id'], 'dataset': case['dataset'],
                'context_tokens': context['context_tokens'], 'spans': context['spans'],
                'evidence_recall': None, 'complete_evidence': None, 'retrieved_evidence_f1': None}
            if case['dataset'] == 'quality':
                if row['answer'] not in ('A','B','C','D'):
                    row['status'] = 'failed'
                row['answer_score'] = int(row['status'] == 'ok' and row['answer'] == gold[case['id']]['label'])
            else:
                ref = refs[case['id']]
                row['answer_score'] = max(official.token_f1_score(row['answer'], a['answer']) for a in ref) if row['status'] == 'ok' else 0
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
        for dataset in ('qasper','quality'):
            group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
            metrics = {}
            for name in ('answer_score','evidence_recall','complete_evidence','retrieved_evidence_f1','context_tokens'):
                values = [r[name] for r in group if r[name] is not None]
                metrics[name] = statistics.mean(values) if values else None
            summary['methods'][method][dataset] = {'questions': len(group), **metrics}
    save(output / 'scores.json', rows)
    save(output / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run','score'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output/isolated-reader-v1')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.command](args.output)
