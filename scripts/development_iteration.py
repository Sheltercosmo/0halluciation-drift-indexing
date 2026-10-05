"""Frozen development experiments on exposed bounded-v1 data only."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import Budget, Codex, Embeddings, save, signature, validate_answers, validate_rankings
from scripts.bounded_eval import (Jev, batches, context_for, covered_paragraphs, parallel_progress,
                                 query_text, rank_chunks, read_json, sha, tokenizer)
from zero_index.context import candidate_pool

METHODS = ('jev_pool_8192', 'codex_pool_8192')
OLD = ROOT / 'output/bounded-v1'


class CachedEmbeddings(Embeddings):
    def count_tokens(self, texts):
        raise RuntimeError('Development iteration requires cached embeddings; no new Gemini calls permitted')


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Iteration is already frozen')
    split = read_json(ROOT / 'evals/iterations/v1/manifest.json')
    ids = set(split['development_screening_ids'])
    cases = [c for c in read_json(OLD / 'cases.json') if c['id'] in ids]
    if len(cases) != len(ids):
        raise ValueError('Development coverage mismatch')
    save(output / 'cases.json', cases)
    paths = ['scripts/development_iteration.py', 'scripts/bounded_eval.py', 'scripts/bounded_clients.py']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'zero_index').glob('*.py')]
    for name in paths:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    save(output / 'manifest.json', {'status': 'frozen_before_inference', 'hypothesis':
        'A shared 8192-source-token candidate allowance repairs evidence loss caused by a 12-chunk cap.',
        'partition': 'exposed_development_only', 'methods': METHODS, 'questions': len(cases),
        'source_token_pool': 8192, 'max_candidates': 128, 'context_tokens': 2048,
        'old_run_manifest_sha256': sha(OLD / 'manifest.json'),
        'split_registry_sha256': split['registry_sha256'], 'cases_sha256': sha(output / 'cases.json'),
        'source_hashes': {name: sha(ROOT / name) for name in paths}})
    print(json.dumps({'status': 'prepared', 'questions': len(cases), 'methods': METHODS}))


def verify(output):
    manifest = read_json(output / 'manifest.json')
    if manifest['partition'] != 'exposed_development_only' or sha(output / 'cases.json') != manifest['cases_sha256']:
        raise ValueError('Changed or unauthorized development input')
    for name, expected in manifest['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Implementation changed after freeze: ' + name)
    if sha(OLD / 'manifest.json') != manifest['old_run_manifest_sha256']:
        raise ValueError('Historical run changed')
    return manifest


def research_budget():
    path = ROOT / 'output/research-budget.json'
    if not path.exists():
        old = read_json(OLD / 'budget.json')
        budget = Budget(path)
        budget.value.update(previous_work_reserved_usd=old['gemini_reserved_upper_bound_usd'],
                            codex_calls_limit=1000, jev_calls_limit=30000, jev_questions_limit=500000)
        save(path, budget.value)
    return Budget(path)


def run(output, stage):
    verify(output)
    cases, docs = read_json(output / 'cases.json'), read_json(OLD / 'documents.json')
    enc, budget = tokenizer(), research_budget()
    embeddings = CachedEmbeddings(output, budget)
    embeddings.cache = OLD / 'embedding-cache'
    codex = Codex(output, budget)
    for folder in ('retrieval', 'predictions'):
        (output / folder).mkdir(exist_ok=True)
    status = {'stage': stage, 'status': 'running', 'questions': len(cases)}
    save(output / 'status.json', status)

    def retrieve(case):
        path = output / 'retrieval' / (signature(case['id']) + '.json')
        if path.exists():
            return
        started = time.perf_counter()
        doc = docs[case['doc_id']]
        index = read_json(OLD / 'indexes' / (signature(case['doc_id']) + '.json'))
        pools = {}
        for kind in ('jev', 'recursive'):
            ranking = rank_chunks(doc, index[kind], case, embeddings)
            pools[kind] = candidate_pool(ranking, lambda s: len(enc.encode(s, disallowed_special=())))
        pool = pools['jev']['candidates']
        cards = [{'node_id': c['id'], 'text': c['text'], 'heading_path': [doc['title'], c['heading']],
                  'paragraph_representative': ''} for c in pool]
        scores = Jev(output, budget).rerank(case['question'], query_text(case), cards)
        ranking = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-scores[i], i))]
        context, spans, tokens = context_for(doc, ranking, enc)
        save(path, {'id': case['id'], 'doc_id': case['doc_id'], 'pools': pools, 'jev_scores': scores,
            'methods': {'jev_pool_8192': {'context': context, 'spans': spans, 'context_tokens': tokens,
                         'ranking': [c['id'] for c in ranking]}}, 'retrieval_seconds': time.perf_counter()-started})

    def rerank(batch):
        request, pools = [], {}
        for case in batch:
            row = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))
            if 'codex_pool_8192' in row['methods']:
                continue
            pool = row['pools']['recursive']['candidates'][:]
            random.Random(signature({'seed': 20261005, 'id': case['id']})).shuffle(pool)
            pools[case['id']] = pool
            request.append({'id': case['id'], 'question': query_text(case),
                'candidates': [{'id': i, 'title': docs[case['doc_id']]['title'], 'heading': c['heading'], 'text': c['text']}
                               for i, c in enumerate(pool)]})
        if not request:
            return
        order = validate_rankings(codex.call(request, 'reranker'), {x['id']: len(x['candidates']) for x in request})
        for case in batch:
            if case['id'] not in pools:
                continue
            path = output / 'retrieval' / (signature(case['id']) + '.json')
            row = read_json(path)
            ranking = [pools[case['id']][i] for i in order[case['id']]]
            context, spans, tokens = context_for(docs[case['doc_id']], ranking, enc)
            row['methods']['codex_pool_8192'] = {'context': context, 'spans': spans, 'context_tokens': tokens,
                                               'ranking': [c['id'] for c in ranking]}
            save(path, row)

    def reader(job):
        method, batch = job
        folder = output / 'predictions' / method
        if all((folder / (signature(c['id']) + '.json')).exists() for c in batch):
            return
        request = [{'id': c['id'], 'question': query_text(c), 'context': read_json(
            output / 'retrieval' / (signature(c['id']) + '.json'))['methods'][method]['context']} for c in batch]
        answers = validate_answers(codex.call(request, 'reader'), [c['id'] for c in batch])
        for case in batch:
            answer = answers[case['id']].strip()
            valid = bool(answer) and (not case['options'] or answer in 'ABCD' and len(answer) == 1)
            save(folder / (signature(case['id']) + '.json'), {'id': case['id'], 'method': method, 'answer': answer,
                'status': 'ok' if valid else 'failed', 'request_sha256': signature(request)})
    try:
        if stage in ('all', 'retrieve'):
            parallel_progress(retrieve, cases, 4, 'development-retrieve')
        if stage in ('all', 'rerank'):
            parallel_progress(rerank, list(batches(cases, 2)), 2, 'development-rerank')
        if stage in ('all', 'read'):
            parallel_progress(reader, [(m, b) for b in batches(cases) for m in METHODS], 2, 'development-read')
        status['status'] = 'inference_complete' if stage == 'all' else 'stage_complete'
    except BaseException as exc:
        status.update(status='incomplete', error_type=type(exc).__name__)
        raise
    finally:
        save(output / 'status.json', status)


def score(output):
    verify(output)
    cases, docs, gold = [read_json(OLD / name) for name in ('cases.json', 'documents.json', 'gold.json')]
    ids = {c['id'] for c in read_json(output / 'cases.json')}
    cases = [c for c in cases if c['id'] in ids]
    spec = importlib.util.spec_from_file_location('qasper_official_iteration', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'paper': {'qas': [{'question_id': c['id'],
        'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases if c['dataset'] == 'qasper']}}, False)
    rows = [r for r in read_json(OLD / 'scores.json') if r['id'] in ids]
    for case in cases:
        retrieval = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))
        for method in METHODS:
            prediction = read_json(output / 'predictions' / method / (signature(case['id']) + '.json'))
            context = retrieval['methods'][method]
            row = {**prediction, 'doc_id': case['doc_id'], 'dataset': case['dataset'],
                'context_tokens': context['context_tokens'], 'spans': context['spans'],
                'evidence_recall': None, 'retrieved_evidence_f1': None, 'complete_evidence': None}
            if case['dataset'] == 'quality':
                row['answer_score'] = int(prediction['status'] == 'ok' and prediction['answer'] == gold[case['id']]['label'])
            else:
                ref = refs[case['id']]
                row['answer_score'] = max(official.token_f1_score(prediction['answer'], r['answer']) for r in ref) if prediction['status'] == 'ok' else 0
                evidence = covered_paragraphs(docs[case['doc_id']], context['spans'])
                row['retrieved_evidence_f1'] = max(official.paragraph_f1_score(evidence, r['evidence']) for r in ref)
                eligible = [r for r in ref if r['evidence']]
                if eligible:
                    row['evidence_recall'] = max(len(set(evidence)&set(r['evidence']))/len(set(r['evidence'])) for r in eligible)
                    row['complete_evidence'] = int(any(set(r['evidence']) <= set(evidence) for r in eligible))
            rows.append(row)
    summary = {'status': 'development_only_not_confirmatory', 'questions': len(cases), 'methods': {}}
    for method in sorted({r['method'] for r in rows}):
        summary['methods'][method] = {}
        for dataset in ('qasper', 'quality'):
            group = [r for r in rows if r['method'] == method and r['dataset'] == dataset]
            metrics = {}
            for key in ('answer_score', 'evidence_recall', 'complete_evidence', 'retrieved_evidence_f1', 'context_tokens'):
                values = [r[key] for r in group if r[key] is not None]
                metrics[key] = statistics.mean(values) if values else None
            summary['methods'][method][dataset] = {'questions': len(group), **metrics}
    save(output / 'scores.json', rows)
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'score'])
    p.add_argument('--output', type=Path, default=ROOT / 'output/development-pool-v1')
    p.add_argument('--stage', choices=['all', 'retrieve', 'rerank', 'read'], default='all')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'run':
        run(args.output, args.stage)
    else:
        score(args.output)
