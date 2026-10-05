"""Live split x representative x router study, with independent retrieval baselines."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import Embeddings, save, signature
from scripts.bounded_eval import Jev, bm25, covered_paragraphs, query_text, read_json, sections, sha, tokenizer
from scripts.live_tree_clients import Gemini, LiveBudget
from scripts.planned_evidence_trial import PLAN_INSTRUCTION, PLAN_SCHEMA, validate_plan
from zero_index.context import candidate_pool
from zero_index.embeddings import CentroidRepresentatives
from zero_index.index import DocumentIndex, Node, reselect_representatives
from zero_index.parse import Block, sentence_spans
from zero_index.tree_search import TreeSearchConfig, EmbeddingTreeRouter, _card, routing_text, search_tree
from zero_index.hybrid_retrieval import tree_passages, fuse_retrieval_paths

OLD = ROOT / 'output/bounded-v1'
OUT = ROOT / 'output/live-tree-v1'
TREES = [''.join(p) for p in itertools.product('EJ', repeat=3)]
METHODS = TREES + ['dense_recursive', 'rrf_recursive', 'rerank_recursive', 'dense_semantic',
                   'rrf_semantic', 'rerank_semantic', 'hybrid_full', 'hybrid_matched']


def tree(doc, groups, partition):
    source = doc['text']
    root = Node('root', 'document', doc['title'], 0, len(source))
    headings = {}
    for number, group in enumerate(groups):
        native = group[0]['section']
        if native not in headings:
            units = [u for u in doc['units'] if u['section'] == native]
            headings[native] = Node(f'h{native}', 'heading', group[0]['heading'] or doc['title'],
                                     units[0]['start'], units[-1]['end'])
            root.children.append(headings[native])
        section = Node(f'b{number}', 'section', 'Section', group[0]['start'], group[-1]['end'])
        headings[native].children.append(section)
        for unit in group:
            paragraph = Node(f'p{unit["paragraph"]}', 'paragraph', f'Paragraph {unit["paragraph"] + 1}',
                             unit['start'], unit['end'])
            section.children.append(paragraph)
            for i, (a, b) in enumerate(sentence_spans(source, Block('paragraph', unit['start'], unit['end']))):
                paragraph.children.append(Node(f'{paragraph.node_id}s{i}', 'sentence', source[a:b], a, b))
    result = DocumentIndex(source, doc['id'], root, {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                           'partition': partition, 'config': {}, 'structure': 'dataset-native headings and paragraph offsets'}, [])
    return DocumentIndex.from_dict(result.to_dict())


def cached_partitions(doc):
    """Recover the original groups, not the 512-token chunks packed from them."""
    import numpy as np
    prior = read_json(OLD / 'indexes' / (signature(doc['id']) + '.json'))
    cuts = {x['candidate'] for x in prior['trace'] if x['cut']}
    output = {'E': [], 'J': []}
    for native in sections(doc):
        vectors = []
        for u in native:
            text = 'task: sentence similarity | query: ' + doc['text'][u['start']:u['end']]
            key = signature({'model': Embeddings.model, 'dimensions': 768, 'text': text})
            vectors.append(np.asarray(read_json(OLD / 'embedding-cache' / (key + '.json'))))
        distances = [1 - float(a @ b) for a, b in zip(vectors, vectors[1:])]
        threshold = float(np.quantile(distances, .85)) if distances else 2.0
        ecuts = {u['start'] for u, distance in zip(native[1:], distances) if distance > threshold}
        for method, boundaries in [('E', ecuts), ('J', cuts)]:
            group = []
            for unit in native:
                if group and unit['start'] in boundaries:
                    output[method].append(group); group = []
                group.append(unit)
            output[method].append(group)
    return output


def prepare(output):
    if (output / 'manifest.json').exists():
        raise ValueError('Already frozen')
    output.mkdir(parents=True, exist_ok=True)
    cases, docs = read_json(OLD / 'cases.json'), read_json(OLD / 'documents.json')
    registry = {r['id']: r for r in read_json(ROOT / 'evals/iterations/v1/registry.json')}
    assert all(registry[c['id']]['partition'] == 'development' for c in cases)
    sizes, node_counts = [], Counter()
    for doc in docs.values():
        for method, groups in cached_partitions(doc).items():
            index = tree(doc, groups, method)
            for node in index.root.walk():
                node_counts[node.kind] += 1
                if node.kind == 'section':
                    sizes.append(node.end - node.start)
            save(output / 'topology' / (signature(doc['id']) + '-' + method + '.json'), index.to_dict())
    paths = ['scripts/live_tree_eval.py', 'scripts/live_tree_clients.py', 'scripts/bounded_clients.py',
             'scripts/bounded_eval.py', 'scripts/planned_evidence_trial.py']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'zero_index').glob('*.py')]
    for name in paths:
        dest = output / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    save(output / 'cases.json', cases)
    save(output / 'manifest.json', {
        'status': 'frozen_before_inference', 'partition': 'exposed_development_only', 'methods': METHODS,
        'questions': len(cases), 'documents': len(docs), 'cases_sha256': sha(output / 'cases.json'),
        'documents_sha256': sha(OLD / 'documents.json'), 'registry_sha256': sha(ROOT / 'evals/iterations/v1/registry.json'),
        'source_hashes': {name: sha(ROOT / name) for name in paths},
        'topology_hashes': {p.name: sha(p) for p in sorted((output / 'topology').glob('*.json'))},
        'factor_order': ['split', 'representative', 'router'], 'representative_candidates': 8,
        'representative_stop_threshold': None, 'jev_max_state_chars': 250000,
        'router': {'beam_width': 2, 'max_node_scores': 256, 'max_preview_tokens': 8192, 'max_depth': 16},
        'shared_llm': {'model': Gemini.model, 'temperature': 0, 'thinking_budget': 0, 'max_output_tokens': 512,
                       'price_input_per_million': .30, 'price_output_per_million': 2.50,
                       'pricing_source': 'https://ai.google.dev/gemini-api/docs/pricing'},
        'embedding_model': Embeddings.model, 'embedding_dimensions': 768, 'jev_model': 'jev-1.13.0',
        'final_context_tokens': 2048, 'flat_candidate_source_tokens': 8192,
        'hybrid_full': '8192 preview tokens plus 8192 direct source tokens, RRF 60 equal weights',
        'hybrid_matched': '4096 preview tokens plus 4096 direct source tokens, RRF 60 equal weights',
        'planner_input': 'question and options, title, native heading titles; no source bodies or gold',
        'reader_replicates': 1, 'reuse_identical_inputs': True,
        'analysis': 'Official QASPER F1/evidence and QuALITY-HARD accuracy, paired document bootstrap; descriptive development only',
        'budget_path': 'output/research-budget.json', 'budget_before': read_json(ROOT / 'output/research-budget.json'),
        'preflight': {'node_counts_across_two_topologies': dict(node_counts), 'maximum_topic_chars': max(sizes),
                      'maximum_reader_outputs': len(cases) * len(METHODS),
                      'dollar_control': 'Native token count plus 256 wrapper tokens and 512 output tokens reserved before each attempt; hard cumulative $30 cap'},
        'limitations': ['RAPTOR is a separate published-baseline experiment, not represented by these custom trees.',
                        'Single affordable reader; no cross-reader robustness or frontier superiority claim.',
                        'All 384 questions are previously exposed development data.']})
    print(json.dumps({'stage': 'prepared', 'questions': len(cases), 'methods': len(METHODS), 'nodes': dict(node_counts), 'max_topic_chars': max(sizes)}), flush=True)


def verify(output):
    m = read_json(output / 'manifest.json')
    if sha(output / 'cases.json') != m['cases_sha256'] or sha(OLD / 'documents.json') != m['documents_sha256']:
        raise ValueError('Input changed')
    for name, expected in m['source_hashes'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen implementation changed: ' + name)
    for name, expected in m['topology_hashes'].items():
        if sha(output / 'topology' / name) != expected:
            raise ValueError('Topology changed')
    return m


def parallel(function, items, workers, stage):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, item) for item in items]
        errors = []
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc)[:160])
            print(json.dumps({'stage': stage, 'completed': done, 'total': len(items), 'failures': len(errors),
                              'last_error': errors[-1] if errors else None}), flush=True)
        if errors:
            raise RuntimeError(str(Counter(errors)))


def run(output, stage, workers=8):
    verify(output)
    cases, docs = read_json(output / 'cases.json'), read_json(OLD / 'documents.json')
    budget = LiveBudget(ROOT / 'output/research-budget.json')
    enc = tokenizer(); count = lambda s: len(enc.encode(s, disallowed_special=()))
    embed = Embeddings(output, budget)
    embed.cache = OLD / 'embedding-cache'  # Exact content-addressed cache; new calls have this run's audit.
    llm = Gemini(output, budget)

    def jev():
        j = Jev(output, budget); j.max_state_chars = 250000
        return j

    def index_doc(doc_id):
        key = signature(doc_id); j = jev()
        base = {s: DocumentIndex.from_dict(read_json(output / 'topology' / (key + '-' + s + '.json'))) for s in 'EJ'}
        if all((output / 'indexes' / (key + '-' + s + r + '.json')).exists() for s in 'EJ' for r in 'EJ'):
            return
        texts = list(dict.fromkeys(n.title for n in base['E'].root.walk() if n.kind == 'sentence'))
        vectors = embed.embed(texts, 'central-sentences')
        lookup = dict(zip(texts, vectors))
        centroid = CentroidRepresentatives(lambda s: lookup[s], model_name=embed.model)
        for s in 'EJ':
            for r, scorer in [('E', centroid), ('J', j)]:
                path = output / 'indexes' / (key + '-' + s + r + '.json')
                if not path.exists():
                    started = time.perf_counter()
                    result = reselect_representatives(base[s], scorer, sentence_budget=8)
                    result.metadata['representative_seconds'] = time.perf_counter() - started
                    save(path, result.to_dict())

    def plan(case):
        path = output / 'plans' / (signature(case['id']) + '.json')
        if path.exists():
            return
        doc = docs[case['doc_id']]
        payload = {'question': query_text(case), 'title': doc['title'],
                   'headings': [g[0]['heading'] for g in sections(doc)]}
        value = llm.structured(PLAN_INSTRUCTION, payload, PLAN_SCHEMA, validate_plan, 'planner')
        save(path, {'id': case['id'], 'needs': validate_plan(value)})

    def retrieve(case):
        path = output / 'retrieval' / (signature(case['id']) + '.json')
        if path.exists():
            return
        started = time.perf_counter(); doc = docs[case['doc_id']]; key = signature(doc['id'])
        needs = read_json(output / 'plans' / (signature(case['id']) + '.json'))['needs']
        question = query_text(case)
        indexes = {s+r: DocumentIndex.from_dict(read_json(output / 'indexes' / (key + '-' + s+r + '.json'))) for s in 'EJ' for r in 'EJ'}
        texts = list(dict.fromkeys(routing_text(_card(index, n)) for index in indexes.values() for n in index.root.walk() if n.kind != 'document'))
        vectors = embed.embed(texts, 'routing-previews')
        lookup = dict(zip(texts, vectors))
        queries = [f'Question: {question}\nEvidence need: {need}' for need in needs]
        qvectors = embed.embed(['task: question answering | query: ' + q for q in queries], 'planned-queries')
        qlookup = dict(zip(queries, qvectors))
        router = EmbeddingTreeRouter(lambda s: lookup[s], model_name=embed.model, embed_query=lambda s: qlookup[s])
        j = jev(); methods, searches, rankings = {}, {}, {}
        def pack(first, second=()):
            return fuse_retrieval_paths(doc['text'], first, list(second), token_count=count, title=doc['title'], budget=2048)
        for method in TREES:
            index = indexes[method[:2]]
            result = search_tree(index, question, needs, router if method[2] == 'E' else j, token_count=count)
            searches[method] = result
            rankings[method] = tree_passages(index, result)
            methods[method] = pack(rankings[method])
        chunks = read_json(OLD / 'indexes' / (key + '.json'))
        flat_rankings = {}
        for partition in ['recursive', 'semantic']:
            pool = chunks[partition]
            texts = [f'title: {doc["title"]} / {c["heading"]} | text: {c["text"]}' for c in pool]
            vectors = embed.embed(texts, 'direct-passages')
            dense_rrf, lexical_rrf = Counter(), Counter()
            for need, qvector in zip(needs, qvectors):
                cosine = vectors @ qvector
                lexical = bm25(texts, question + '\nEvidence need: ' + need)
                for scores, target in [(cosine, dense_rrf), (lexical, lexical_rrf)]:
                    for rank, i in enumerate(sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i]['start'])), 1):
                        target[i] += 1 / (60 + rank)
            dense = [pool[i] for i in sorted(dense_rrf, key=lambda i: (-dense_rrf[i], pool[i]['start']))]
            combined = dense_rrf + lexical_rrf
            rrf = [pool[i] for i in sorted(combined, key=lambda i: (-combined[i], pool[i]['start']))]
            for name, ranking in [('dense', dense), ('rrf', rrf)]:
                shortlist = candidate_pool(ranking, count)['candidates']
                flat_rankings[name+'_'+partition] = shortlist
                methods[name+'_'+partition] = pack(shortlist)
            candidates = flat_rankings['rrf_'+partition]
            request = {'id': case['id'], 'question': question, 'needs': needs,
                       'candidates': [{'id': i, 'heading': c['heading'], 'text': c['text']} for i, c in enumerate(candidates)]}
            order = llm.call([request], 'ranker')['rankings'][0]['order']
            methods['rerank_'+partition] = pack([candidates[i] for i in order])
        methods['hybrid_full'] = pack(rankings['JJJ'], flat_rankings['dense_recursive'])
        matched_search = search_tree(indexes['JJ'], question, needs, j, token_count=count,
                                    config=TreeSearchConfig(max_preview_tokens=4096))
        searches['hybrid_matched'] = matched_search
        direct_matched = candidate_pool(flat_rankings['dense_recursive'], count, max_source_tokens=4096)['candidates']
        methods['hybrid_matched'] = pack(tree_passages(indexes['JJ'], matched_search), direct_matched)
        save(path, {'id': case['id'], 'needs': needs, 'methods': methods, 'searches': searches,
                    'flat_rankings': flat_rankings, 'seconds': time.perf_counter() - started})

    def read(job):
        case, method = job
        path = output / 'predictions' / method / (signature(case['id']) + '.json')
        if path.exists():
            return
        retrieval = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))['methods'][method]
        request = {'id': case['id'], 'question': case['question'], 'options': case['options'], 'context': retrieval['context']}
        value = llm.call([request], 'reader')['answers'][0]
        save(path, {'id': case['id'], 'method': method, 'status': 'ok', 'answer': value['answer']})

    for name, function, items in [('plan', plan, cases), ('index', index_doc, sorted(docs)),
                                  ('retrieve', retrieve, cases), ('read', read, [(c, m) for c in cases for m in METHODS])]:
        if stage in ('all', name):
            parallel(function, items, workers, name)
    save(output / 'last-stage.json', {'stage': stage, 'budget': budget.value})


def score(output):
    verify(output)
    cases, docs, gold = read_json(output / 'cases.json'), read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    spec = importlib.util.spec_from_file_location('qasper_official', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'paper': {'qas': [
        {'question_id': c['id'], 'answers': [{'answer': a} for a in gold[c['id']]['answers']]}
        for c in cases if c['dataset'] == 'qasper']}}, False)
    rows = []
    for case in cases:
        retrieved = read_json(output / 'retrieval' / (signature(case['id']) + '.json'))
        for method in METHODS:
            prediction = read_json(output / 'predictions' / method / (signature(case['id']) + '.json'))
            assert prediction['id'] == case['id'] and prediction['method'] == method
            context = retrieved['methods'][method]
            row = {**prediction, 'doc_id': case['doc_id'], 'dataset': case['dataset'],
                   'context_tokens': context['context_tokens'], 'evidence_recall': None, 'complete_evidence': None,
                   'routing_status': retrieved['searches'].get(method, {}).get('status'),
                   'preview_tokens': retrieved['searches'].get(method, {}).get('preview_tokens')}
            if case['dataset'] == 'quality':
                row['answer_score'] = int(prediction['answer'] == gold[case['id']]['label'])
            else:
                row['answer_score'] = max(official.token_f1_score(prediction['answer'], r['answer']) for r in refs[case['id']])
                evidence = set(covered_paragraphs(docs[case['doc_id']], context['spans']))
                eligible = [r for r in refs[case['id']] if r['evidence']]
                if eligible:
                    row['evidence_recall'] = max(len(evidence & set(r['evidence'])) / len(set(r['evidence'])) for r in eligible)
                    row['complete_evidence'] = int(any(set(r['evidence']) <= evidence for r in eligible))
            rows.append(row)
    summary = {'questions': len(cases), 'predictions': len(rows), 'partition': 'exposed_development_only', 'methods': {}}
    for dataset in ['qasper', 'quality']:
        summary['methods'][dataset] = {}
        for method in METHODS:
            subset = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            summary['methods'][dataset][method] = {'n': len(subset), **{
                k: statistics.mean([r[k] for r in subset if r[k] is not None]) if any(r[k] is not None for r in subset) else None
                for k in ['answer_score', 'evidence_recall', 'complete_evidence', 'context_tokens']},
                'routing_status': dict(Counter(r['routing_status'] for r in subset if r['routing_status']))}
    summary['budget'] = read_json(ROOT / 'output/research-budget.json')
    save(output / 'scores.json', rows); save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'score'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--stage', choices=['all', 'index', 'plan', 'retrieve', 'read'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'run': run(args.output, args.stage, args.workers)
    else: score(args.output)
