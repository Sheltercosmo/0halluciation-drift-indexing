"""Development repair: recover exhausted searches and actually read bottom-up."""
import argparse
from pathlib import Path
import statistics
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import Jev, covered_paragraphs, read_json, sha, query_text, tokenizer, paired_interval
from scripts.live_tree_clients import Gemini, LiveBudget, LiveEmbeddings, GenerationBlocked
from scripts.live_tree_eval import OUT as BASE, OLD, parallel
from scripts.live_tree_reader_v2 import OUT as READER_V2, read_case
from zero_index.index import DocumentIndex
from zero_index.tree_search import EmbeddingTreeRouter, routing_text, search_tree
from zero_index.hybrid_retrieval import tree_passages, fuse_retrieval_paths

OUT = ROOT / 'output/live-tree-bottom-up-v1'
METHODS = ['EEE_bottom_up', 'JJJ_bottom_up']


def bottom_up_context(index, search, count, budget=2048):
    """Keep reached paragraphs first; expand their topic parents and nearby peers."""
    cores = tree_passages(index, search)
    ranked, seen = [], set()
    def add(node):
        span = (node.start, node.end)
        if span not in seen:
            seen.add(span)
            ranked.append({'start': node.start, 'end': node.end, 'text': index.source[node.start:node.end],
                           'node_id': node.node_id})
    for core in cores:
        add(index._node(core['node_id']))
    # Oversized paragraphs may be skipped by the packer; keep their reached leaves usable.
    for leaf in search['leaves']:
        add(index._node(leaf['node_id']))
    parents = []
    for core in cores:
        parent = index._node(index._parents[core['node_id']])
        if parent.node_id not in [n.node_id for n in parents]:
            parents.append(parent)
            add(parent)
    # If a whole topic does not fit, expand neighboring paragraphs, round-robin
    # across the independently reached cores, staying inside those topics.
    queues = []
    for core in cores:
        node = index._node(core['node_id']); parent = index._node(index._parents[node.node_id])
        siblings = [n for n in parent.children if n.kind == 'paragraph']
        position = next(i for i, n in enumerate(siblings) if n.node_id == node.node_id)
        queues.append([siblings[i] for i in sorted(range(len(siblings)), key=lambda i: (abs(i-position), i)) if i != position])
    for depth in range(max(map(len, queues), default=0)):
        for queue in queues:
            if depth < len(queue): add(queue[depth])
    # A small native heading can be read last, after its more precise topic nodes.
    for parent in parents:
        heading = index._node(index._parents[parent.node_id])
        if heading.kind == 'heading': add(heading)
    result = fuse_retrieval_paths(index.source, ranked, [], token_count=count, title=index.root.title, budget=budget)
    result['expansion_policy'] = 'core paragraphs, reached leaves, topic parents, nearby same-topic paragraphs, small native headings'
    return result


def prepare():
    if (OUT / 'manifest.json').exists(): raise ValueError('Already frozen')
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_json(BASE / 'cases.json')
    assert len(list((BASE / 'retrieval').glob('*.json'))) == len(cases)
    paths = ['scripts/live_tree_followup.py', 'scripts/live_tree_reader_v2.py', 'scripts/live_tree_clients.py', 'scripts/live_tree_eval.py',
             'scripts/bounded_clients.py', 'scripts/bounded_eval.py', 'scripts/planned_evidence_trial.py']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'zero_index').glob('*.py')]
    for name in paths:
        target = OUT / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / name).read_bytes())
    save(OUT / 'cases.json', cases)
    save(OUT / 'manifest.json', {'status': 'frozen_before_followup_inference', 'partition': 'exposed_development_only',
        'methods': METHODS, 'questions': len(cases), 'documents': 307, 'cases_sha256': sha(BASE / 'cases.json'),
        'base_manifest_sha256': sha(BASE / 'manifest.json'),
        'reader_v2_manifest_sha256': sha(READER_V2 / 'manifest.json'),
        'source_hashes': {name: sha(ROOT / name) for name in paths},
        'motivation': 'Complete primary retrieval diagnostics: JJJ has only 297 QASPER context tokens on average and 109/192 empty QuALITY contexts. No aggregate primary answer scores were used to choose this repair.',
        'search': 'Reuse original EEE/JJJ search. On budget exhaustion or no leaves, retry with the first shared LLM need, beam 2, 256 node scores and 8192 preview tokens. Use retry only if it reaches a leaf; otherwise retain original leaves.',
        'cost_limit': 'At most two search attempts, each capped at 8192 preview tokens and 256 node scores. This is extra compute, not a free or compute-matched gain.',
        'reading': 'Read cores first, then topic parents and nearby paragraphs within the reached topics, then small headings, without displacing core evidence. Exact-source union, 2048 final cl100k tokens.',
        'shared_reader': {'model': Gemini.model, 'temperature': 0, 'thinking_budget': 0, 'max_output_tokens': 512,
                          'policy': 'QASPER isolated-input-id-first-usable-stop-on-block; QuALITY labelled-options-enum-v2'},
        'blocked_inputs': 'Skip questions blocked in either primary reader study; no new model calls for them. Keep zero-score rows and report common-unblocked sensitivity.',
        'selection_policy': 'Publish both repairs and the original controls; no default promotion or validation/test claims.',
        'budget_snapshot_before_reader_rerun': read_json(READER_V2 / 'manifest.json')['budget_before']})
    print({'prepared_followup': METHODS, 'questions': len(cases)})


def verify():
    m = read_json(OUT / 'manifest.json')
    assert sha(BASE / 'manifest.json') == m['base_manifest_sha256']
    assert sha(READER_V2 / 'manifest.json') == m['reader_v2_manifest_sha256']
    assert sha(OUT / 'cases.json') == m['cases_sha256']
    for name, digest in m['source_hashes'].items():
        assert sha(ROOT / name) == digest, name
    return m


def run():
    verify(); budget = LiveBudget(ROOT / 'output/research-budget.json')
    embed = LiveEmbeddings(OUT, budget); embed.cache = OLD / 'embedding-cache'
    reader = Gemini(OUT, budget); reader.cache = BASE / 'gemini-cache'; reader.responses = BASE / 'gemini-responses'
    mc_reader = Gemini(OUT, budget); mc_reader.cache = READER_V2 / 'gemini-cache'; mc_reader.responses = READER_V2 / 'gemini-responses'
    enc = tokenizer(); count = lambda s: len(enc.encode(s, disallowed_special=()))
    cases = read_json(OUT / 'cases.json')
    blocked_ids = {r['id'] for r in read_json(BASE / 'scores.json') + read_json(READER_V2 / 'scores.json') if r['status'] == 'blocked_by_reader'}
    def retrieve(case):
        path = OUT / 'retrieval' / (signature(case['id']) + '.json')
        if path.exists(): return
        if case['id'] in blocked_ids:
            save(path, {'id': case['id'], 'skip_reader': 'Known primary reader block; no new model calls for this question',
                'methods': {m: {'context': '', 'context_tokens': 0, 'spans': []} for m in METHODS},
                'searches': {m: {'used_search': {'status': 'not_run_known_reader_block'}, 'total_preview_tokens': 0} for m in METHODS}})
            return
        original = read_json(BASE / 'retrieval' / path.name); outputs, traces = {}, {}
        for method in METHODS:
            base = method[:3]; search = original['searches'][base]
            index = DocumentIndex.from_dict(read_json(BASE / 'indexes' / (signature(case['doc_id']) + '-' + base[:2] + '.json')))
            attempted = None
            if search['status'] == 'routing_budget_exhausted' or not search['leaves']:
                if base == 'JJJ':
                    router = Jev(OUT, budget); router.cache = BASE / 'jev-cache'; router.max_state_chars = 250000
                else:
                    def edoc(s): return embed.embed([s], 'retry-previews')[0]
                    def equery(s): return embed.embed(['task: question answering | query: ' + s], 'retry-query')[0]
                    inner = EmbeddingTreeRouter(edoc, model_name=embed.model, embed_query=equery)
                    class BatchedRouter:
                        name = inner.name
                        def route(self, question, need, cards):
                            embed.embed(list(dict.fromkeys(routing_text(c) for c in cards)), 'retry-previews')
                            return inner.route(question, need, cards)
                    router = BatchedRouter()
                attempted = search_tree(index, query_text(case), original['needs'][:1], router, token_count=count)
                if attempted['leaves']: search = attempted
            outputs[method] = bottom_up_context(index, search, count)
            traces[method] = {'original_status': original['searches'][base]['status'], 'used_search': search,
                              'retry_search': attempted, 'total_preview_tokens': original['searches'][base]['preview_tokens'] + (attempted['preview_tokens'] if attempted else 0)}
        save(path, {'id': case['id'], 'methods': outputs, 'searches': traces})
    parallel(retrieve, cases, 8, 'followup-retrieve')
    def read(job):
        case, method = job; path = OUT / 'predictions' / method / (signature(case['id']) + '.json')
        if path.exists(): return
        retrieved = read_json(OUT / 'retrieval' / path.name); context = retrieved['methods'][method]
        if retrieved.get('skip_reader'):
            value = {'status': 'blocked_by_reader', 'answer': '', 'failure': retrieved['skip_reader']}
        else:
            try:
                answer = read_case(mc_reader if case['options'] else reader, case, context['context'])
                value = {'answer': answer, 'status': 'ok'}
            except GenerationBlocked as exc:
                value = {'status': 'blocked_by_reader', 'answer': '', 'failure': str(exc)}
            except RuntimeError as exc:
                if 'generation retry limit exhausted' not in str(exc) and 'Gemini structural/transient retry limit exhausted' not in str(exc): raise
                value = {'status': 'failed', 'answer': '', 'failure': str(exc)}
        save(path, {'id': case['id'], 'method': method, **value})
    parallel(read, [(c, m) for c in cases for m in METHODS], 8, 'followup-read')


def score():
    verify(); cases = read_json(OUT / 'cases.json'); docs, gold = read_json(OLD / 'documents.json'), read_json(OLD / 'gold.json')
    spec = importlib.util.spec_from_file_location('official_followup', OLD / 'qasper_evaluator.py')
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    refs = official.get_answers_and_evidence({'paper': {'qas': [{'question_id': c['id'], 'answers': [{'answer': a} for a in gold[c['id']]['answers']]} for c in cases if c['dataset'] == 'qasper']}}, False)
    rows = []
    for case in cases:
        contexts = read_json(OUT / 'retrieval' / (signature(case['id']) + '.json'))
        for method in METHODS:
            prediction = read_json(OUT / 'predictions' / method / (signature(case['id']) + '.json'))
            context = contexts['methods'][method]
            row = {**prediction, 'doc_id': case['doc_id'], 'dataset': case['dataset'], 'spans': context['spans'],
                   'context_tokens': context['context_tokens'], 'context_sha256': __import__('hashlib').sha256(context['context'].encode()).hexdigest(),
                   'evidence_recall': None, 'complete_evidence': None, 'routing_status': contexts['searches'][method]['used_search']['status']}
            if case['dataset'] == 'quality': row['answer_score'] = int(row['status'] == 'ok' and row['answer'] == gold[case['id']]['label'])
            else:
                row['answer_score'] = max(official.token_f1_score(row['answer'], r['answer']) for r in refs[case['id']]) if row['status'] == 'ok' else 0
                eligible = [set(r['evidence']) for r in refs[case['id']] if r['evidence']]
                if eligible:
                    evidence = set(covered_paragraphs(docs[case['doc_id']], context['spans']))
                    row['evidence_recall'] = max(len(evidence & e)/len(e) for e in eligible)
                    row['complete_evidence'] = int(any(e <= evidence for e in eligible))
            rows.append(row)
    summary = {'questions': len(cases), 'predictions': len(rows), 'methods': {}, 'budget': read_json(ROOT / 'output/research-budget.json')}
    for dataset in ['qasper', 'quality']:
        summary['methods'][dataset] = {}
        for method in METHODS:
            part = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            summary['methods'][dataset][method] = {'n': len(part), **{k: statistics.mean([r[k] for r in part if r[k] is not None]) if any(r[k] is not None for r in part) else None for k in ['answer_score', 'evidence_recall', 'complete_evidence', 'context_tokens']}, 'empty_contexts': sum(r['context_tokens'] == 0 for r in part)}
    comparisons = rows + read_json(READER_V2 / 'scores.json'); intervals = []
    failed_ids = {r['id'] for r in comparisons if r['status'] != 'ok'}
    for candidate, baseline in [('EEE_bottom_up', 'EEE'), ('JJJ_bottom_up', 'JJJ'), ('JJJ_bottom_up', 'EEE_bottom_up'), ('JJJ_bottom_up', 'rerank_recursive'), ('JJJ_bottom_up', 'rerank_semantic'), ('JJJ_bottom_up', 'hybrid_full')]:
        for dataset in ['qasper', 'quality']:
            part = [{**r, 'method': 'jev_blocking_rerank' if r['method'] == candidate else 'comparison'} for r in comparisons if r['dataset'] == dataset and r['method'] in [candidate, baseline]]
            intervals.append({'factor': 'development-repair', 'candidate': candidate, 'baseline': baseline, 'dataset': dataset,
                'metrics': {k: paired_interval(part, k, 'comparison') for k in (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])},
                'common_unblocked_metrics': {k: paired_interval([r for r in part if r['id'] not in failed_ids], k, 'comparison') for k in (['answer_score', 'evidence_recall'] if dataset == 'qasper' else ['answer_score'])}})
    save(OUT / 'scores.json', rows); save(OUT / 'summary.json', summary); save(OUT / 'paired-intervals.json', intervals)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'score']); args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.command]()
