"""Second development iteration: fill from successive source ancestors."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.bounded_eval import read_json, sha, tokenizer
from scripts.live_tree_clients import Gemini, LiveBudget, GenerationBlocked
from scripts.live_tree_eval import OUT as BASE, parallel
from scripts.live_tree_reader_v2 import OUT as READER, read_case
from scripts.live_tree_followup import OUT as FIRST
from zero_index.index import DocumentIndex
from zero_index.hybrid_retrieval import tree_passages, fuse_retrieval_paths

OUT = ROOT / 'output/live-tree-ancestor-v1'
METHODS = ['EEE_ancestor', 'JJJ_ancestor']


def ancestor_context(index, search, initial, count, budget=2048):
    """Keep prior evidence, then expand by ancestor level and source proximity."""
    ranked = [{'start': a, 'end': b, 'text': index.source[a:b]} for a, b in initial['spans']]
    seen = {(r['start'], r['end']) for r in ranked}
    cores = [index._node(c['node_id']) for c in tree_passages(index, search)]
    ancestors = [index._node(index._parents[n.node_id]) for n in cores]
    def add(node):
        span = (node.start, node.end)
        if span not in seen:
            seen.add(span); ranked.append({'start': node.start, 'end': node.end, 'text': index.source[node.start:node.end]})
    # Each round moves one level toward the root. Within a level, interleave
    # paragraphs by distance to the independently retrieved cores. No new model
    # scores or embedding retrieval are used to find these adjacent passages.
    while ancestors:
        queues = []
        for core, parent in zip(cores, ancestors):
            paragraphs = [n for n in parent.walk() if n.kind == 'paragraph']
            queues.append(sorted(paragraphs, key=lambda n: (max(0, n.start-core.end, core.start-n.end), n.start)))
        for i in range(max(map(len, queues), default=0)):
            for queue in queues:
                if i < len(queue): add(queue[i])
        if all(n.node_id == index.root.node_id for n in ancestors): break
        ancestors = [index._node(index._parents[n.node_id]) if n.node_id != index.root.node_id else n for n in ancestors]
    return fuse_retrieval_paths(index.source, ranked, [], token_count=count, title=index.root.title, budget=budget)


def freeze():
    if (OUT / 'manifest.json').exists(): raise ValueError('Already frozen')
    cases = read_json(FIRST / 'cases.json')
    assert len(list((FIRST / 'retrieval').glob('*.json'))) == len(cases)
    paths = ['scripts/live_tree_ancestor.py', 'scripts/live_tree_reader_v2.py', 'scripts/live_tree_clients.py']
    for name in paths:
        p = OUT / 'source' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT / name).read_bytes())
    save(OUT / 'manifest.json', {'status': 'frozen_before_second_followup_inference', 'partition': 'exposed_development_only',
        'methods': METHODS, 'questions': len(cases), 'source_hashes': {n: sha(ROOT / n) for n in paths},
        'first_followup_manifest_sha256': sha(FIRST / 'manifest.json'),
        'first_followup_retrieval_sha256': {p.name: sha(p) for p in (FIRST / 'retrieval').glob('*.json')},
        'motivation': 'Completed first-followup retrieval still supplies only 678 mean QASPER tokens for Jev. This second policy is frozen before inspecting first-followup QA aggregates.',
        'policy': 'Reuse exactly the first follow-up searches. Preserve all first-followup spans first. Move through topic, heading and root ancestors; interleave whole paragraphs nearest each reached core at every ancestor level. Exact source union within 2048 tokens.',
        'reader': 'Same QASPER reader and corrected labelled-options MCQ reader. One final read, deterministic source expansion, no interactive reader search.',
        'compute': 'No additional routing or embedding calls beyond first follow-up. All candidate paragraphs examined locally; extra reader calls audited.',
        'selection': 'Publish both embedding and Jev arms. Keep both prior iterations. No test or validation access.'})


def verify():
    m = read_json(OUT / 'manifest.json')
    assert sha(FIRST / 'manifest.json') == m['first_followup_manifest_sha256']
    for n, digest in m['source_hashes'].items(): assert sha(ROOT / n) == digest, n
    for name, digest in m['first_followup_retrieval_sha256'].items(): assert sha(FIRST / 'retrieval' / name) == digest
    return m


def run():
    verify(); cases = read_json(FIRST / 'cases.json'); budget = LiveBudget(ROOT / 'output/research-budget.json')
    reader = Gemini(OUT, budget); reader.cache = BASE / 'gemini-cache'; reader.responses = BASE / 'gemini-responses'
    mc = Gemini(OUT, budget); mc.cache = READER / 'gemini-cache'; mc.responses = READER / 'gemini-responses'
    enc = tokenizer(); count = lambda s: len(enc.encode(s, disallowed_special=()))
    blocked = {r['id'] for r in read_json(FIRST / 'scores.json') + read_json(READER / 'scores.json') if r['status'] == 'blocked_by_reader'}
    def process(case):
        name = signature(case['id']) + '.json'; original = read_json(FIRST / 'retrieval' / name)
        path = OUT / 'retrieval' / name
        if path.exists(): retrieved = read_json(path)
        else:
            retrieved = {'id': case['id'], 'methods': {}}
            for method in METHODS:
                previous = method[:3] + '_bottom_up'
                if case['id'] in blocked: context = {'context': '', 'context_tokens': 0, 'spans': []}
                else:
                    index = DocumentIndex.from_dict(read_json(BASE / 'indexes' / (signature(case['doc_id']) + '-' + method[:2] + '.json')))
                    context = ancestor_context(index, original['searches'][previous]['used_search'], original['methods'][previous], count)
                retrieved['methods'][method] = context
            save(path, retrieved)
        for method in METHODS:
            path = OUT / 'predictions' / method / name
            if path.exists(): continue
            if case['id'] in blocked: value = {'status': 'blocked_by_reader', 'answer': '', 'failure': 'Known reader block; no new request'}
            else:
                try: value = {'status': 'ok', 'answer': read_case(mc if case['options'] else reader, case, retrieved['methods'][method]['context'])}
                except GenerationBlocked as exc: value = {'status': 'blocked_by_reader', 'answer': '', 'failure': str(exc)}
            save(path, {'id': case['id'], 'method': method, **value})
    parallel(process, cases, 8, 'ancestor-read')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['freeze', 'run']); a = p.parse_args()
    {'freeze': freeze, 'run': run}[a.command]()
