"""Offline RAPTOR adapter integration checks, not a benchmark result."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save, signature
from scripts.raptor_adapter import build_tree, load_upstream, provider_adapters, restore_tree, retrieve, tree_record


def check(output):
    import numpy as np
    import tiktoken
    logging.getLogger().setLevel(logging.WARNING)
    enc = tiktoken.get_encoding('cl100k_base')
    calls = {'embedding': 0, 'summary': 0}
    def embed(text):
        calls['embedding'] += 1
        raw = hashlib.sha512(text.encode()).digest()
        vector = np.asarray(list(raw), dtype=float)-127.5
        return (vector/np.linalg.norm(vector)).tolist()
    def summarize(text, limit):
        calls['summary'] += 1
        return enc.decode(enc.encode(text, disallowed_special=())[:limit])
    text = '\n'.join(f'Report {i} describes {topic} observations using repeated measurements '
        'from several instruments to compare distant and nearby sources and explain measured differences.'
        for i, topic in enumerate(['astronomy', 'forestry', 'materials', 'oceanography']*24))
    first = build_tree(text, embed, summarize)
    first_record = tree_record(first)
    second = build_tree(text, embed, summarize)
    assert tree_record(second) == first_record, 'Fixed-seed tree did not reproduce'
    assert len(first.leaf_nodes) > 11 and len(first.all_nodes) > len(first.leaf_nodes), 'Clustering/summarization was not exercised'
    assert calls['summary'] > 0
    rebuilt = restore_tree(first_record)
    assert tree_record(rebuilt) == first_record, 'JSON tree replay changed the topology'
    upstream = load_upstream()
    embedding, _ = provider_adapters(upstream, embed, summarize, enc)
    cfg = upstream.tree_retriever.TreeRetrieverConfig(embedding_model=embedding, context_embedding_model='matched')
    original = upstream.tree_retriever.TreeRetriever(cfg, first)
    nodes, context = original.retrieve_information_collapse_tree('nearby astronomy measurements', len(first.all_nodes), 2048)
    adapted = retrieve(first, 'nearby astronomy measurements', embed)
    assert adapted['upstream_selected_nodes'] == [n.index for n in nodes]
    assert adapted['selected_nodes'] == [n.index for n in nodes[:len(adapted['selected_nodes'])]]
    assert adapted['context_tokens'] <= 2048
    assert retrieve(rebuilt, 'nearby astronomy measurements', embed) == adapted
    # Show the actual rendered separator budget, which upstream omits.
    Node, Tree = upstream.tree_structures.Node, upstream.tree_structures.Tree
    tiny_nodes = {i: Node(word, i, set(), {'matched': [1.,0.]}) for i, word in enumerate(['apple', 'banana', 'orange'])}
    tiny = Tree(tiny_nodes, tiny_nodes, tiny_nodes, 0, {0:list(tiny_nodes.values())})
    bounded = retrieve(tiny, 'fruit', lambda _: [1.,0.], context_tokens=3)
    assert bounded['separator_budget_drops'] > 0 and bounded['context_tokens'] <= 3
    try:
        upstream.EmbeddingModels.OpenAIEmbeddingModel()
    except RuntimeError:
        pass
    else:
        raise AssertionError('An unaudited default provider was instantiated')
    relevant = ['numpy','scipy','scikit-learn','umap-learn','numba','llvmlite','pynndescent','tiktoken','openai','tenacity']
    report = {'status':'offline_adapter_checks_passed_not_qa_performance', 'external_model_calls':0,
        'upstream_commit':'7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767',
        'runtime':{name:importlib.metadata.version(name) for name in relevant},
        'fixture':{'source_tokens':len(enc.encode(text)), 'leaves':len(first.leaf_nodes),
                   'nodes':len(first.all_nodes), 'layers':first.num_layers, 'tree_sha256':signature(first_record)},
        'checks':['all upstream source blobs verified', 'real upstream UMAP/GMM and tree builder exercised',
                  'fixed-seed tree reproduced', 'JSON topology replay exact', 'collapsed ranking matches upstream',
                  'rendered source budget enforced', 'default unaudited providers rejected'],
        'synthetic_callback_invocations':calls}
    save(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/raptor-adapter-check.json')
    args = parser.parse_args()
    check(args.output)
