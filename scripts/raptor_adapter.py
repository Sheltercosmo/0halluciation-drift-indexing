"""Run pinned RAPTOR core code with explicit embedding/summarization callbacks.

Core clustering, tree construction and collapsed retrieval run unchanged. We
load only upstream abstract provider interfaces so unused Torch/SBERT/QA models
are not imported. Default provider constructors deliberately fail closed.
"""
import ast
import importlib
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_raptor import verify

PACKAGE = '_zero_index_pinned_raptor'
DEFAULT_SOURCE = ROOT / 'output/vendor/raptor'


def load_upstream(source=DEFAULT_SOURCE):
    source = Path(source).resolve()
    verify(source)
    if PACKAGE in sys.modules:
        package = sys.modules[PACKAGE]
        if package.__path__ != [str(source / 'raptor')]:
            raise ValueError('A different RAPTOR checkout is already loaded')
    else:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(source / 'raptor')]
        sys.modules[PACKAGE] = package
        for filename, base, defaults in [
            ('EmbeddingModels', 'BaseEmbeddingModel', ['OpenAIEmbeddingModel', 'SBertEmbeddingModel']),
            ('SummarizationModels', 'BaseSummarizationModel', ['GPT3TurboSummarizationModel', 'GPT3SummarizationModel'])]:
            path = source / 'raptor' / (filename + '.py')
            original = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            selected = [node for node in original.body if
                        (isinstance(node, ast.ImportFrom) and node.module == 'abc') or
                        (isinstance(node, ast.ClassDef) and node.name == base)]
            if len(selected) != 2:
                raise ValueError('Unexpected upstream provider interface')
            module = types.ModuleType(PACKAGE + '.' + filename)
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), module.__dict__)
            def unavailable(self, *args, **kwargs):
                raise RuntimeError('Supply an explicit audited RAPTOR provider adapter')
            for name in defaults:
                setattr(module, name, type(name, (), {'__init__': unavailable}))
            sys.modules[module.__name__] = module
    names = ('cluster_tree_builder', 'tree_retriever', 'tree_structures', 'utils', 'EmbeddingModels', 'SummarizationModels')
    return types.SimpleNamespace(**{name: importlib.import_module(PACKAGE + '.' + name) for name in names})


def provider_adapters(upstream, embed, summarize, tokenizer):
    import numpy as np
    class Embedding(upstream.EmbeddingModels.BaseEmbeddingModel):
        def create_embedding(self, text):
            vector = np.asarray(embed(text), dtype=float)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                raise ValueError('Invalid RAPTOR embedding')
            return vector.tolist()
    class Summarizer(upstream.SummarizationModels.BaseSummarizationModel):
        def summarize(self, context, max_tokens=100):
            result = summarize(context, max_tokens)
            if not isinstance(result, str) or not result.strip():
                raise ValueError('Empty RAPTOR summary')
            if len(tokenizer.encode(result, disallowed_special=())) > max_tokens:
                raise ValueError('RAPTOR summary exceeds its registered token limit')
            return result
    return Embedding(), Summarizer()


def build_tree(text, embed, summarize, *, source=DEFAULT_SOURCE, seed=224,
               leaf_tokens=100, summary_tokens=100, layers=5):
    import numpy as np
    import numba
    import tiktoken
    from threadpoolctl import threadpool_limits
    upstream = load_upstream(source)
    enc = tiktoken.get_encoding('cl100k_base')
    embedding, summarizer = provider_adapters(upstream, embed, summarize, enc)
    config = upstream.cluster_tree_builder.ClusterTreeConfig(tokenizer=enc,
        max_tokens=leaf_tokens, num_layers=layers, summarization_length=summary_tokens,
        summarization_model=summarizer, embedding_models={'matched': embedding},
        cluster_embedding_model='matched', reduction_dimension=10,
        clustering_params={'threshold': .1, 'max_length_in_cluster': 3500})
    # Upstream UMAP uses NumPy's global RNG. Serial construction, a fixed RNG
    # state and one numerical thread avoid nondeterministic scheduling effects.
    np_state, py_state, threads = np.random.get_state(), random.getstate(), numba.get_num_threads()
    try:
        np.random.seed(seed)
        random.seed(seed)
        numba.set_num_threads(1)
        with threadpool_limits(limits=1):
            tree = upstream.cluster_tree_builder.ClusterTreeBuilder(config).build_from_text(text, use_multithreading=False)
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        numba.set_num_threads(threads)
    return tree


def retrieve(tree, query, embed, *, source=DEFAULT_SOURCE, context_tokens=2048):
    import tiktoken
    upstream = load_upstream(source)
    enc = tiktoken.get_encoding('cl100k_base')
    embedding, _ = provider_adapters(upstream, embed, None, enc)
    config = upstream.tree_retriever.TreeRetrieverConfig(tokenizer=enc,
        embedding_model=embedding, context_embedding_model='matched')
    reader = upstream.tree_retriever.TreeRetriever(config, tree)
    # Search every level as in upstream collapsed retrieval; rank by cosine and
    # stop at the first non-fitting node. There is no artificial top-10 cap.
    nodes, raw = reader.retrieve_information_collapse_tree(query, len(tree.all_nodes), context_tokens)
    selected = list(nodes)
    context = raw
    # Upstream counts node text but omits separators. Enforce the shared budget
    # on the actual rendered context, dropping whole nodes and preserving rank.
    while len(enc.encode(context, disallowed_special=())) > context_tokens and selected:
        selected.pop()
        context = upstream.utils.get_text(selected)
    return {'context': context, 'context_tokens': len(enc.encode(context, disallowed_special=())),
            'selected_nodes': [n.index for n in selected],
            'upstream_selected_nodes': [n.index for n in nodes],
            'separator_budget_drops': len(nodes)-len(selected)}


def tree_record(tree):
    """Portable JSON; no pickle execution is needed to replay stored trees."""
    return {'num_layers': tree.num_layers,
            'nodes': [{'id': n.index, 'text': n.text, 'children': sorted(n.children),
                       'embeddings': n.embeddings} for _, n in sorted(tree.all_nodes.items())],
            'layers': {str(k): [n.index for n in v] for k, v in sorted(tree.layer_to_nodes.items())},
            'roots': sorted(tree.root_nodes), 'leaves': sorted(tree.leaf_nodes)}


def restore_tree(record, source=DEFAULT_SOURCE):
    upstream = load_upstream(source)
    Node, Tree = upstream.tree_structures.Node, upstream.tree_structures.Tree
    nodes = {r['id']: Node(r['text'], r['id'], set(r['children']), r['embeddings']) for r in record['nodes']}
    return Tree(nodes, {i: nodes[i] for i in record['roots']}, {i: nodes[i] for i in record['leaves']},
                record['num_layers'], {int(k): [nodes[i] for i in ids] for k, ids in record['layers'].items()})
