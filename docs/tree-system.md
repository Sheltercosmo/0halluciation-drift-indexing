# Root-to-leaf retrieval and an independent dense path

The query LLM proposes evidence needs. Jev or embeddings then route through the tree from its root to sentence leaves. The final reader receives original source passages. The hybrid combines a complete Jev retrieval path with a separate direct embedding search; it does not mix scoring methods inside the tree.

## Compare central sentences on one fixed tree

```python
from zero_index import CentroidRepresentatives, JevScorer, build_index, reselect_representatives

jev = JevScorer(provider="typesafe")
# Your callback supplies the embedding model; none is loaded implicitly.
centroid = CentroidRepresentatives(embed_sentence, model_name="your-embedding-model")
jev_tree = build_index(source, scorer=jev)
embedding_representatives = reselect_representatives(jev_tree, centroid, sentence_budget=8)
jev_representatives = reselect_representatives(jev_tree, jev, sentence_budget=8)
```

The two copies have identical splits, node IDs, source offsets and native headings. Only central sentences and their selection metadata change. Embedding centrality is exact average cosine similarity to the other source sentences, computed with a vector sum. Jev judges how well a sentence represents its full node context. Both use the same outside-in candidate order and budget.

`build_index` also accepts `representative_scorer` separately from its splitting `scorer`. Embedding-based segmentation comparisons must specify their partition algorithm; changing representatives does not change boundaries.

## Share the LLM proposal and compare tree search

```python
from zero_index import EmbeddingTreeRouter, TreeSearchConfig, propose_needs, search_tree

# Call the LLM once and preserve this plan for every compared method.
needs = propose_needs(jev_tree, question, call_your_planner)
limits = TreeSearchConfig(beam_width=2, max_node_scores=256, max_preview_tokens=8192)
dense_router = EmbeddingTreeRouter(
    embed_document, embed_query=embed_query, model_name="your-embedding-model",
)
jev_result = search_tree(jev_representatives, question, needs, jev,
                         token_count=count_tokens, config=limits)
dense_result = search_tree(jev_representatives, question, needs, dense_router,
                           token_count=count_tokens, config=limits)
```

`call_your_planner(state)` returns one to three strings, at most 300 characters each. It sees the original question, document title and native headings, without gold labels or method-specific representatives. Include multiple-choice options in `question` when needed. `count_tokens(text)` must use the final reader's tokenizer and return a nonnegative integer.

Each search begins at the root and ranks only children of selected nodes. Previews contain heading paths and extractive central sentences, or the exact leaf sentence. Full unvisited subtrees are not sent to the router. Whole rounds are checked against shared budgets before scoring; reaching a limit returns an explicit status. Traces include every candidate preview, score and selected branch. Jev routing is a dedicated decision task; it does not pretend a heading already contains an answer.

## Combine Jev retrieval with direct embedding retrieval

```python
from zero_index import EmbeddingPassageRetriever, fuse_retrieval_paths, tree_passages

direct = EmbeddingPassageRetriever(
    embed_document, embed_query=embed_query, model_name="your-embedding-model",
)
# canonical_chunks covers the whole source, independently of visited tree paths.
# Each item has start, end, exact text, and optionally a heading.
dense_ranking = direct.retrieve(source, question, needs[0], canonical_chunks, limit=64)
jev_ranking = tree_passages(jev_representatives, jev_result)
hybrid = fuse_retrieval_paths(
    source, jev_ranking, dense_ranking, title=jev_tree.root.title,
    token_count=count_tokens, budget=2048,
)
answer = call_your_reader(question, hybrid["context"])
```

The example uses one dense evidence need. For a multi-need experiment, retrieve for every shared need and freeze a deterministic, deduplicating merge rule before comparing arms. Direct retrieval scores every supplied chunk; its `limit` truncates results after ranking. It can return passages in branches that Jev never visited. Embedding callbacks may precompute vectors in batches and serve the cached vectors here.

The merger applies equal-weight reciprocal-rank fusion with constant 60, deduplicates exact spans, and packs the source union under one final budget. Overlap is counted once and gaps are not filled. Its return value includes source spans, each path's ranks, selected and skipped candidates, and actual rendered token count. Use this same renderer with one path empty for matched Jev-only and dense-only comparisons. Original source spans provide citations; neither route scores nor representative previews establish answer correctness.

The earlier `retrieve()` API retains its leaf-first, upward-reading behavior for existing callers; `search_tree()` implements the new root-to-leaf direction. No API silently changes retrieval strategy. The [component and whole-system protocol](../evals/TREE_SYSTEM_PROTOCOL.md) defines the research controls; live performance for these new APIs is not yet measured.
