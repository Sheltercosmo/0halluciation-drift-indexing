# Standard evidence retrieval

The standard is **Jev traversal, bidirectional pairwise ranking and shared-context selection**. The retained measured version achieves 91.45% full-Jev evidence Recall@5 and 92.44% hybrid Recall@5 on 640 eligible historical QASPER questions. The [comparison report](../evals/RETRIEVAL_STANDARD_REPORT.md) gives the full population, uncertainty and limitations.

The authoritative policy is [retrieval-standard.json](../configs/retrieval-standard.json). The implementation is [`scripts/retrieval_standard.py`](../scripts/retrieval_standard.py). Later experimental selection methods are separate from this entrypoint.

## How it retrieves evidence

1. Use the Jev split/central-sentence index and native heading hierarchy. A query-time LLM proposes evidence needs; the original question is also a routing request.
2. Jev evaluates children of promising nodes from root to paragraphs. The measured beam is five, acceptance threshold 0.2, and global routing ceiling 4,096 decisions. Accepted deferred branches receive up to 25% additional decisions relative to the initial traversal, within that global ceiling.
3. Full Jev uses tree candidates. The hybrid fuses them with an independent direct-embedding ranking. Embedding retrieval does not navigate the tree.
4. Compare the first 30 candidate paragraphs with Jev in both orientations. Consistent preferences contribute a win and conflicting preferences tie. Read complete source paragraphs, not only central sentences.
5. Assemble the first 12 pairwise-ranked paragraphs as selectable targets. A shared packet includes complete targets, nearby source introductions, preceding/following paragraphs, headings and source links. Context supports interpretation; only the 12 targets are eligible to be returned.
6. Jev scores every target while viewing the shared packet in source order and reverse source order. Average the two scores and return the top five complete paragraphs. Exact ties retain earlier pairwise order.

Jev evaluates useful evidence for the original question, not answerability. Useful corroboration and restatements are not penalized for overlap. There is no Bayesian or weighted position prior in ranking. The statistical prior remains confined to topic-boundary detection during indexing.

## Use the standard pipeline

Run from the repository root. Supply the prepared source document, matching Jev index, evidence needs and the three Jev callbacks:

```python
from scripts.retrieval_standard import retrieve_standard, make_standard_clients

# cache is a pathlib.Path; budget exposes reserve("jev_calls", questions=...).
# The evaluation's RetrievalBudget can supply that accounting interface.
router, pairwise, shared = make_standard_clients(cache, budget)

result = retrieve_standard(
    document, index, question, evidence_needs,
    router.route_content,
    pairwise.compare,
    shared.score_pool,
)

evidence_paragraph_ids = result["selected"]
```

The document uses the evaluation adapter's `title`, `text` and `units` fields; each unit has a paragraph number, section, heading and exact start/end offsets. `index.source` must equal `document["text"]`. `scripts.bounded_eval.build_document` constructs this representation from native sections. Paragraph IDs are `p0`, `p1`, and so on.

For the hybrid, pass `dense_ranking=paragraph_ids` from an independent embedding retriever over those same original paragraphs. The measured run used the saved direct Gemini hit list, with at most 80 hits, before fusion and the common top-30 pairwise stage. Omitting that argument selects full Jev.

`make_standard_clients` reads `TYPESAFE_API_KEY` from the environment and uses separate adapters for routing, pair comparison and shared selection. These callbacks incur Jev API usage. Set an explicit budget through the supplied accounting object. Source text and paragraph boundaries remain intact.

If upstream pairwise ranking is already cached, call `select_standard_evidence(document, question, pairwise_ranking, shared.score_pool)`. This reuses the retained standard selector without rerunning traversal, embeddings or pairwise decisions.

## Verification

```sh
python -m unittest discover -s tests -v
python scripts/replay_standard_results.py
```

The archive checker uses saved source packets and decisions to reconstruct every final selection, verifies all 18 historical baseline rows are unchanged, and reproduces aggregate evidence scores. The standard entrypoint was also replayed through traversal, pairwise ranking and shared selection for all 1,456 full-Jev/hybrid predictions without new inference; every stage matched the retained run.

These are previously inspected historical questions, not untouched confirmation. Shared-context selection has higher observed aggregate scores than its pairwise controls, but those incremental gains are not statistically established. The standard is the retained measured policy, not a guarantee of optimal evidence or error-free retrieval.
