# Configurable evidence retrieval

**EEJ is the deployment default:** embeddings split topic blocks and select central sentences; Jev explores the tree, reranks complete paragraphs, and selects evidence in shared context. Letters always mean **splitting / central sentences / search**. JJJ remains available for indexing without embeddings. EJJ and JEJ let you change one indexing component at a time.

The [earlier controlled comparison](../evals/RETRIEVAL_V4_REPORT.md) measured EEJ at **87.87%** and JJJ at **87.56%** Recall@5 with its common final selector. EEJ avoids Jev decisions in both indexing stages. The later shared-context scores, **91.45% JJJ** and **92.44% JJJ + direct embeddings**, remain historical JJJ measurements; EEJ with the later improvements has not been evaluated. We do not transfer those scores to the new default or claim a measured end-to-end cost reduction.

## One adapter for indexing and retrieval

Install with `python -m pip install .` from the checkout, then import the complete pipeline from `zero_index`. See the [local/API quick start](local-and-api.md) for installed CLI commands and all provider choices. The [configuration](../configs/retrieval-standard.json) is executable input, with validated fields and JSON round-tripping.

```python
from dataclasses import replace
from zero_index import RetrievalAdapter, RetrievalConfig, load_document
from zero_index.providers import OllamaEmbeddings, JevAPI

config = RetrievalConfig.load("configs/retrieval-standard.json")  # EEJ
config = config.with_search(beam=3, acceptance=0.25, max_decisions=2048)
config = replace(config, indexing=replace(config.indexing, paragraph_embedding_prefix=""),
                 pairwise_candidates=20, shared_targets=10)
config.save("my-retrieval.json")

adapter = RetrievalAdapter.from_models(
    OllamaEmbeddings("embeddinggemma"),
    JevAPI(api_key_env="TYPESAFE_API_KEY"),
    config=config,
)
document = load_document("document.md")
index = adapter.build_index(document)
result = adapter.retrieve(document, index, question, evidence_needs)
evidence_paragraphs = result["paragraphs"]
```

The lower-level callback constructor remains available for custom task APIs. The implementation is [`zero_index/standard.py`](../zero_index/standard.py); the older `scripts.retrieval_standard` imports are retained for experiment compatibility.

`embed(texts, purpose)` must return one finite, nonzero vector per text, in input order; vectors are normalized by the adapter. `purpose` is `splitting` or `central-sentences`. Adapt your embedding SDK to this small interface and handle its batching, caching and billing limit there. The historical paragraph prefix is configurable as `indexing.paragraph_embedding_prefix`; set it to `""` when your embedding provider supplies its own task instruction. Central sentences are embedded without that prefix. No provider or model is silently substituted.

`document` has `id`, `title`, `text` and `units`. Each unit has a sequential `paragraph` number, a native `section` number, `heading`, and exact `start`/`end` character offsets. Section occurrences have distinct IDs. `zero_index.documents.build_document(id, title, [(heading, paragraphs), ...])` constructs this representation. Use `A ::: B` for native nested heading paths. Paragraph IDs stay `p0`, `p1`, and so on; paragraphs are returned whole.

To use JJJ, load [`retrieval-jjj-measured.json`](../configs/retrieval-jjj-measured.json) and pass a decision backend to `RetrievalAdapter.from_models`. The lower-level constructor accepts an explicit indexing scorer as `jev=`. No embedding callback is required for JJJ. Changing splitting or central-sentence controls requires rebuilding the index. The adapter rejects mismatched index factors and recorded indexing settings before making retrieval calls.

For the **hybrid**, pass `dense_ranking=paragraph_ids` to `adapter.retrieve`. This is an independent direct embedding ranking over the same source paragraphs, fused with Jev candidates before the common final selector. It does not change tree routing. `dense_candidates=null` preserves the supplied ranking, as in the retained historical pipeline; an explicit integer caps it before fusion. Unknown or duplicate IDs are rejected before any Jev retrieval call.

## Search effort

```python
config = RetrievalConfig.for_effort("low")        # default variant remains EEJ
config = RetrievalConfig.for_effort("standard")   # retained search/selection settings
config = RetrievalConfig.for_effort("high", variant="JJJ")
```

| Preset | Beam per need/layer | Routing decision ceiling | Deferred allowance | Pairwise candidates | Shared targets | Output paragraphs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 3 | 1,024 | 10% | 16 | 8 | 5 |
| standard | 5 | 4,096 | 25% | 30 | 12 | 5 |
| high | 8 | 8,192 | 50% | 40 | 16 | 5 |

These presets are resource choices, **not validated accuracy tiers**. Every preset retains accepted-branch revisiting, bidirectional pairwise comparisons, shared source context, and both presentation orders. More effort does not guarantee better retrieval.

`search.max_decisions` covers the initial and deferred traversal together, across all search needs. Deferred work receives at most `ceil(deferred_search_fraction × initial_decisions)` within that ceiling. A complete next layer can exceed the remaining allowance; the search then stops and records truncation instead of exceeding the cap. `search.beam` restricts internal nodes, not paragraph leaves.

Pairwise work can dominate query cost: comparing `n` paragraphs in both orientations uses `n × (n − 1)` decisions, followed by at most `2 × shared_targets` shared decisions. With 30 candidates this is 870 pairwise decisions; 16 candidates use 240. `config.decision_ceiling()` reports a query bound of routing + pairwise + shared decisions. It excludes indexing, embedding calls, planning, retries and provider billing. **Decisions are not HTTP requests or dollars**; enforce those limits in the provider/accounting adapters separately.

## Decision boundaries and other controls

| Setting | Default | Effect |
| --- | ---: | --- |
| `search.acceptance` | 0.2 | Minimum Jev score for keeping an internal branch. Paragraph evidence is not discarded by this threshold. |
| `search.refine_below` | 0.85 | Open additional source cues and rescore uncertain internal nodes before pruning, within the routing budget. Must be at least acceptance. |
| `search.child_cues` | 6 | Child representative cues shown on upper-node cards. |
| `search.extra_sentences` | 8 | Additional outside-in sentences in detailed source cards. |
| `pairwise_threshold` | 0.5 | A wins only if `P(A > B) > threshold` and `P(B > A) < 1 − threshold`; uncertainty or orientation disagreement ties. Range 0.5–1. |
| `shared_targets` | 12 | Eligible targets in the shared source packet. Context paragraphs help interpretation but do not become extra output slots. |
| `output_paragraphs` | 5 | Whole paragraphs selected. Require output ≤ shared targets ≤ pairwise candidates. Changing this changes the evaluation's @k. |
| `dense_candidates` | `null` | Optional cap on direct embedding hits before hybrid fusion. `null` preserves all supplied hits; final Jev work is still bounded by pairwise/shared limits. |
| `indexing.embedding_split_quantile` | 0.85 | E splitting cuts when adjacent-paragraph cosine distance exceeds this quantile within a native heading. Higher values usually produce fewer cuts; ties do not cut. |
| `indexing.same_topic_prior` | 0.7 | J splitting's explicit same-topic prior. |
| `indexing.posterior_cutoff` | 0.5 | J splitting requires adjusted same-topic probability at or below this value. |
| `indexing.minimum_drop` | 0.2 | J splitting also requires a probability drop at least this large. |
| `indexing.sentence_budget` | 8 | Outside-in central-sentence candidates per target; `null` searches all. |
| `indexing.sentence_stop_threshold` | `null` | Optional early stop, for example 0.9, after a candidate wave reaches this score. |

The prior, cutoff and drop apply only to **J splitting**. E splitting uses the distance quantile. Embedding centrality and Jev decision scores are different quantities; the same early-stop value is not a calibrated equivalence. Embedding centrality still needs all source-sentence vectors for its centroid even when candidate scoring stops early. A finite candidate budget can miss a better central sentence.

`indexing.probability_reference_prior` defaults to 0.5 and specifies the reference-prior assumption for J splitting's probability adjustment. None of these settings becomes a Bayesian ranking prior. Final selection keeps the original question, whole paragraphs, source links, forward/reverse source presentations, mean score aggregation, and earlier pairwise order only for exact ties. Overlap is not penalized and Jev does not classify answerability.

## Results and reproducibility

Every retrieval result includes `variant`, the full effective `configuration`, its fingerprint, source-backed selections, and traversal/selection traces. Unknown configuration keys fail explicitly. Indexes record their factors, indexing controls, embedding identity, boundaries and central-sentence provenance.

The [historical score archive](../evals/RETRIEVAL_STANDARD_REPORT.md) and its frozen source snapshots are unchanged. Use `MEASURED_JJJ_CONFIG` or `retrieval-jjj-measured.json` to replay that pipeline. Do not label a tuned or EEJ run with the historical JJJ scores.

```sh
python -m unittest discover -s tests -v
python scripts/replay_standard_results.py
```

The archive checker reconstructs all 1,456 saved selections and verifies unchanged baselines and aggregate scores without inference. These historical questions were previously inspected; they are not untouched confirmation. The small shared-context gains over pairwise controls are not statistically established.
