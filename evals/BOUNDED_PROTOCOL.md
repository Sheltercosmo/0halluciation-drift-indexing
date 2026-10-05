# Bounded comparison on hard and evidence-dense questions

Registered configuration: 2026-10-05. This study evaluates decision-based topic blocking, including its statistical prior and probability-drop cut rule, as one method. It does not test the prior separately or claim calibrated probabilities. The [research audit](FRONTIER_EVALUATION.md) explains benchmark choices; running every downloaded benchmark is not required for this bounded study.

## Questions and controls

Select **384 questions** before inference, with seed 20261005: 192 QuALITY development HARD questions and 192 QASPER test questions (160 with at least two evidence passages in an annotation, 16 single/zero-evidence answerable cases, and 16 unanimously unanswerable cases). Mixed-answerability questions remain eligible for answerable strata. Within each stratum, shuffle IDs deterministically and first spread across documents, then permit a second question; at most two selected questions per document overall. Save exact IDs, source hashes and code before model calls.

This intentionally enriched sample is not an estimate of either full benchmark's average. QuALITY-HARD is human-defined difficulty, not a guarantee against current-model saturation. No model-based difficulty filter, answer-based retry, post-hoc replacement or optional stopping is permitted. Native titles/headings and source paragraphs are shared across methods. QASPER includes abstract, body and captions, but not image content. No gold answers or evidence labels enter indexing, retrieval, reranking or reader prompts.

## Four pipelines

| Pipeline | Chunking | Retrieval / reranking |
| --- | --- | --- |
| hybrid_recursive | Greedy paragraph packing within native sections; split oversized paragraphs at exact UTF-8-safe token boundaries | BM25 + dense RRF |
| hybrid_semantic | Adjacent paragraph cosine distance; cut above the within-section 85th percentile, then common size packing | Same hybrid retrieval |
| hybrid_codex_rerank | Same recursive chunks | Hybrid top 12, then Codex listwise reranking |
| jev_blocking_rerank | Fixed-anchor Jev topic decisions with the repository's prior/drop rule, then common size packing | Hybrid top 12, then native Jev relevance decisions |

These are implemented pipelines, not reproductions of RAPTOR, PageIndex, HiChunk or a leaderboard. Both chunk boundaries and reranker differ in the combined Jev arm: outcomes support a pipeline comparison, not a causal claim about either component alone. The prior belongs to the cut rule and has no separate validation arm.

Maximum chunk size is **512 cl100k_base tokens**, with no overlap for all methods. Each answer receives at most **2,048 retrieved tokens**, including title, heading and passage wrappers. Pack ranked chunks greedily, skipping chunks that do not fit; render chosen passages in source order. Reranked arms select only from their top-12 pool. Record source offsets and fully covered source paragraphs, not fuzzy text matches.

Dense model: gemini-embedding-2, 768 dimensions, L2 normalized. Retrieval document prefixes use title/text; query prefixes use question-answering; semantic boundaries use the separate sentence-similarity prefix. BM25 uses k1=1.2, b=0.75; RRF constant 60. Multiple-choice options accompany questions for all retrievers and readers. No benchmark-specific parameter tuning is performed.

Jev model: native TypeSafe jev-1.13.0. Within each existing section, compare the anchor paragraph to subsequent paragraphs, fetching up to eight independent candidate decisions at a time. Apply repository defaults (target prior 0.7, reference prior 0.5, cutoff 0.5, minimum drop 0.2), cut when both cutoff and drop tests hold, and reset the anchor at each cut. Prefetched decisions for an old anchor are discarded and counted in usage. Native headings require no Jev inference. Index construction never receives a question. Representative selection, full tree traversal and query proposals are outside this comparison.

Codex model: gpt-6.1-sol, low reasoning effort, through the authenticated CLI with user configuration ignored, an ephemeral read-only working directory and a strict JSON output schema. All arms use the exact same reader prompt. Batch up to eight independent source documents per reader call and four per reranking call; never place two questions about one document in a batch. Reader batch grouping is shared across arms. Calls receive no gold answers or method labels. Reject calls that invoke tools. Candidate order is deterministically shuffled for Codex listwise reranking. Reader outputs must be A/B/C/D for QuALITY, or a brief source-grounded answer / Unanswerable for QASPER.

## Scores and uncertainty

- QuALITY-HARD accuracy; invalid answers score zero.
- QASPER official maximum-over-annotations answer token F1, retaining unanswerable cases.
- QASPER retrieved-context paragraph F1 using the official paragraph scorer. This is retrieval coverage, not reader-selected evidence F1.
- Maximum recall and complete recovery against each nonempty reference evidence set; report the eligible question count. Keep visual evidence references; text-only retrieval can miss them.
- Paired Jev-minus-baseline differences with 10,000 whole-document bootstrap replicates and descriptive 95% intervals, separately for each dataset. No pooled score or multiplicity-adjusted superiority claim.
- Question counts, document counts, invalid outputs, source spans, context tokens, indexing/query usage, request failures, latency and budget reservations.

An answer can be correct while its evidence is missing; report both. Publish unfavorable comparisons and ceiling effects. A provider/budget interruption leaves the registered run incomplete, with its original denominator. Resume identical cached requests; never replace a difficult case. Code fixes before any predictions are allowed with a fresh manifest; changes after predictions require a recorded amendment and a new comparable run.

## Budget and reproducibility

Gemini is used only for embeddings. Enforce the user's **$30 total cap** with a persistent reservation before every attempted request, including retries. Reserve $4 for earlier Gemini work. Count each batch with the provider tokenizer, add 32 tokens per input as a buffer, and reserve at the verified standard price of $0.20 per million input tokens. The ledger is a conservative reservation, not an invoice. Reject individually oversized inputs rather than silently truncate. Stop if the cap is insufficient.

Codex CLI calls use the user's existing account allowance; they are not an unlimited free API. Cap the run at 450 calls, with two concurrent calls. Jev has separate request/question caps recorded in the ledger. All providers use content-addressed caches; store secrets only in process environment variables, never result artifacts.

Sources: [Gemini embeddings and task prefixes](https://ai.google.dev/gemini-api/docs/embeddings), [pricing](https://ai.google.dev/gemini-api/docs/pricing), [native token counting](https://ai.google.dev/api/tokens), [Codex non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode).

~~~sh
python -m pip install -r evals/requirements-bounded.txt
python scripts/bounded_eval.py prepare
# Set TYPESAFE_API_KEY, GEMINI_API_KEY and CODEX_EVAL_BINARY in the process environment.
python scripts/bounded_eval.py run
python scripts/bounded_eval.py score
~~~

The manifest verifies source and input hashes before inference and scoring. Raw benchmark documents, API caches and provider credentials stay in ignored output/. Public artifacts contain selection IDs, configuration, source hashes, predictions, passage offsets, aggregate scores and sanitized usage. No performance claim follows from this protocol until the registered comparison finishes.
