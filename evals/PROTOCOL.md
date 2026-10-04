# Pilot v1 — frozen development protocol

This is a small **synthetic development pilot**, not proof of general superiority. Documents, questions and labels were authored by the same assistant implementing the method. They have no independent annotation. Freeze this protocol, dataset, runtime source and runner hashes before the first live evaluation. Never tune thresholds against these results and call the same data held out.

## Dataset and measurements

- Six original documents, six paragraphs each, three sentences per paragraph. Two documents have one transition, four have two; two return to a previous topic. Headings identify documents but do not reveal internal boundaries.
- Four queries per document: literal, paraphrase, false-premise correction and context. A query uses its own wording as the requested content; no gold answer, topic or representative label is sent to Jev.
- Paragraph representative labels rotate first/middle/last equally. Gold is a single intended main-point sentence; plausible alternatives remain scoring disagreements and must be shown.
- Boundaries: exact paragraph-gap micro precision, recall and F1. Baselines: headings only (no internal cuts), fixed two-paragraph chunks, lexical Jaccard with the same boundary policy. Ablation: Jev with prior 0.5 versus default 0.7, reference prior 0.5, cutoff 0.5, minimum drop 0.2. Each method follows its own anchors; cached identical decisions may be reused.
- Representatives: paragraph exact-label agreement for full pipeline and first-sentence baseline. Budget-two outside-in selection replays the same scored paragraph candidates, exposing the budget tradeoff. This is not a second independent model run. Section selections are preserved for inspection, without inventing gold section accuracy.
- Retrieval: BM25-style ranking on lexical-built tree; BM25-style ranking on Jev-built tree; Jev reranking on the Jev-built tree. Every method searches all 18 leaves per document, and gets the same query. Thus reranker comparisons on the Jev tree isolate reranking; whole-pipeline comparisons include representation changes. Report hit@1, mean evidence recall@3, MRR and query-level results. No unanswerable queries: no abstention claim.
- Bottom-up context: take the top-ranked sentence and its immediate paragraph, then measure gold evidence coverage. This is a fixed expansion policy, **not an evaluated LLM agent**. Report read characters including the sentence reread within its paragraph. Do not count generated answers or LLM accuracy.
- Citation integrity: each delivered span must exactly match the original source and source hash. This is an extraction property, not semantic faithfulness of a generated answer.
- Uncertainty: 10,000 paired document-cluster bootstrap resamples with seed 1729 for Jev-minus-BM25 hit@1 on the same Jev tree. Only six clusters; descriptive interval, not a generalization guarantee.

## Efficiency probe

Use the first two paragraphs of the first document, all six paragraph-context candidates. Three repetitions, alternating which mode runs first. Fresh Jev clients avoid cache hits. Compare batch size 1 against 64 using `representatives()` in both cases. Record request counts, full client round-trip time, provider token usage, probability differences and winner agreement. Shared-state composition differs between modes, so quality equivalence is measured rather than assumed. This small sequential-versus-batch transport probe does not establish production throughput or the speedup of the entire tree builder.

## Frozen settings and limits

TypeSafe native endpoint, pinned `jev-1.13.0`; batch size 64; full outside-in representative search; no embedding model; no generative LLM. At most 220 HTTP attempts and 2,000 decision questions across the entire run; each state capped at 60,000 characters. No automatic retries or provider fallback. Stop and save partial results on errors. Report failed attempts and missing usage as missing, never zero. All live calls contain only this published synthetic corpus. Keys stay in the process environment or memory and are never archived.

The raw audit contains request **bodies**, validated response probabilities, timings and provider usage; authorization headers are never recorded. Preserve every result, including failures. Dollar cost is left unknown unless authoritative billing evidence is available. Freeze timestamps and SHA-256 manifests identify the exact implementation; replay verifies hashes before recomputing metrics without network access.
