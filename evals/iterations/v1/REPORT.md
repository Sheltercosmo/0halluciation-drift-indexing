# Improvement log: development screens, not held-out results

The improvement goal is active. **Frontier performance has not been established.** Validation and test are still locked. The prior and sudden-drop rule have not been tested separately.

## Data separation

The [protocol](../../ITERATION_PROTOCOL.md) and [registry](registry.json) assign whole documents, with no overlap by canonical ID, normalized full text or normalized title. The split reconstruction reproduces every assignment and input/gold hash without model calls.

| Partition | QASPER questions / papers | QuALITY-HARD questions / articles |
| --- | ---: | ---: |
| Development | 3,316 / 1,080 | 1,065 / 115 |
| Validation | 1,005 / 281 | 420 / 50 |
| Locked test | 728 / 224 | 831 / 100 |

The 384 questions already published in bounded-v1 and their documents are exposed development material. QuALITY validation/test use a custom document partition of the public training release, not the hidden official test. No held-out failure cases or scores have been inspected.

## Failure audit and candidate exposure

The old 12-chunk candidate cap exposed unequal amounts of source text. On the 192 exposed QASPER questions, Jev supplied a mean 2,339 candidate source tokens, compared with 3,115 for recursive chunks with Codex reranking. Among text-representable evidence cases, the candidate pool was incomplete in 13 Jev cases and 5 baseline cases. Final ranking/packing lost complete evidence in another 6 cases per pipeline. The [audit](../../results/development-v1/diagnostic-summary.json) separates these stages from missing visual/textual evidence and reader errors.

The first fix gives both rerankers up to **8,192 source tokens / 128 candidates**, keeping the same reader prompt and **2,048-token final context**. It reuses existing embeddings and indices. A seeded 128-question screen was [registered before inference](https://github.com/Sheltercosmo/0halluciation-drift-indexing/commit/f72e59d5b22e4f28a6f24dcebc7f47b9e6048443); its 64 questions per benchmark were selected before new predictions, not chosen from failures.

## First screen: a useful fix with a trade-off

All scores are on a 0–100 scale. Evidence recall covers 58 QASPER questions with nonempty reference evidence; answer scores include all 64 per benchmark.

| Pipeline | QASPER answer F1 | QASPER evidence recall | QuALITY-HARD accuracy |
| --- | ---: | ---: | ---: |
| Original Jev, 12 candidates | 52.64 | 82.18 | 89.06 |
| Jev, 8,192-token pool | **55.72** | **90.56** | **87.50** |
| Original Codex reranker, 12 candidates | 53.81 | 89.62 | 87.50 |
| Codex reranker, 8,192-token pool | 53.78 | 91.35 | 90.63 |

Jev's answer changes versus its original pipeline were **+3.08 QASPER F1** (95% document-bootstrap interval −3.90 to +10.34) and **−1.56 QuALITY points** (−8.20 to +4.84). The evidence-recall increase was **+8.37 points** (+3.08 to +14.78). Against the equally expanded Codex baseline, answer differences were **+1.94 QASPER F1** (−3.19 to +7.66) and **−3.13 QuALITY points** (−9.38 to +3.08).

These are descriptive development intervals, with no multiplicity correction; they do not establish answer superiority. There were three QuALITY changes from correct to incorrect and two in the other direction. The candidate-pool fix remains an experimental candidate, not an unconditional new default.

A separately registered control answered the same 64 QuALITY questions without source context and scored **37/64 = 57.81%**, versus 87.50% with the expanded Jev context. Retrieval is informative on this screen, although this control cannot rule out model pretraining exposure to public data.

## Integrity and usage

All **256 new pipeline answers** and **64 no-source control answers** are present. One Codex ranking batch failed complete-permutation validation twice. Before reader calls, the [format amendment](POOL_SCREEN_AMENDMENT.md) registered a deterministic repair: retain returned valid unique IDs and append omitted IDs in original hybrid order. The last response omitted one candidate ID; the raw and repaired orders and both failed attempts are retained. No question or denominator was removed.

The pool screen made 99 Codex attempts (97 successful, 2 rejected) and 137 Jev requests covering 4,568 decisions. The no-source control made 9 Codex calls. Codex's successful-call token totals are reported; the original client did not retain token usage for its two structurally rejected responses. These attempts remain counted in reservations. No new Gemini requests were made: its conservative cumulative reservation remains **$5.410937 of the $30 cap**. Jev/Codex calls still consume their respective service/account allowances.

Artifacts: [scores](../../results/pool-screen-v1/scores.json), [summary](../../results/pool-screen-v1/summary.json), [paired intervals](../../results/pool-screen-v1/paired-intervals.json), [repair record](../../results/pool-screen-v1/ranking-repair.json), [usage](../../results/pool-screen-v1/usage.json), and [no-source control](../../results/closed-book-v1/summary.json).

## Next registered experiments

Two context-selection changes are prepared on the same development screen, before their reader predictions:

1. Combine the original hybrid rank (weight 0.25) and Jev relevance rank (0.75), using RRF constant 60. In an offline development sweep, hybrid weights 0.25 / 0.50 / 0.75 produced QASPER evidence recall 90.56 / 89.22 / 83.34. Only the 0.25 candidate proceeds to reader testing; this selection is development tuning.
2. Read selected fragments bottom-up into their existing decision-defined topic parent when at least two children are selected, coverage is sufficient and the full parent fits the same final token limit. This follows HiChunk's auto-merge retrieval idea, not its trained chunker. Offline adaptive/0.4/0.6 coverage trials all left QASPER recall at 90.56; reader testing will check whether complete narrative context helps QuALITY.

Neither trial changes topic-cut probabilities or uses an LLM during indexing. Their predictions will be reported whether favorable or unfavorable. Successful screening still requires broader development testing, selection on registered validation candidates, a published-method comparator, and one final untouched test.
