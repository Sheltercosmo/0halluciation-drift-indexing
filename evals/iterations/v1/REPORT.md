# Improvement log: development screens, not held-out results

The improvement goal is active. **Frontier performance has not been established.** Validation and test are still locked. The prior and sudden-drop rule have not been tested separately. Early screens exposed substantial reader variability for identical per-question contexts in different batches. The completed [isolated-reader comparison](#isolated-reader-results) supersedes those provisional answer deltas; historical results remain below for transparency.

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

## Follow-up context experiments

Two context-selection changes are prepared on the same development screen, before their reader predictions:

1. Combine the original hybrid rank (weight 0.25) and Jev relevance rank (0.75), using RRF constant 60. In an offline development sweep, hybrid weights 0.25 / 0.50 / 0.75 produced QASPER evidence recall 90.56 / 89.22 / 83.34. Only the 0.25 candidate proceeds to reader testing; this selection is development tuning.
2. Read selected fragments bottom-up into their existing decision-defined topic parent when at least two children are selected, coverage is sufficient and the full parent fits the same final token limit. This follows HiChunk's auto-merge retrieval idea, not its trained chunker. Offline adaptive/0.4/0.6 coverage trials all left QASPER recall at 90.56; reader testing will check whether complete narrative context helps QuALITY.

Neither trial changes topic-cut probabilities or uses an LLM during indexing. Both completed all 256 expected predictions, using 32 new reader calls and two exact-batch cache hits.

| Context policy | QASPER answer F1 | QASPER evidence recall | QuALITY-HARD accuracy |
| --- | ---: | ---: | ---: |
| Expanded pool, relevance order | 55.72 | 90.56 | 87.50 |
| Rank fusion | 56.37 | 90.56 | 84.38 |
| Topic-parent expansion | 53.09 | 90.56 | 89.06 |

Neither follow-up is a clear improvement across both tasks. [All predictions](../../results/context-trial-v1/scores.json) and [usage](../../results/context-trial-v1/usage.json) are retained. No new Gemini or Jev calls were needed.

## Reader variability changes the next step

For rank fusion, 27 QASPER questions received exactly the same individual source context as the expanded-pool run; **14 nevertheless changed answer score**. For parent expansion, **19 of 46** such questions changed score. QuALITY had 0 of 25 and 1 of 39 score changes respectively. See the [exact-context audit](../../results/context-trial-v1/reader-variability.json).

The batch prompts included other cases whose contexts differed. This audit cannot separate stochastic generation from cross-case batch interference; either can contaminate a small retrieval comparison. Evidence recall is computed from deterministic source spans and is unaffected by this reader issue. Historical outputs are preserved, but their answer deltas are provisional development observations.

The [isolated-reader rerun](isolated-reader-manifest.json) keeps all five retrieval configurations and all 128 questions, submits **one case per reader call**, and shares one prediction whenever case ID, query and context are identical across methods. Its 640 method/question predictions require **488 distinct calls**. It keeps the same Codex reader model and instruction and makes no new indexing, reranking or Gemini calls. It removes batch coupling and matched-context resampling; a single draw still leaves uncertainty for genuinely different contexts.

Successful development screening still requires broader development testing, selection on registered validation candidates, a published-method comparator, and one final untouched test. No candidate has been promoted to the production default.

## Holdout protection and published comparator preparation

The holdout gate now requires a registration saved before validation, complete predictions for every registered method/question/reader replicate, and a winner recomputed from answers with the hashed evaluator. It rejects status-only claims, missing or duplicated predictions, changed registrations and changed frozen artifacts. Failed predictions count as zero. The expanded test suite passes 108 tests, including eight validation-gate checks; reconstructing the splits still reproduces every assignment and prepared data hash. The real validation and test opening commands have not been run.

The [RAPTOR adapter](../../RAPTOR_BASELINE.md) now runs pinned upstream clustering, tree construction and collapsed retrieval with explicit model callbacks. Its offline integration check passes without model calls, including fixed-seed reproduction and the actual rendered context budget. Live model adapters and measured QA comparisons remain pending; this is implementation evidence, not a performance result.

## Isolated-reader results

The registered rerun completed all **488 distinct requests**, producing **640 method/question predictions** on the same exposed-development sample. One malformed answer-ID response was retained and retried unchanged; there were 489 attempts, 488 valid responses and no missing or failed final predictions. Every identical ID/query/context combination shares one answer across methods. Exact source spans, token limits, official scores and aggregate summaries replay without model calls.

| Pipeline | QASPER answer F1, 64 questions | QASPER evidence recall, 58 eligible questions | QuALITY-HARD accuracy, 64 questions |
| --- | ---: | ---: | ---: |
| Original Jev, 12 candidates | 51.08 | 82.18 | 87.50 |
| Jev, expanded pool | 58.55 | 90.56 | 85.94 |
| Jev, rank fusion | **58.88** | 90.56 | 85.94 |
| Jev, topic-parent expansion | 58.40 | 90.56 | 85.94 |
| Codex reranker, expanded pool | 55.49 | 91.35 | 85.94 |

Expanded Jev versus original Jev gains **7.46 QASPER F1 points**, with a descriptive 95% document-bootstrap interval **+1.79 to +14.18**. Evidence recall gains **8.37 points** (+3.08 to +14.78). QuALITY loses **1.56 points** (−8.33 to +5.00). These development intervals are not multiplicity-adjusted confirmatory tests.

Against the expanded Codex reranker, Jev rank fusion leads by **3.39 QASPER F1 points**, with interval **−0.10 to +8.20**. QuALITY is tied at **55/64**, with a paired interval of −6.06 to +6.35 points. Neither a reliable cross-task advantage nor frontier performance is established. The slight QASPER differences among the expanded Jev policies do not justify selecting a new default yet.

The completed audit finds **zero answer/score differences for identical contexts**, by construction through prediction reuse. This repairs batch coupling and matched-context resampling. Different contexts still receive a single stochastic answer, so reader uncertainty remains. The earlier 56.37 F1 and differing QuALITY policy scores are historical observations, not scores to mix with this run.

### Failure cases and the next hypothesis

Expanded Jev has 16 QASPER score losses against the expanded Codex reranker. In **13 of those 16**, both contexts contain complete annotated evidence; only **one** has lower evidence recall. Inspection shows several differences in verbosity or wording, such as returning “English tweets only” instead of the gold “English.” We retain the official metric and reader unchanged rather than crediting these losses entirely to retrieval.

The genuine evidence-selection failure asks which models a paper compared. The DenseNet/HighwayLSTM passage was in Jev's candidate pool, scored **0.63**, and was omitted from the final context. Two QuALITY losses also contain relevant material in the pool but omit it during final selection: a passage establishing a character's presence scored **0.44**, and a story's final escape passage scored **0.69**. Candidate exposure alone cannot repair these cases.

The next hypothesis is to use the query-time LLM to propose explicit evidence needs, then let Jev select complementary passages for those needs under the same final budget. This follows the intended retrieval design while keeping indexing free of generative LLM calls. Its added query cost must be recorded, and a matched planner/reranker control is needed. It is a development hypothesis, not a registered or measured improvement yet.

Artifacts: [scores](../../results/isolated-reader-v1/scores.json), [summary](../../results/isolated-reader-v1/summary.json), [paired intervals](../../results/isolated-reader-v1/paired-intervals.json), [exact-context audit](../../results/isolated-reader-v1/analysis.json), [failure comparisons](../../results/isolated-reader-v1/failure-comparisons.json), [usage](../../results/isolated-reader-v1/usage.json), and [frozen source](../../results/isolated-reader-v1/source/scripts/isolated_reader_trial.py). Successful calls recorded 8,819,680 input tokens and 29,570 output tokens; the frozen client did not retain usage for the single rejected response. No new Gemini or Jev requests were needed. Gemini's cumulative conservative reservation remains **$5.410937 / $30**.

## Direction change: component and whole-system comparisons

On 2026-10-05 the research focus changed from flat reranking to central-sentence selection, root-to-leaf tree search, complete-system comparisons and a hybrid combining Jev tree retrieval with independent direct embedding retrieval. The useful split comparison is retained. The new [protocol](../../TREE_SYSTEM_PROTOCOL.md) defines matched Jev/embedding controls; the earlier screen is historical evidence rather than proof of those components.

The preregistered planned-evidence run completed 128 plans and 128 retrievals, then stopped on a reader process timeout with 28 of 256 distinct reader outputs present. No aggregate scores were computed. Partial outputs and reservations remain preserved locally; the frozen registration is unchanged. This incomplete run is deprioritized following the direction change, not reported as a successful or failed method comparison.

Independent representative selection, Jev/embedding root-to-leaf routing, direct dense retrieval and final-output fusion now have offline control tests. Those fixtures verify software behavior, including dense recovery outside the visited tree, but establish no benchmark advantage. Canonical benchmark trees, audited live callbacks, complete experiment registration and baseline execution are still required. Validation and test remain unopened.
