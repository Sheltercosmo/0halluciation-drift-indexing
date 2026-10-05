# Iterative improvement with an untouched test set

Started 2026-10-05. The objective is a statistically supported improvement over strong, reproducible RAG systems. No result currently establishes frontier performance. The prior and sudden-drop rule remain one blocking method; there is no separate prior ablation.

## Separate development, validation and test by document

The published bounded-v1 questions and their entire documents are **exposed development data**. Their original benchmark split names do not make them held out after failure analysis. QASPER training papers are additional development material. All QuALITY development articles are also development material, including questions not selected in bounded-v1.

The new assignment uses seed 20261005 and canonical document IDs, normalized text hashes and normalized titles to keep matching documents together. A duplicate of an exposed/development document is excluded from validation and test. When a duplicate spans validation and test, it stays in validation. Identical generic questions about different papers do not themselves imply duplicate documents. This prevents the detectable overlap checked here; it cannot establish that foundation models never saw these public benchmarks during pretraining.

| Benchmark release | Development | Validation | Locked test |
| --- | --- | --- | --- |
| QASPER v0.3 | Training release and the 192 previously evaluated test papers | Official development release | Remaining official test papers, excluding exposed/duplicate documents |
| QuALITY v1.0.1 HARD | Official development articles | 50 deterministically selected training articles | Remaining 100 training articles, after duplicate exclusion |

The QuALITY split is a **custom document-held-out split**, not its hidden-label official test or a leaderboard submission. Public manifests record every assignment and source checksum. The split script reads labels only to serialize and hash gold files; it prints no held-out questions, labels, examples or outcome statistics. Test gold is separate from model inputs.

Development is available for unrestricted error inspection and iteration. A fixed development screening subset contains up to 64 questions per benchmark from bounded-v1, selected by seeded ID hash before new predictions. The first candidate change expands the reranking pool using a common source-token cap rather than 12 chunks; all compared pipelines receive the same cap. Changes are evaluated on all screening cases, not just previous failures. Promising changes then run on all exposed questions.

Validation is for selecting frozen candidates, not inspecting individual failures. Before accessing it, register at most three candidate configurations and a deterministic selection rule. Report every candidate and aggregate scores, including regressions. Pick the highest average standardized improvement in QASPER answer F1 and QuALITY-HARD accuracy over the strongest matched baseline; break ties by lower query usage, then configuration hash. Do not add candidates in response to validation outcomes. A later research round needs a new untouched validation/test allocation.

The test-opening gate requires a saved frozen configuration, implementation hashes and the baseline list. It records test access before returning paths. Resumes must use exactly the same frozen bytes. No test failure inspection or tuning occurs until all registered predictions and the final report are complete. Once opened, this test is spent for subsequent method development; an unsuccessful result does not authorize retuning and calling it held out again.

## Comparisons and success criterion

Keep one reader model/prompt, source parsing, final context limit and scoring implementation across pipelines. Indexing remains independent of questions and gold labels. Record candidate-token exposure separately from final reader tokens, as well as candidate counts, reranking calls, preprocessing calls, latency and failures. Improvements may change retrieval, but their extra query compute must be explicit. Include the original Jev pipeline to measure regression.

Required controls are hybrid retrieval with recursive and semantic chunking and a strong generative reranker. Before a frontier claim, also reproduce at least one applicable published open-source retrieval algorithm, pin its code and dependencies, check adapter fidelity, and use matched data/reader/budgets. A paper-inspired implementation is labeled as such, not presented as the paper's complete system. Full-context reading can be an additional diagnostic but has a different context budget.

Relevant primary sources checked on 2026-10-05:

- [HiChunk / HiCBench](https://github.com/TencentCloudADP/hichunk): hierarchical chunking and automatic parent merging. Its README still has empty model/data links and requires a locally supplied trained checkpoint. Audit release availability before claiming a reproduction.
- [RAPTOR](https://github.com/parthsarthi03/raptor): recursive clustering and abstractive summaries; retain model and summarization-cost differences in any adapted comparison.
- [SARA](https://github.com/Ahren09/SARA): trained compression/projector and reader adapters, requiring a different model interface; not a plug-in comparator for the identical Codex reader.
- [VecTree-RAG](https://arxiv.org/abs/2607.23006): its QASPER headline is model-judged correctness, not official token F1. Do not compare its 0.800 directly with our F1.
- [QASPER](https://huggingface.co/datasets/allenai/qasper) and [QuALITY](https://github.com/nyu-mll/quality) provide the open data and annotations used here.

The planned confirmatory primary outcomes are official QASPER answer token F1 and QuALITY-HARD accuracy, reported separately. Use whole-document paired bootstrap intervals and document-level paired randomization tests, with Holm correction across all registered superiority comparisons. A substantive target is at least **2 points improvement** and adjusted p < .05 on both benchmarks against each required strong baseline, without a material evidence-recall regression. Evidence recall/F1, invalid answers and abstention behavior remain secondary outcomes. Fix any non-inferiority margin, number of reader replicates and final baseline list before opening validation. No pooled leaderboard or SOTA claim follows merely from beating our earlier implementation.

This target is an objective, not a guaranteed outcome. If a fixed confirmatory test fails, publish that result and continue only on newly designated data. Never repeatedly inspect test results until significance appears.

## Cost and records

Reuse content-addressed embeddings, indices and unchanged predictions. Gemini's existing $30 total cap continues across iterations; initialize the research ledger with the prior $5.410937 conservative reservation. Reserve before every attempted request, including retries. Codex uses the user's existing account allowance; record calls and tokens and stop on provider limits. Jev calls and decisions are counted separately.

Every iteration records its hypothesis, frozen configuration, input IDs, source hashes, observed failures, all scores and the decision to accept or reject it. Frozen historical runs are never rewritten. Regression tests verify source offsets, complete paragraph coverage, token budgets, split isolation, deterministic resumes and budget enforcement; they do not substitute for measured benchmark gains.
