# Evaluation grounded in current research

Research and dataset audit: 2026-10-05. **Status: complete datasets verified; model comparisons not yet run.** This replaces the proposed 60-claim SciFact extension as the main evaluation plan. The existing synthetic and 12-claim SciFact results remain smoke tests, not evidence of competitive RAG performance.

## What the papers actually evaluate

| Paper or system | Evaluation used | Implication for this project |
| --- | --- | --- |
| [HiChunk, ACL 2026](https://aclanthology.org/2026.acl-long.1372.pdf), sections 3, 5 and appendix A | HiCBench: 130 documents, 659 single-chunk and 541 multi-chunk evidence-dense questions. Boundary F1 by hierarchy level, evidence recall, ROUGE and Fact-Cov; Fact-Cov averaged over five evaluations. Retrieval budgets of 2k–4k tokens. Also evaluates QASPER, LongBench, GutenQA and OHRBench. | Most directly relevant: its analysis finds that sparse evidence can conceal differences between chunkers. Evaluate recovery of complete evidence under the same context budget. Its QA pairs are synthesized, so pair this benchmark with human-authored questions. |
| [Bright-Pro, ACL 2026](https://arxiv.org/html/2605.04018v1), sections 4–5 and appendix F | Static retrieval across 739 queries and seven domain corpora totaling 526,319 documents; 2,763 annotated reasoning aspects. Primary metrics are alpha-nDCG and weighted aspect recall. Its agentic experiments use a smaller 175-query subset, fixed one/two/three search rounds or adaptive stopping, and answer judging. | Use all 739 queries for static evaluation. Coverage of complementary evidence is more informative than whether any one correct passage appeared. Treat this as a retrieval test, not a direct test of long-document blocking. |
| [LongBench v2, ACL 2025](https://arxiv.org/abs/2412.15204), [official implementation](https://github.com/THUDM/LongBench) | 503 multiple-choice questions, 8k–2M-word contexts, six task categories. Accuracy broken down by difficulty and length; RAG experiments vary retrieval depth with 512-token chunks. | Use the full release and disclose all category results. Match retrieved token budgets rather than numbers of chunks. Hard multiple-choice scoring supplies an outcome independent of an LLM judge. |
| [RAPTOR, ICLR 2024](https://arxiv.org/html/2401.18059v1), section 4 | NarrativeQA, QASPER and QuALITY, including QuALITY-HARD. Answer F1 for QASPER and accuracy for QuALITY. Controlled comparisons hold the reader and retrieval model fixed while adding the hierarchy. | Established structural RAG comparator. Copy the experimental control, not its historical leaderboard scores or an assumption that a newer reader reproduces them. |
| [PageIndex OSS benchmark](https://github.com/VectifyAI/PageIndex-OSS-Benchmark) | 62 text fact-lookup questions over 34 PDFs. The selection excludes visual and numerical-reasoning tasks and requires successful Flash indexing. | Useful compatibility check, too restricted to be our headline comparison. An indexing failure must not remove a question from our denominator. |

HiCBench access is unresolved: the official repository's data badge has no target, its linked Drive file is described as an example LongBench dataset, and the candidate `Youtu-RAG/HiCBench` Hugging Face endpoint returned HTTP 401. This does not establish that the data are unavailable everywhere. Do not claim to have downloaded or evaluated HiCBench, and do not recreate synthetic questions and label them as that benchmark.

## Complete releases already verified

Counts below come from the downloaded files, not rounded paper totals. Source URLs, revisions and SHA-256 hashes are in [frontier/sources.json](frontier/sources.json). [Inventory](frontier/inventory.json) and [complete question IDs](frontier/selection.json) are committed; source document text stays in ignored `output/`.

| Benchmark | Fixed evaluation scope | Purpose and scoring |
| --- | --- | --- |
| [QASPER v0.3](https://huggingface.co/datasets/allenai/qasper) | **1,451 test questions; 416 papers; 20,221 body paragraphs** | Primary within-document evidence and QA evaluation. Official answer/evidence F1; evidence coverage against actual returned source spans at each token budget. Keep all questions, including 79 unanimously unanswerable and 164 with mixed answerability annotations. |
| [QuALITY v1.0.1](https://github.com/nyu-mll/quality) | **2,086 development questions; 115 distinct articles; 1,065 HARD questions** | Report overall and HARD accuracy. The 230 records are question sets, not 230 independent articles. Official test labels are withheld; do not call this a test-set result. Tune on training data only. |
| [LongBench v2](https://github.com/THUDM/LongBench) | **All 503 questions; 311 hard; 462 exact distinct contexts** | Overall accuracy and difficulty/length/domain slices. Include all 175 single-document and 125 multi-document QA questions, plus the other four categories. Upstream names the release `train`; we use it entirely for evaluation. |
| [Bright-Pro](https://huggingface.co/datasets/yale-nlp/Bright-Pro) | **All 739 queries; all 526,319 documents across seven domain corpora** | Official alpha-nDCG@10/25, weighted aspect recall@10/25 and nDCG@10. Retrieve from each query's full domain corpus, following the authors; never inject gold passages into a shortlist. |

That is **4,779 questions across distinct evaluation tasks**, not one pooled accuracy score. Bright-Pro documents are supplied retrieval units, often short passages; calling them half a million long documents would be misleading. HiCBench would add 1,200 questions once its release and annotations are accessible, and is not counted in 4,779.

The previous SciFact extension was stopped after 33 embedding requests covering 2,112 abstracts, before reranking or claim evaluation. It produced no comparative result. Its local cache and stop record are retained; it will not become the headline benchmark by increasing its sample slightly.

## Experiment 1: compare decision-based blocking

Use QASPER, QuALITY and LongBench v2. Every arm gets identical source parsing, titles, section paths, embedding model, retrieval algorithm, query, reader prompt and answer budget. The only changing component is the blocking method:

1. Fixed 512-token chunks with 64-token overlap.
2. Recursive paragraph/sentence packing targeting 512 tokens, with the same overlap cap.
3. Embedding semantic splitting with a development-selected similarity threshold and common size limits.
4. Jev topic blocking, using the statistical prior and sudden probability-drop cut rule together as one method.
5. LumberChunker-style generative boundary decisions, adapted to the same source units and common provider. Label this adaptation rather than an exact reproduction of a paper's model configuration.

Use a common BM25+dense RRF retriever (constant 60) and Gemini Embedding 2 for this controlled comparison. Query-time generative reranking, query rewriting, representative-sentence selection and tree expansion are disabled here: these would change additional components. This experiment enables embeddings in retrieval; it does not measure the embedding-free deployment path. All arms receive native heading information equally. Fixed boundaries should also respect the shared heading partitions; report any forced size splits separately.

The primary reader budget is **4,096 retrieved tokens**, with full-set sensitivity runs at **2,048 and 8,192**. Include title prefixes, repeated overlapping text, wrappers and any generated summaries in the budget. Pin a tokenizer and renderer before inference; archive both the nominal tokenizer count and provider-reported input tokens. Oversized blocks are split by the same deterministic source-preserving rule before ranking. Build each document once per configuration and reuse it for its questions. Never use a question, reference answer or evidence label to construct its index.

If tuning is undertaken in a later full study, choose semantic and Jev thresholds using QASPER development and QuALITY training data, grouped by source document, with equal configuration-search budgets. Freeze configurations before test predictions and report all development configurations. LongBench v2 and Bright-Pro receive no benchmark-specific tuning. The bounded run uses fixed defaults and performs no such search.

The prior is part of detecting a sudden drop and deciding where to split a block. Evaluate the complete blocking method through evidence retrieval and answer quality; a separate prior ablation is not required. No probability-calibration or Bayesian-optimality claim is made.

## Experiment 2: compare competitive retrieval systems

On complete Bright-Pro, compare BM25, dense Gemini embeddings, BM25+dense RRF, hybrid plus a strong reranker, and hybrid plus Jev reranking. All rerankers receive the same top-100 candidate pool and metadata, and get credit for no evidence lost before that pool. Report first-stage recall@100 alongside reranking outcomes. Candidate order is deterministically shuffled for listwise methods. Gold aspects, reference answers and supporting-document IDs are evaluation-only inputs.

Add a released reasoning retriever such as ReasonIR-8B or RTriever-4B using the authors' implementation and pinned checkpoint if the required serving hardware is available. Bright-Pro compares this class of model with general-purpose embeddings; a dense-only comparison would leave out a relevant competitor. Do not present a locally approximated method or an unavailable model as a completed baseline.

On long-document QA, add actual RAPTOR and PageIndex adapters as a separate end-to-end comparison, using the same source text and reader wherever their APIs allow. Freeze adapter revisions, index-time models, retrieval budgets and parser differences. A shared reader does not equalize indexing cost: report summary-generation and decision-model calls separately. If an adapter requires a different PDF/OCR path, report that as a separate system condition rather than attributing its result solely to blocking.

After the controlled experiment, test production features independently: central-sentence budgets of 2/4/8/full, then upward context expansion, then LLM query proposals. Equal candidate-decision budgets are required when comparing outside-in, sequential and random sentence orders. Full search supplies an upper reference; a budgeted search is not expected to inspect every possible better middle sentence.

## Metrics that can support the intended claim

- **Evidence and answers:** official dataset answer metrics; annotated evidence recall and precision at a token budget; complete-evidence-set recovery; supported answer quality. Keep multiple valid evidence annotations separate instead of requiring their union. Unanswerable questions remain in answer evaluation; report evidence metrics only for their defined eligible population, with its count.
- **Blocking:** exact boundary F1, Pk and WindowDiff only where genuine reference boundaries exist. Parsed headings are not inferred topic-boundary ground truth. Report explicit-heading coverage separately; use masked-heading experiments only as a declared diagnostic. QASPER evidence labels alone do not establish semantic boundary accuracy.
- **Reliability:** unanswerable handling, invalid outputs, retrieval misses and source-support failures. An answer selected correctly without retrieving its support is an answer success and an evidence failure; retain both measurements.
- **Efficiency:** indexing and query calls/tokens/cost separately, index size, p50/p95 observed latency and concurrency, and evidence/answer quality versus retrieved tokens. Count failed requests and retries. Batching reduces request count, but is not automatically a wall-clock speedup.

Use paired differences and 10,000 bootstrap replicates, resampling whole source documents for QASPER/QuALITY. For LongBench v2 cluster exact repeated contexts, disclose that partial source overlap may remain, and report all domain slices. For Bright-Pro resample queries within domains and report both per-domain and domain-macro results. Never combine repeated questions or runs as independent documents. Correct the predeclared primary baseline comparisons for multiplicity (Holm); label other analyses exploratory.

Dataset size is not a guarantee of statistical power: QuALITY has only 115 independent article groups. Before model inference, calculate sensitivity using development-set discordance and document clustering for a target three-percentage-point paired gain. If the resulting interval cannot resolve that gain, report the uncertainty or expand independent data; do not keep adding only questions from the same documents or stop when a favorable p-value appears.

Predeclare QuALITY-HARD and LongBench v2's hard slice as primary difficulty analyses, retaining full-set results alongside them. Historical difficulty does not guarantee that current models avoid ceiling effects. If contemporary baselines saturate an endpoint, publish that result and treat it as non-discriminating; a harder follow-up must be a separately registered experiment, not a replacement chosen after seeing which examples favor Jev.

## Execution and reporting requirements

The complete dataset audit is implemented. The [bounded protocol](BOUNDED_PROTOCOL.md) specifies the currently authorized 384-question, four-pipeline study, using Codex calls and a $30 Gemini ceiling. The larger experiments above are research directions, not a requirement to run every downloaded dataset. This document is not a completed performance report. Freeze prompts, tokenization, model versions, code hashes, source licenses and planned comparisons before inference. Keep a manifest entry for every expected question. A provider error must not silently remove a question, switch models for difficult cases, or trigger a retry based on answer correctness.

```sh
python -m pip install -r evals/requirements-frontier.txt
python scripts/frontier_data.py --download --output output/frontier-manifest

# Later, reject incomplete or duplicate prediction coverage for each method:
python scripts/frontier_data.py --dataset quality --check-predictions output/run/quality.jsonl
```

Prediction JSONL rows require `id` from the committed selection and `status` equal to `ok` or `failed`. Coverage validation is necessary but does not score answers. The preparation script makes no inference calls. It validates every file hash, full question count, unique ID and Bright-Pro gold/aspect-to-corpus link.

Complete large inference is materially different from a bounded study. Five arms across all 4,040 prepared QA questions would require 20,200 question-level reader evaluations at one budget, before indexing and reranking. A run needs an explicit spend ceiling and provider limits; a budget stop yields an incomplete result, not a smaller post-hoc benchmark. Do not claim superiority until paired results, uncertainty, failures and cost have been published together.
