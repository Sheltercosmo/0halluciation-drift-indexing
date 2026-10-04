# SciFact · public-data retrieval micro-pilot

**Jev placed an annotated evidence sentence first in 11/12 cases, compared with 7/12 for BM25-style ranking.** There were five improvements, one regression and six ties. This independently annotated public-data check supports the reranking component under a supplied document pool; it does not establish open-corpus or end-to-end performance.

The data comes from [SciFact, Wadden et al., EMNLP 2020](https://github.com/allenai/scifact). Its [data schema](https://github.com/allenai/scifact/blob/master/doc/data.md) supplies evidence documents, rationale sentence indices, and SUPPORT/CONTRADICT labels. We use those existing annotations, with no new gold labels from the implementing assistant.

## Frozen setup

The [protocol](SCIFACT_PROTOCOL.md) selected twelve development claims with seed 1729: six SUPPORT and six CONTRADICT, with distinct single evidence documents. Each pool contains the annotated evidence abstract and four high-scoring BM25 distractor abstracts selected from the complete downloaded corpus. This deliberately ensures the positive document is present. Pools contain **35–86 sentence candidates**.

Both rankers receive the same declared title/sentence tree, first-sentence paragraph representatives and unmodified claim text. Every candidate is ranked. Jev uses the existing relevance prompt without training, tuning, an embedding model or a generative LLM. This test bypasses learned index construction and evaluates **reranking only**. Retrieval relevance includes evidence that contradicts the claim; it is not a verdict about whether the claim is true.

## Results

| Metric | BM25-style | Jev reranking |
| --- | ---: | ---: |
| Hit@1 | 7/12 · 58.3% | **11/12 · 91.7%** |
| Hit@3 | 8/12 · 66.7% | **12/12 · 100%** |
| Mean union-rationale recall@3 | 52.8% | **91.0%** |
| Mean reciprocal rank | 0.653 | **0.944** |
| Hit@1, SUPPORT | 3/6 | **6/6** |
| Hit@1, CONTRADICT | 4/6 | **5/6** |

Recall uses the union of annotated rationale sentences and is a diagnostic defined by this project, not the official SciFact rationale-set metric. There is no label classification or official leaderboard score here.

| Claim ID | Annotation | BM25 first hit | Jev first hit |
| --- | --- | ---: | ---: |
| 921 | SUPPORT | No | Yes |
| 636 | SUPPORT | No | Yes |
| 784 | SUPPORT | No | Yes |
| 274 | CONTRADICT | Yes | **No** |
| 1216 | SUPPORT | Yes | Yes |
| 56 | SUPPORT | Yes | Yes |
| 1262 | SUPPORT | Yes | Yes |
| 142 | CONTRADICT | Yes | Yes |
| 338 | CONTRADICT | Yes | Yes |
| 48 | CONTRADICT | No | Yes |
| 171 | CONTRADICT | Yes | Yes |
| 718 | CONTRADICT | No | Yes |

For claim 274, Jev placed two sentences from document 5114940 ahead of the annotated rationale in document 11614737; the first gold sentence appeared at rank three. This is a real regression under the supplied labels and is retained. Unannotated distractors may contain other useful evidence; we did not relabel them after seeing the result.

## Usage and replay

Pinned `jev-1.13.0` returned **607 decisions in 14 requests**, using **191,364 input tokens and 10,850 output tokens**. Summed client request time was **5.30 seconds**, excluding corpus preparation and lexical indexing/ranking. There were no provider failures; dollar cost was not measured. Offline replay reproduced the complete stored rankings, scores and metrics exactly.

The [selection](scifact-selection.json), [full numeric results](results/scifact-v1-live/results.json), [decision audit](results/scifact-v1-live/decisions.jsonl), and [frozen source/data hashes](results/scifact-v1-live/manifest.json) are included. Abstracts remain under ignored `output/` and are not redistributed in this repository. Audit records store request-body hashes, not the abstract text or authorization headers.

```sh
# Download the authors' public archive and verify its pinned SHA-256.
python scripts/download_scifact.py

# Replay using the archived source; no model API key or model call is needed.
python evals/results/scifact-v1-live/source/scripts/scifact_pilot.py run --data output/scifact --replay evals/results/scifact-v1-live --output output/scifact-replay

# New paid run on the frozen selection; set TYPESAFE_API_KEY first.
python scripts/scifact_pilot.py run --data output/scifact --output output/new-scifact-run
```

The download needs network access once; replay then uses only local dataset files and stored decisions. Each new run is capped at 60 HTTP requests and 1,500 questions. The archive's source snapshot permits historical replay after later implementation changes.

## Interpretation

These are twelve seeded development examples with supplied positive documents and four lexical distractors, not a random sample of all scientific questions. They exclude unanswerable claims and multi-evidence-document cases. Model exposure to this public benchmark is unknown. This protocol was frozen after the synthetic pilot and before any SciFact calls; prompts and retrieval parameters were unchanged. The observed improvement is useful initial evidence for Jev reranking, with one explicit failure, rather than proof of broad superiority or zero hallucination.
