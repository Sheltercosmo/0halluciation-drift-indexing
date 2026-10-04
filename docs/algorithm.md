# Index construction and decision rules

## Input and structure

The parser recognizes ATX (`# Title`) and single-line Setext headings. It preserves heading nesting, including skipped levels, and treats headings as hard boundaries. Blank lines delimit paragraphs; soft line breaks stay within a paragraph. Fenced code is preserved as one verbatim block. This is a small parser, not a complete CommonMark implementation or a PDF layout parser.

Preamble text lives under the document root. Within a heading, adjacent paragraphs are segmented into topic sections. A later return to an earlier topic is a new section, not a noncontiguous merge.

The structural stage also accepts source-aligned `HeadingHint` values and separates recognized contents lists into navigation nodes. Contents links resolve to existing headings and do not reach Jev. See [structural design and provenance](structure-and-retrieval-design.md). New indexes use schema v2; v1 indexes remain readable.

Source offsets are zero-based Unicode code-point indices with exclusive ends, not byte offsets or JavaScript UTF-16 indices. Lines are one-based. The CLI preserves CRLF when reading input. The index stores source text once and derives all node content by slicing it.

## One-way topic comparisons

For a group starting at paragraph `a`, compare `(a, a+1)`, `(a, a+2)`, and so on. Never advance `a` until a cut. Never compare all paragraph pairs. At a cut before paragraph `i`, create a new group anchored at `i`.

The Jev adapter asks a binary `noul` question: do the anchor and candidate discuss the same specific topic? It reads the returned `noul` probability, not a separate confidence value. Headings are not submitted as paragraph text. Each request is evaluated separately and successful identical requests are cached within the scorer instance.

## Bayesian prior: probability versus likelihood

Jev already returns a model probability `q`. It is not a likelihood ratio. If `q` were calibrated at a known reference prior `r`, and class-conditional evidence were unchanged, then:

```text
odds(x) = x / (1 - x)
LR = odds(q) / odds(r)
posterior odds = LR × odds(pi)
P(same | evidence, pi) = posterior odds / (1 + posterior odds)
```

Here `pi` is the chosen same-topic prior. Implementation uses log odds for numerical stability and preserves exact zero and one inputs. With `pi = r`, no adjustment occurs.

**The default `r = 0.5` is an explicit experimental assumption.** It is not an established Jev training or calibration prior. Until `r` and calibration have been measured on representative paragraph pairs, call the result a prior-adjusted decision probability, not an empirically verified posterior. A stronger production approach would fit a calibration map on labeled pairs with a documented sampling prior.

For lexical/embedding similarity `s`, which is not a probability, the offline baseline instead assumes:

```text
s | same      ~ Beta(same_alpha, same_beta)           # default Beta(2, 1)
s | different ~ Beta(different_alpha, different_beta) # default Beta(1, 2)

P(same | s) = pi f_same(s) / [pi f_same(s) + (1-pi) f_different(s)]
```

Endpoints are clipped to `[1e-9, 1-1e-9]` for these densities. The default distributions are illustrative, not fitted. Jev probabilities do not pass through this Beta model.

Pairs share an anchor and are correlated. The algorithm does **not** repeatedly multiply their posteriors as though they were independent evidence. Each pair uses the same configured prior; the previous posterior is only a boundary baseline.

## Sudden-drop rule

```text
previous = pi at the start of a section
current = prior-adjusted probability for (anchor, candidate)
drop = previous - current
cut = current <= posterior_cutoff AND drop >= minimum_drop
```

Defaults are `pi=0.7`, `posterior_cutoff=0.5`, `minimum_drop=0.2`. A first comparison can cut relative to the prior, allowing a one-paragraph section. After a cut, reset `previous` to `pi`; otherwise set it to `current`. Every comparison records its anchor, candidate, score, prior, posterior, drop, and cut decision.

This deliberately implements the proposed sharp-drop rule, not full Bayesian online change-point detection. Gradually falling probabilities may never trigger a cut. A short off-topic paragraph may trigger two cuts. These behaviors should be assessed before adding smoothing, lookahead, minimum section lengths, or drift detection.

## Central sentences

After segmentation, select the section representative and its paragraph representatives. Only existing source sentences are eligible. Their scores are independent, so section and paragraph judgments can share a Jev request without waiting for one another.

Search proceeds in waves across every paragraph in the same topic section:

| Wave | Paragraph 1 | Paragraph 2 | Paragraph 3 |
| --- | --- | --- | --- |
| 1 | First + last | First + last | First + last |
| 2 | Second + second-last | Second + second-last | Second + second-last |
| 3 onward | Continue inward | Continue inward | Continue inward |

Odd-length paragraphs evaluate the middle sentence once; finished paragraphs leave subsequent waves. For each candidate, one question uses the whole section as context and another uses its own paragraph. Jev's multi-question API evaluates the independent judgments in parallel. A one-sentence target represents itself without a model question. Ties favor the earlier evaluated candidate.

**The default is exhaustive**: every sentence is visited. An optional budget `B` caps candidates per target node (section or paragraph). A capped section can exhaust its budget before every paragraph contributes a candidate; use exhaustive mode when that coverage is required. The selected sentence, visited indices, and exhaustive flag remain inspectable.

Jev scores whether each candidate expresses the central topic of the complete section/paragraph context, using a separate `noul` question. The lexical and embedding adapters use mean similarity between the candidate and all other sentences, excluding itself. These are distinct scoring definitions and should be evaluated separately.

No automatic confidence-based early stopping is claimed. Finite `B` gives an approximate maximum; `None` in Python or `0` in the CLI evaluates all candidates. The possibility of missing a higher-scoring unevaluated sentence applies to any incomplete search, regardless of ordering. Output records visited indices, candidate count, score evaluations, method, and whether the search was exhaustive. Even exhaustive search maximizes the chosen model score, not an objective notion of truth.

## Cost and determinism

Paragraph grouping needs at most `n-1` model decisions within each heading run. These anchor comparisons remain sequential because a cut changes the anchor. Representative selection visits up to `min(B,m)` candidates per target node, or all `m` in exhaustive mode. Questions for all active paragraphs and the section are combined per wave; repeated contexts and sentence strings occur once in each request. Identical judgments use the in-memory cache.

For a wave of `Q` distinct uncached questions and batch size `K`, the question-count limit alone needs `ceil(Q/K)` requests; context-size splitting may require more. The default `K=64` is an application batching setting, not a claimed provider maximum. Batches within a wave are sent sequentially, while questions within each request use Jev's parallel evaluation. With no splitting, the number of wave round trips follows the longest paragraph's half-length rather than the sum of paragraph lengths. Parallelism reduces round trips and repeated context transmission; it does not by itself reduce the number of candidate judgments or prove a latency/token-cost improvement.

Lexical pairwise centrality needs at most `min(B,m) × (m-1)` comparisons per target; exhaustive selection is quadratic. The lexical fallback runs locally and does not claim Jev-style parallel execution.

Jev HTTP requests are bounded by `max_calls`; the limit counts requests, not questions. Exceeding it aborts the build rather than silently returning an incomplete exhaustive result. Reusing a scorer shares its cache and call budget. Response model identifiers, request counts, and answered-question counts are recorded.

A configurable `max_state_chars` guard (default 60,000) splits batches when their combined contexts are too large. A single oversized context raises an error without truncation. This is a character guard, not a tokenizer or a guarantee of fitting the provider's token limit. Persisted caching, automatic retries, and splitting an individual section context are not implemented. The code is deterministic for fixed scores, but repeated remote model runs are not promised identical.

The parallel-question request design follows [TypeSafe's building guide](https://docs.typesafe.ai/concepts/how-to-build-with-system-one) and the [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request). Tests simulate responses, including out-of-order answers and partial failures; a live provider benchmark is still needed.

An index build is only written after successful completion. Failures raise explicit errors. Source hashes detect accidental source changes in saved indexes; they are not authentication signatures.

## Evaluation before making accuracy claims

Use held-out documents with labeled paragraph boundaries, representative sentences, and answer-evidence spans. Measure boundary precision/recall/F1 and WindowDiff, representative coverage, evidence recall at a fixed reading/token budget, citation validity, latency, calls, and input tokens. Measure calibration with reliability bins and Brier scores on the actual same-topic task. Tune thresholds on a separate development split.

Compare anchor-only versus adjacent-pair and windowed alternatives; prior adjustment versus raw Jev probabilities; finite candidate budgets versus exhaustive selection; lexical versus Jev versus optional embeddings. Include gradual drift, short bridges, repeated vocabulary, heading-free documents, middle-position topic sentences, and cross-references. No such benchmark has been run yet.
