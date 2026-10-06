# Standard shared-context retrieval comparison

**The retained standard is Jev traversal, bidirectional pairwise ranking and shared-context evidence selection.** It reaches **91.45% full-Jev Recall@5 and 92.44% hybrid Recall@5** on the same 640 historical questions with aligned evidence references.

All 728 questions from 224 QASPER papers have predictions for both systems. The primary evidence population contains 640 questions from 217 papers. Every result returns up to five whole original paragraphs. No generated-answer reader is used.

## Comparable progression

| Version | Full Jev Recall@5 | Hybrid Recall@5 |
| --- | ---: | ---: |
| V4 | 87.56% | 89.51% |
| Improved traversal + pairwise ranking | 90.80% | 91.51% |
| Shared-context selection — standard | 91.45% | 92.44% |

These rows use the same question IDs, source paragraphs, reference alignment and five-paragraph output budget. The standard preserves the shared-context method; later experimental selectors do not replace it.

## Complete systems and cached baselines

| System | Evidence Recall@5, % [95% CI] | Complete evidence@5, % | Evidence F1@5, % |
| --- | ---: | ---: | ---: |
| Full Jev — shared context (standard) | 91.45 [89.31, 93.48] | 88.12 | 37.68 |
| Jev + independent direct embeddings — shared context (standard) | 92.44 [90.28, 94.43] | 89.22 | 38.04 |
| Full Jev — traversal + pairwise ranking | 90.80 [88.77, 92.76] | 87.50 | 36.67 |
| Hybrid — traversal + pairwise ranking | 91.51 [89.55, 93.41] | 87.97 | 36.86 |
| Full Jev v4 | 87.56 [84.83, 90.16] | 84.38 | 36.18 |
| Hybrid v4 | 89.51 [86.97, 91.88] | 86.41 | 36.36 |
| Gemini Embedding 2, direct paragraphs | 80.78 [77.91, 83.60] | 76.88 | 31.62 |
| Direct Gemini + Jev v4 reranking | 86.87 [84.54, 89.17] | 83.44 | 34.77 |
| BM25 + direct Gemini, RRF | 72.97 [69.63, 76.23] | 68.75 | 28.25 |
| Qwen3 Embedding 0.6B | 63.77 [60.06, 67.52] | 58.91 | 24.74 |
| Qwen3 Embedding + Qwen3 Reranker 0.6B | 75.64 [72.27, 78.96] | 71.88 | 28.95 |
| Qwen3 Embedding + Jev v4 reranking | 75.04 [71.96, 78.09] | 70.00 | 29.21 |
| BGE-M3 dense | 69.97 [66.47, 73.41] | 66.09 | 26.95 |
| BGE-M3 + BGE Reranker v2-M3 | 72.35 [68.84, 75.79] | 68.75 | 27.42 |
| BGE-M3 + Jev v4 reranking | 81.22 [78.35, 84.12] | 77.66 | 31.63 |

All 18 historical baseline rows are reused without model calls and preserved in the score archive. Qwen uses 0.6B checkpoints; BGE-M3 uses dense retrieval. The [historical crossed experiment](RETRIEVAL_V4_REPORT.md) still isolates splitting / central sentences / search under its own common final selector.

## Paired comparisons

| Comparison | Recall difference, percentage points [95% CI] | Holm-adjusted p |
| --- | ---: | ---: |
| Full Jev — shared context (standard) minus Full Jev v4 | 3.89 [1.96, 5.95] | 0.0010 |
| Jev + independent direct embeddings — shared context (standard) minus Hybrid v4 | 2.93 [1.04, 4.87] | 0.0120 |
| Full Jev — shared context (standard) minus Full Jev — traversal + pairwise ranking | 0.65 [-0.83, 2.12] | 0.5253 |
| Jev + independent direct embeddings — shared context (standard) minus Hybrid — traversal + pairwise ranking | 0.92 [-0.67, 2.46] | 0.5253 |
| Full Jev — shared context (standard) minus Direct Gemini + Jev v4 reranking | 4.58 [2.27, 6.89] | 0.0012 |
| Jev + independent direct embeddings — shared context (standard) minus Direct Gemini + Jev v4 reranking | 5.57 [3.22, 7.85] | 0.0006 |

Intervals use 10,000 paired document-cluster bootstrap draws with question-weighted means. Two-sided document sign-randomization tests use Holm correction across these six retrospective comparisons. The small shared-context gains over the pairwise controls have intervals including zero.

Compared with the pairwise controls, shared selection improves recall on 34 full-Jev questions and worsens 19; the hybrid improves 38 and worsens 18. The aggregate gains do not imply every question improves.

## Standard method

Headings and the Jev topic/central-sentence index provide the tree. Jev evaluates children of promising nodes from root to paragraphs, using the original question and LLM-proposed evidence needs. The evaluated beam is five, the acceptance threshold is 0.2, and accepted deferred branches receive up to 25% additional routing decisions within the 4,096-decision global ceiling.

The hybrid independently retrieves direct-embedding hits and fuses them with Jev candidates. Both systems then compare the top 30 whole paragraphs in both orientations with Jev. Consistent preferences contribute a win; conflicting preferences tie.

The top 12 pairwise-ranked paragraphs become selectable targets in a shared evidence packet. Complete source paragraphs, heading paths, nearby introductions, preceding/following context and source links are visible together. Jev scores each target in source order and reverse source order; the mean score determines the final five. Exact ties retain earlier pairwise order. Context-only paragraphs clarify the source but are not returned or credited as targets.

Selection evaluates evidence useful for the original question; it does not classify answerability or penalize useful repetition. No Bayesian or weighted candidate-position prior enters ranking. The statistical prior is confined to topic-boundary detection. Central sentences guide navigation and never replace evidence paragraphs.

## Validation scope and reproduction

**These are previously inspected historical questions, not an untouched confirmation set.** Shared-context selection was originally chosen on separate development papers before its historical run. The retained standard was chosen after observing subsequent experiments. This is within-paper evidence retrieval and does not establish full-corpus or frontier superiority, or a compute advantage.

Evidence alignment is unchanged: exact or unambiguous whitespace-normalized paragraph matching, best acceptable reference per metric, and no artificial credit for empty references. Complete evidence@5 means an accepted reference set is fully retrieved. Evidence F1 measures paragraph precision and recall together.

The new standard entrypoint was replayed using exact routing caches and input-keyed saved comparison scores: **all 1,456 system predictions match the retained traversal, pairwise ranking, source packet and final selection**. This verification required no new model calls.

[Standard configuration](../configs/retrieval-standard.json) · [Implementation and usage](../docs/retrieval-standard.md) · [Manifest](results/shared-context-standard/test/manifest.json) · [Statistics](results/shared-context-standard/test/statistics.json) · [Artifact catalog](results/shared-context-standard/test/catalog.json)

Run `python scripts/replay_standard_results.py` to verify archive hashes, per-question scores, exact baseline reuse and the shared selector offline. The archive contains source packets, probabilities, upstream traversal/pairwise traces and a snapshot of the runtime sources.
