# Engineering inspection of validation retrieval failures

**The matched search result is strong:** Jev tree search beats global embedding search in all four fixed split/central-sentence settings by **7.19–8.64 recall@5 percentage points**. The average difference is **+8.01 points**, with a 95% document-cluster interval of **+5.32 to +10.65**. Each conditional comparison has Holm-adjusted p = 0.0012. [Complete validation results](RETRIEVAL_V3_VALIDATION.md)

This inspection asks where the current implementation loses additional evidence. It uses the completed validation partition and saved decisions, makes no model calls, and does not inspect or modify the incomplete test. These are post-outcome diagnostics, separate from the registered comparisons. A failure location establishes what the program did; it does not by itself prove that a different setting will improve accuracy.

## Where evidence was lost

JJJ has incomplete top-five evidence on **203 of 864 eligible questions**. In each question, select the acceptable reference with highest JJJ recall@5, resolving ties by annotation order. Those references contain **310 missed paragraph instances**. The same paragraph can count again for another question.

| Observed loss location | Missed paragraph instances |
| --- | ---: |
| Heading accepted by score but outside the five-branch beam | 73 |
| Topic block accepted by score but outside the beam | 26 |
| Paragraph accepted by score but outside the beam | 110 |
| Heading below the 0.20 acceptance cutoff | 14 |
| Topic block below the cutoff | 5 |
| Paragraph below the cutoff | 12 |
| Retrieved, but ranked below fifth in the final output | 70 |
| **Total** | **310** |

Thus **209/310 instances encounter a beam limit**, compared with 31 at the acceptance cutoff. Only seven beam losses tie a retained candidate's score. All 31 threshold losses have observed scores between 0.05 and 0.20. These observations prioritize branch retention and final selection over treating the acceptance threshold as the sole problem.

Nineteen chosen references contain more than five paragraphs. Their size alone imposes at least **46 missing paragraph instances** at output k=5. Those instances are included above; they cannot all be repaired while keeping a five-paragraph output.

## Confirmed adapter defect: native heading hierarchy was flattened

QASPER stores nested section paths such as `Methods ::: Features`; examples are visible in the [dataset maintainer's release](https://huggingface.co/datasets/allenai/qasper). The evaluation adapter in `scripts/live_tree_eval.py` attaches every native section directly to the document root. Topic blocks and paragraphs are still hierarchical, but the native heading hierarchy is lost.

This affects **91/281 validation papers and 1,061 nested heading paths**. With a fixed beam of five, flattened subsections compete directly at the root instead of being explored under their parent section. That is a concrete implementation discrepancy from hierarchical root-to-leaf exploration, rather than evidence that the general tree approach is ineffective.

An isolated repair in [`retrieval_v4_structure.py`](../scripts/retrieval_v4_structure.py) restores contiguous native heading paths and inserts structural ancestors when their own section has no paragraph entry. It keeps repeated, noncontiguous headings separate and creates no model-generated content.

The offline audit verifies all **15,476 validation paragraphs**, including their IDs and exact source spans, and preserves topic groups, central sentences and source hashes. Among the 91 affected papers, median root branch count changes from **18 to 8**; across all papers, it changes from 15 to 11. Regression tests cover nested headings, missing ancestors, duplicate titles, source preservation and idempotence. **Retrieval accuracy after this repair is unmeasured.** The repair is intentionally outside the frozen v3 inference path.

## Recovered cases and what they establish

Cases below are diagnostic examples, not a representative estimate of each cause. The private review selected the first two records by question ID within specified failure categories, deduplicated overlapping selections, and separately inspected one fully omitted paragraph. Source excerpts remain in the original benchmark; the published prediction archive contains the trace IDs and scores.

| Case | Saved behavior | Engineering implication |
| --- | --- | --- |
| `qasper/0ec56e15005a627d0b478a67fd627a9d85c3920e`, p6: representational meaning of a word subspace | The Introduction branch scores **0.77**, above acceptance, but loses to five branches scoring 0.81–0.94. Direct Gemini + Jev retrieves the annotated paragraph. | A reasonably promising branch is permanently discarded. A deferred exploration queue is worth testing before making a hard cut. |
| `qasper/18942ab8c365955da3fd8fc901dfb1a3b65c1be1`, p4: origin of hotel reviews | The paragraph names TripAdvisor in a middle sentence. The refined Introduction preview includes other sentences from that paragraph, but omits that sentence; the branch scores **0.24** and is outside the beam. Direct Gemini + Jev recovers p4. | Seeing an excerpt from the correct paragraph does not mean seeing the requested fact. Centrality and outside-in excerpts need a query-directed inspection fallback. |
| `qasper/1062a0506c3691a93bb914171c2701d2ae9621cb`, p15: extracted features | The Jev-split topic block contains the complete two-sentence paragraph, scores **0.71**, and loses among 17 accepted topic candidates. EJJ, changing only the split factor, recovers the same paragraph at top five. | Splitting changes the number and composition of competing blocks. The observed failure is a split/beam interaction, not missing source text. The acceptable reference here contains five paragraphs. |
| `qasper/0682bf049f96fa603d50f0fdad0b79a5c55f6c97`, p18: analysis of particular terms | The heading scores **0.71** and loses among 39 accepted root candidates. JEJ, changing only central-sentence selection, recovers p18. | Central cues can change an individual route even though their aggregate validation replacement has no significant gain. The native heading path is also flattened in v3. |
| `qasper/03ebb29c08375afc42a957c7b2dc1a42bed7b713`, p35: pipeline evaluation | Neither the base nor refined heading card contains the short paragraph reporting evaluation corpus size. Its parent branch scores **0.23** and is excluded by the beam. | Bounded excerpts can omit an entire useful middle paragraph. This is a concrete coverage gap. |
| `qasper/01f4a0a19467947a8f3bdd7ec9fac75b5222d710`, p26: evaluation metrics | The full paragraph is evaluated at **0.84**, but five higher-scored paragraphs take the beam. The planner also narrows a general metrics question to Results and Discussion and captions. | Leaf retention and final evidence ranking are separate from branch exploration. The location restriction is visible prompt narrowing; its causal contribution to this miss is not isolated. |

Across all 118 misses at an internal node, **117 cards contain some excerpt from the missed paragraph**. This does **not** rule out cue loss: the TripAdvisor example demonstrates that the answer-bearing sentence can still be absent. Paragraph overlap is a coarse visibility diagnostic, not proof of sufficient information.

Switching only splitting recovers 24 of the same 310 missed instances at top five; switching only central-sentence selection recovers 44. These counts concern JJJ's misses and omit failures introduced by the alternative arm. The complete matched comparisons, which include both gains and regressions, remain the basis for performance claims.

## Prompt scope and annotation limitations

A systematic sample of 24 incomplete questions was inspected for question-to-plan scope, alongside the detailed cases above. The sample uses evenly spaced records after sorting distinct incomplete question IDs. Plans usually preserve the requested subject, but some introduce location or specificity restrictions absent from the original question, such as requiring explicitly designated baselines in experimental sections or metrics in a named section. The original question is still supplied to Jev; the decision instruction nevertheless prioritizes the generated evidence need.

The next controlled planner test should compare the current needs with an additional request preserving the complete original question, using the same requests for the embedding and Jev arms. This is a testable engineering hypothesis, not a measured repair. A scope audit cannot establish how a new request would score unseen branches.

Some missed annotations are also difficult to interpret as direct answers: one baseline-model reference supplies broad introductory background, and the metrics case above contains `INLINEFORM` placeholders where mathematical names are expected. The official references and source text remain unchanged. These cases caution against labeling every annotated miss as a navigation bug.

## Offline replay rejects a simplistic fix

Using only saved paragraph scores, remove terminal paragraph acceptance and beam filtering while preserving observed internal traversal, per-need ordering and reciprocal-rank fusion. No unvisited descendant receives an invented score.

| Validation diagnostic | Original JJJ | Retain all already-scored paragraphs |
| --- | ---: | ---: |
| Recall@5 | 82.21% | 80.04% |
| Recall@10 | 85.69% | 88.25% |

At top five, 24 questions improve and 47 worsen. This replay changes final candidate membership, so it is not a live rerun of a wider tree. It shows why broader exploration should feed a deliberate final paragraph-selection step instead of simply enlarging every list and keeping the same fusion.

## Next controlled engineering experiments

1. Restore native heading hierarchy, then rerun routing with new decisions. Keep the frozen v3 results intact.
2. Separate exploration beam from final paragraph count; defer promising overflow branches under a fixed decision budget instead of permanently deleting them.
3. Inspect query-relevant source sentences within a promising branch when central and outside-in cues do not cover the request. Return complete paragraphs for comparison.
4. Preserve the original question as a fallback evidence need and test scope-preserving plans under shared requests.
5. Compare final Jev reranking of collected tree paragraphs with the existing fusion, holding the candidate union and paragraph budget fixed.

Tune these changes on development, select settings on validation, and evaluate a newly registered version. These diagnostics do not justify declaring repaired performance on the current incomplete test or claiming that every remaining limitation is purely engineering.

## Reproduce

```sh
python scripts/diagnose_retrieval_v3_validation.py
python scripts/audit_retrieval_v4_structure.py
python -m unittest tests.test_retrieval_v4_structure tests.test_retrieval_failure_analysis -v
```

The scripts require the prepared local validation artifacts described in [reproduction instructions](RETRIEVAL_V3_REPRODUCTION.md). They make no API calls. [Failure-location aggregates](results/tree-retrieval-v3/validation/failure-analysis.json), [engineering diagnostics](results/tree-retrieval-v3/validation/engineering-diagnostics.json), [structure audit](results/tree-retrieval-v3/validation/structure-repair.json), and their [catalog hashes](results/tree-retrieval-v3/validation/catalog.json) provide the public audit trail.
