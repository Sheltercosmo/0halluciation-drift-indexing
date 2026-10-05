# Retrieval comparison: global embedding search and Jev tree exploration

Updated 2026-10-05. **Status: revised plan, not a completed experiment.** The study identifier is `tree-retrieval-v3`. Historical EEE/EJE/etc. results in the [live report](LIVE_TREE_REPORT.md) used a different search procedure and cannot be relabeled as these arms.

The question is whether decision-based topic blocking, central-sentence selection and search improve recovery of source evidence. A central sentence is the exact source sentence chosen to represent a paragraph or section; selecting it does not restrict retrieval to that sentence. Indexing uses no generative LLM. A query-time LLM proposes evidence needs. Embedding search ranks across tree depths; Jev explores promising branches from root to leaves.

## Eight crossed configurations

Letters always mean **split / central-sentence selection / search**. E means embedding-based; J means Jev-based. The prior and probability-drop rule are one Jev splitting method, with no separate prior ablation.

| Arm | Topic splitting | Central sentences | Search procedure |
| --- | --- | --- | --- |
| EEE | Embedding semantic cuts | Embedding centrality | Global embedding ranking across all depths |
| EJE | Embedding semantic cuts | Jev central-sentence selection | Global embedding ranking across all depths |
| EEJ | Embedding semantic cuts | Embedding centrality | Jev exploration from root to leaves |
| EJJ | Embedding semantic cuts | Jev central-sentence selection | Jev exploration from root to leaves |
| JEE | Jev prior/drop cuts | Embedding centrality | Global embedding ranking across all depths |
| JJE | Jev prior/drop cuts | Jev central-sentence selection | Global embedding ranking across all depths |
| JEJ | Jev prior/drop cuts | Embedding centrality | Jev exploration from root to leaves |
| JJJ | Jev prior/drop cuts | Jev central-sentence selection | Jev exploration from root to leaves |

Record identifiers as `tree-retrieval-v3/EEE`, etc. Run all eight on the same development questions; retain all eight frozen factorial arms for the held-out component comparison. They are fixed controls, not eight configurations from which to cherry-pick a winner.

| Factor changed | Four matched contrasts (E to J) |
| --- | --- |
| Split | EEE -> JEE; EJE -> JJE; EEJ -> JEJ; EJJ -> JJJ |
| Central sentences | EEE -> EJE; EEJ -> EJJ; JEE -> JJE; JEJ -> JJJ |
| Search | EEE -> EEJ; EJE -> EJJ; JEE -> JEJ; JJE -> JJJ |

Report all twelve contrasts, each factor's mean effect, and interactions. The search contrasts compare two practical search procedures, not only a scorer swap. Any optional same-candidate scorer diagnostic must be labeled separately; embedding search is never forced through Jev's beam for the main comparison.

## Shared index and access to content

Parse native titles, headings and contents identically, without Jev or an LLM. Under headings, vary topic splitting as the first factor. Keep original paragraph/sentence boundaries, stable source IDs and exact offsets in every arm. Never split using a query or gold evidence.

For a fixed split, build the topology once, then change only central sentences. Both selectors use the same outside-in candidate order and candidate allowance, initially eight per target; shorter nodes use all candidates. Batch independent paragraphs and topic blocks in parallel waves. Embedding centrality uses the full node's sentence vectors; Jev evaluates how well each candidate sentence represents the node's source content. Count the different indexing work explicitly.

Each node retains its native heading path, selected central sentence where applicable, child references and complete source-span references. The central sentence is an initial cue, not the only text that can be examined. Additional source sentences and full paragraph text remain accessible through a deterministic, logged content accessor. Retrieval never gets evidence annotations or reference answers.

Freeze the content-access policy on development: initial preview fields, when more context is requested, how much can be inspected per request, and how source candidates are chosen. A request for more detail can use Jev to choose query-relevant sentences. For central-sentence contrasts, keep that accessor and its scorer identical so changing the central sentence is the only intervention. For the factorial search contrasts, retain the same accessor and final extraction module; count their calls as shared downstream work in every arm. Consequently each letter names only its stated component: the third letter E specifies embedding navigation. Shared downstream extraction is reported separately, so an E search arm is not mislabeled as an entirely embedding-only pipeline.

Compare sentence and paragraph output as separate registered policies. The common paragraph policy ranks original paragraph IDs. The sentence policy selects exact query-relevant spans from candidate paragraphs, including non-central sentences. Both can read local context for references or negation. A central sentence match, heading hit or selected ancestor is not automatically successful evidence retrieval.

## One shared LLM proposal

Generate one to three evidence needs once per question and reuse the exact requests across all eight arms. The planner sees the question and shared native structure for the within-document task, without method-specific central sentences, gold annotations or reference answers. Cache and pin its model, prompt and generation settings.

The initial factorial study holds the LLM evidence requests fixed throughout search. Adaptive re-planning after reading retrieved content is a separate later experiment because paths would otherwise receive different queries. Planner failures follow a frozen original-question fallback, applied to every arm and reported.

## E search: global similarity ranking

1. Embed the requests and search the representations of headings, topic blocks, paragraphs and sentence leaves across all depths. Use exact ranking for the within-document comparison; any approximate corpus index needs a measured recall check.
2. Internal node representations include native paths and selected central sentences. Source sentence/paragraph representations remain searchable independently. Freeze representation construction, long-text handling and any combination of these views; do not silently replace the central sentence with an unrelated summary.
3. Rank globally for each need. There is no root-to-leaf beam or ancestor acceptance requirement. A strong paragraph or sentence match remains eligible even when its ancestors have low similarity.
4. Open selected nodes through the common accessor and resolve them to ranked original paragraphs or exact source sentences. Selecting a broad ancestor does not return all descendants for free; expansion work and returned text are counted.
5. Deduplicate overlapping hits, merge the need-specific lists using the same frozen rule, and apply the common output policy.

Start with 30 globally ranked node hits per need, then compare 10/30/100 on development. Overlap handling and refill rules must be fixed before inference. Report how often evidence arrived through central sentences versus direct content hits. Direct leaf access may reduce the effect of central-sentence choice; that is a legitimate result, not a reason to weaken embedding search.

Cosine similarity is a ranking score. Do not treat an affine conversion to [0,1] as a calibrated probability. Any similarity cutoff is selected separately from Jev's acceptance threshold.

## J search: keep promising nodes and explore further

Jev follows a document like a reader moving from its outline into promising sections. Its decision asks **whether the requested evidence could be somewhere beneath a node**, rather than whether the node's central sentence already answers the question.

1. Start each evidence need at the root. Examine the children of the current promising nodes.
2. Use native headings, central sentences and additional source content supplied by the accessor to evaluate those children. Batch independent needs and sibling decisions, preserving Jev's parallel execution.
3. Discard nodes below the frozen acceptance threshold. Keep the top three eligible nodes per need across that layer's frontier as the starting configuration. This is three active nodes per need, not an unconditional three children per parent.
4. Explore the children of those retained nodes at the next layer. Process reached evidence while remaining promising branches continue. Do not require every need to finish at the same depth.
5. At paragraphs, select query-relevant evidence sentences or return ranked paragraphs under the registered output policy. Central sentences do not limit which evidence can be selected.
6. Stop a branch when it reaches source evidence or no child remains sufficiently promising. Finish when all active branches terminate, the frozen evidence-completion condition is met, or a declared resource limit is reached.

The initial policy is a layer-wise beam; it does not silently revisit discarded branches. A deferred-node/backtracking policy can be a separately registered development variant, including its extra work. Do not compare uncalibrated scores from different layers as if they were one global probability scale.

Separate provider context limits from cumulative query work. Batch a large layer into valid requests and make a coherent selection after obtaining its scores; do not reject the whole layer merely because cumulative serialized payload has reached the previous experiment's 8,192-token allowance. Record every inspected token and decision. An explicit safety limit preserves already selected evidence and marks the run truncated; it must not turn partial retrieval into an unexplained empty result.

## Output, metrics and fair resource accounting

The factorial primary outcome is **QASPER evidence F1 for up to five ranked original paragraphs**. Report precision and recall at 1/3/5/10, complete evidence-set recovery at five paragraphs, and the full ranking. Score acceptable alternative evidence sets according to the frozen benchmark convention, rather than requiring the union of all annotators' sets. Report evidence-bearing and unanswerable/empty-evidence strata separately; retain failures in their applicable denominators.

Also compare complete paragraph output under 512/1,024/2,048 source-token limits. Freeze packing and oversized-paragraph handling, account for every skipped item, and report actually delivered evidence separately from pre-packing candidate recall. Use one tokenizer and count unique delivered source text once. A returned sentence merely lying inside an annotated paragraph does not establish that the sentence supports the answer.

Use a sentence-annotated benchmark for the sentence-output policy and report supporting-fact precision, recall, F1 and complete-set recovery. This establishes whether query-specific sentence selection works, rather than assuming paragraph labels are sentence labels. Generated-answer quality is an optional downstream experiment after retrieval is frozen, with one shared reader.

Match questions, accessible source collection and output allowances. Do not equate a cheap cached vector lookup with a Jev decision or force both methods to inspect the same number of nodes. Report quality-versus-work curves, indexing time/storage, query latency (median and p95), embeddings, decisions, model calls, inspected source tokens, prompt tokens and failures. Measure each deployed method's standalone logical usage even when experiment caches share work; report measured latency under stated hardware, batching and concurrency.

## Complete systems and independent hybrid

Compare the complete JJJ pipeline with reproducible embedding systems beyond the factorial EEE control:

- [Qwen3-Embedding with Qwen3-Reranker](https://github.com/QwenLM/Qwen3-Embedding), retaining the upstream retrieval/reranking interfaces.
- [ColBERTv2](https://github.com/stanford-futuredata/ColBERT), retaining its late-interaction scoring.
- A frozen current reasoning retriever, initially RTriever-4B through the [BRIGHT-Pro reference implementation](https://github.com/yale-nlp/Bright-Pro).
- [RAPTOR](https://github.com/parthsarthi03/raptor) as the hierarchical comparator, with its actual clustering and summaries. An evidence adaptation must resolve summaries to source spans and be labeled explicitly.
- BM25 and direct dense retrieval as diagnostic controls, not the sole evidence of competitiveness.

Pin code, model revisions, prompts and dependencies, and verify adapters on development before evaluation. A resource-unavailable comparator is reported as missing, never replaced by a weak approximation bearing its name. Whole-system baselines retain their native architecture. Report original-question runs and shared-proposal augmentation distinctly; do not attribute planner gains to a retriever.

The hybrid runs JJJ and direct embedding retrieval independently, then combines evidence. It does not blend cosine and Jev scores inside the Jev route. Start with reciprocal-rank fusion (constant 60), deduplicate source overlap and use the common output module. Compare hybrid, JJJ alone and dense alone at the same output limits; report both full-path and explicitly constrained cost comparisons. The dense path searches all source chunks, including branches Jev never visited.

A separate same-pool reranker experiment can compare Jev, a dedicated reranker and a generative model on identical paragraphs or identical sentence candidates. It must not compare full-passage generative judgments with central-sentence-only Jev judgments and call that model superiority.

## Data and execution sequence

[QASPER](https://arxiv.org/abs/2105.03011) is the primary within-document evidence task. The existing registry contains 3,316 development, 1,005 validation and 728 locked test questions, separated by document. The exposed 192 QASPER questions from the old 384-question mixed benchmark are development only. QuALITY has no equivalent paragraph evidence labels and is reserved for optional downstream QA.

[HotpotQA](https://hotpotqa.github.io/) supplies sentence-level supporting facts for a separately registered sentence-selection study. Its distractor setting tests selection within supplied candidates; it does not establish full-corpus retrieval performance. Freeze document-overlap controls and split assignments before inspecting evaluation outcomes.

[BRIGHT-Pro](https://github.com/yale-nlp/Bright-Pro) provides a harder external corpus-retrieval study: the prepared inventory contains 739 queries across seven domains. Use full task corpora and official nDCG/recall and aspect-coverage metrics, not a small handpicked candidate collection. The current tree method starts inside a document; an embedding first stage followed by Jev is a composite system and must be labeled accordingly. A shared top-100-pool comparison tests reranking/refinement, not pure Jev corpus retrieval. Freeze the corpus entry method before claiming an end-to-end corpus comparison.

Execution order:

1. Implement global E search, Jev promising-frontier traversal and full content access; verify offsets, non-central evidence access, overlap handling, partial returns and shared plans on development fixtures.
2. Validate upstream baseline adapters and measure resource requirements. Use a small development pilot only for software correctness and timing.
3. Run all eight crossed arms over the registered QASPER development set. Tune on document-grouped development folds with a bounded, recorded search. Inspect central sentence, splitting, traversal, extraction and fusion failures there.
4. Freeze shared stage settings, all eight factorial arms, complete-system baselines, metrics and at most three tuned system finalists. Update the retrieval validation gate before opening validation.
5. Run all 1,005 validation questions; select finalists by the registered retrieval rule without case-level debugging or adding candidates. Keep all eight factorial arms in the final comparison.
6. Freeze the selected system configuration and evaluate once on all 728 locked test questions, alongside the eight component arms and required baselines. Publish paired document-bootstrap intervals, adjusted tests for registered contrasts and every outcome, including regressions.
7. Confirm transfer on separately frozen sentence/corpus tracks. Optional reader QA follows the retrieval study.

Gold labels enter scoring only. Record every requested question, including failed and truncated runs. Do not repeatedly tune against a spent test set until significance appears.

## Parameter selection and implementation status

Tune split separation, central-sentence stopping, Jev branch acceptance and global embedding candidate count as distinct parameters. A central-sentence score of 0.90 can stop outside-in central-sentence selection; it is not a retrieval stopping rule or evidence of a 90% success rate. The parent's normalized split-separation score is lower-is-better; its exact probability scopes and adapter remain prerequisites for live use.

The [threshold note](THRESHOLD_SEARCH.md) preserves the older sampled settings as historical material. It is not the executable configuration for this revision. Jev's initial beam is three; embedding search has a global candidate count, not a beam. Give each factor a declared development tuning allowance and then freeze one shared setting across all arms that use it, so retuning another module does not contaminate a matched contrast.

Existing code still implements the historical bounded traversal and answer-based validation gate. This plan requires implementation and a fresh manifest before any v3 inference. No new results or held-out claims are implied by this document. Preserve historical registrations, code snapshots and scores unchanged. Public reports should contain scientific settings and aggregate results; private operational spending records remain separate.
