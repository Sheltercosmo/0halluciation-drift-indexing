# Retrieval comparison: global embedding search and Jev tree exploration

Updated 2026-10-05. **Status: implemented and calibrated on development; confirmatory settings frozen before validation.** The study identifier is `tree-retrieval-v3`. Historical EEE/EJE/etc. results in the [live report](LIVE_TREE_REPORT.md) used a different search procedure and cannot be relabeled as these arms.

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

The frozen accessor initially exposes the native path, central sentence and at most six child central sentences. Paragraph and sentence nodes expose their full source text. Jev internal-node scores below 0.85 trigger another inspection containing up to eight additional non-central sentences, spread outside-in across descendant paragraphs, before pruning. This also applies to initially low-scoring nodes. Internal previews are bounded and can miss detail; full paragraph text is used when a paragraph is reached. Every excerpt retains exact source offsets. The same accessor is fixed across central-sentence contrasts.

**Every compared system returns whole original paragraphs.** Sentence hits resolve to their parent paragraph; the main study does not return or score sentence snippets. Central sentences are index cues, not the only available evidence. There is no shared Jev reranker applied to E arms. E ranks source representations globally; J makes branch and paragraph decisions. Their matched contrasts compare these complete search procedures, with different work reported explicitly.

## One shared LLM proposal

Generate one to three evidence needs once per question and reuse the exact requests across all eight arms. The planner sees the question and shared native structure for the within-document task, without method-specific central sentences, gold annotations or reference answers. The final shared planner is `gpt-6.1-sol`, low reasoning effort, through an isolated tool-free Codex invocation. Eight independent questions are batched per request; each receives one to three validated needs. Exact responses are cached and reused across the eight arms, Gemini dense control and independent hybrid.

The initial factorial study holds the LLM evidence requests fixed throughout search. Adaptive re-planning after reading retrieved content is a separate later experiment because paths would otherwise receive different queries. Planner failures stop the affected case for a logged retry; there is no silent fallback or dropped question.

## E search: global similarity ranking

1. Embed the requests and search the representations of headings, topic blocks, paragraphs and sentence leaves across all depths. Use exact ranking for the within-document comparison; any approximate corpus index needs a measured recall check.
2. Internal node representations include native paths and selected central sentences. Source sentence/paragraph representations remain searchable independently. Freeze representation construction, long-text handling and any combination of these views; do not silently replace the central sentence with an unrelated summary.
3. Rank globally for each need. There is no root-to-leaf beam or ancestor acceptance requirement. A strong paragraph or sentence match remains eligible even when its ancestors have low similarity.
4. Open selected nodes through the common accessor and resolve them to ranked original paragraphs. Selecting a broad ancestor does not return all descendants for free; expansion work and returned text are counted.
5. Deduplicate overlapping hits, merge the need-specific lists using the same frozen rule, and apply the common output policy.

The development sweep compared 10/30/100 globally ranked hits per need and froze 30. Internal hits open the three highest globally scored descendant paragraphs; sentence hits open their original parent paragraph. Need-specific deduplicated rankings are fused with equal reciprocal-rank fusion, constant 60. Overlap handling and refill rules must be fixed before inference. Report how often evidence arrived through central sentences versus direct content hits. Direct leaf access may reduce the effect of central-sentence choice; that is a legitimate result, not a reason to weaken embedding search.

Cosine similarity is a ranking score. Do not treat an affine conversion to [0,1] as a calibrated probability. Any similarity cutoff is selected separately from Jev's acceptance threshold.

## J search: keep promising nodes and explore further

Jev follows a document like a reader moving from its outline into promising sections. Its decision asks **whether the requested evidence could be somewhere beneath a node**, rather than whether the node's central sentence already answers the question.

1. Start each evidence need at the root. Examine the children of the current promising nodes.
2. Use native headings, central sentences and additional source content supplied by the accessor to evaluate those children. Batch independent needs and sibling decisions, preserving Jev's parallel execution.
3. Discard nodes below the frozen acceptance threshold. Keep the top five eligible nodes per need across that layer's frontier, at acceptance 0.20. The development sweep compared beams three/five and acceptance 0/0.20/0.40. This is five active nodes per need, not five children per parent.
4. Explore the children of those retained nodes at the next layer. Process reached evidence while remaining promising branches continue. Do not require every need to finish at the same depth.
5. At paragraphs, assess the full source paragraph and return ranked original paragraphs. Central sentences do not limit which evidence can be selected.
6. Stop a branch when it reaches source evidence or no child remains sufficiently promising. Finish when all active branches terminate, the declared resource limit of 4,096 scored node decisions per question is reached.

The initial policy is a layer-wise beam; it does not silently revisit discarded branches. A deferred-node/backtracking policy can be a separately registered development variant, including its extra work. Do not compare uncalibrated scores from different layers as if they were one global probability scale.

Separate provider context limits from cumulative query work. Batch a large layer into valid requests and make a coherent selection after obtaining its scores; do not reject the whole layer merely because cumulative serialized payload has reached the previous experiment's 8,192-token allowance. Record every inspected token and decision. An explicit safety limit preserves already selected evidence and marks the run truncated; it must not turn partial retrieval into an unexplained empty result.

## Output, metrics and fair resource accounting

The factorial primary outcome is **aligned paragraph evidence recall at five ranked original paragraphs**. F1, precision and complete evidence-set recovery are secondary; this avoids rewarding a method primarily for returning fewer paragraphs. This paragraph-identity metric is an explicit adaptation of QASPER evidence scoring, not a renamed official score. Report precision and recall at 1/3/5/10, complete evidence-set recovery at five paragraphs, and the full ranking. Score acceptable alternative evidence sets according to the frozen benchmark convention, rather than requiring the union of all annotators' sets. Report evidence-bearing and unanswerable/empty-evidence strata separately; retain failures in their applicable denominators.

Also compare complete paragraph output under 512/1,024/2,048 source-token limits. Freeze packing and oversized-paragraph handling, account for every skipped item, and report actually delivered evidence separately from pre-packing candidate recall. Use one tokenizer and count unique delivered source text once. A returned sentence merely lying inside an annotated paragraph does not establish that the sentence supports the answer.

Match gold evidence by exact text or an unambiguous whitespace-normalized match to an original paragraph. An acceptable annotation must map every non-figure text evidence item and contain at least one paragraph. A question needs at least one such annotation for the primary denominator. Heading-only, ambiguous or unmapped evidence is reported separately; it is not assigned to arbitrary descendants. Keep the raw official string F1 separately for transparency. Empty/unanswerable references do not receive perfect retrieval credit. Each metric takes the best acceptable annotation; references are never unioned. No answer-generating reader runs in this study.

Match questions, accessible source collection and output allowances. Do not equate a cheap cached vector lookup with a Jev decision or force both methods to inspect the same number of nodes. Report quality-versus-work curves, indexing time/storage, query latency (median and p95), embeddings, decisions, model calls, inspected source tokens, prompt tokens and failures. Measure each deployed method's standalone logical usage even when experiment caches share work; report measured latency under stated hardware, batching and concurrency.

## Complete systems and independent hybrid

Compare the complete JJJ pipeline with reproducible embedding systems beyond the factorial EEE control:

**Same-Jev control:** direct Gemini paragraph retrieval followed by Jev reranking uses the same shared LLM requests as JJJ. This comparison holds the decision model and query planner fixed while comparing direct retrieval against tree exploration. The [pre-outcome amendment](registrations/tree-retrieval-v3-jev-rerank-amendment.json) adds this control plus Jev replacements for the Qwen and BGE rerankers. Each replacement receives exactly its corresponding dense retriever's top 30 original paragraphs, with native headings and full paragraph text. Qwen/BGE replacements retain the original-question query policy. All outputs remain whole paragraphs.

- [Qwen3-Embedding-0.6B with Qwen3-Reranker-0.6B](https://github.com/QwenLM/Qwen3-Embedding): exact whole-paragraph dense retrieval and its top-30 dedicated reranker, using upstream pooling, query instructions and yes/no scoring.
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) with [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3): dense retrieval and its top-30 reranker, using published checkpoints and pooling.
- Gemini Embedding 2 direct paragraph retrieval, plus BM25/dense RRF as diagnostic controls.

These are actual pinned checkpoint pipelines, run locally on an RTX 5060 Laptop GPU. Qwen uses compact 0.6B models; this does not establish parity with larger Qwen models or frontier corpus retrievers. ColBERTv2, RTriever-4B and RAPTOR are **not included in this frozen run** and must not be implied by its results. They remain future external comparisons.

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
3. Run a deterministic 16-question, 16-document development pilot drawn only from the already exposed development allocation. Compare six prespecified shared-search settings, inspect implementation failures there, then repeat the selected setting with the final shared planner. This small calibration is not a large development benchmark or evidence of statistical superiority.
4. Freeze shared stage settings, all eight factorial arms, complete-system baselines, metrics and at most three tuned system finalists. Update the retrieval validation gate before opening validation.
5. Run all 1,005 validation questions; select JJJ or the independent hybrid by recall@5, then complete@5, then F1@5, then method ID without case-level debugging or adding candidates. Keep all eight factorial arms in the final comparison.
6. Freeze the selected system configuration and evaluate once on all 728 locked test questions, alongside the eight component arms and required baselines. Publish paired document-bootstrap intervals, adjusted tests for registered contrasts and every outcome, including regressions.
7. Confirm transfer on separately frozen sentence/corpus tracks. Optional reader QA follows the retrieval study.

Gold labels enter scoring only. Record every requested question, including failed and truncated runs. Do not repeatedly tune against a spent test set until significance appears.

## Parameter selection and implementation status

This run freezes the existing split settings and eight outside-in central candidates without early stopping. Only branch width/acceptance and global embedding candidate count were calibrated on development. A central-sentence score of 0.90 can stop outside-in central-sentence selection; it is not a retrieval stopping rule or evidence of a 90% success rate. The parent's normalized split-separation score is lower-is-better; its exact probability scopes and adapter remain prerequisites for live use.

The [threshold note](THRESHOLD_SEARCH.md) preserves the older sampled settings as historical material. It is not the executable configuration for this revision. The frozen Jev beam is five; embedding search has a global candidate count, not a beam. Give each factor a declared development tuning allowance and then freeze one shared setting across all arms that use it, so retuning another module does not contaminate a matched contrast.

The executable implementation is `zero_index/evidence_search.py` with `scripts/retrieval_v3.py`. The [original immutable registration](registrations/tree-retrieval-v3.json) is preserved. The Jev-reranker amendment was added at the user's request during indexing/planning, before any validation outcome was inspected. It expands coverage to 18 methods. `scripts/retrieval_v3_jev_controls.py` implements the added controls and amended test gate. Integrity checks require every registered question/method, identical reranker candidate pools, valid paragraph IDs, matching manifests and exact whole-paragraph packing before test access.

The confirmatory study uses 10,000 paired document-cluster bootstrap draws and document-level paired sign randomization, with seed 20261005. Question-weighted means retain document clustering. Apply Holm adjustment separately across the 12 component contrasts, ten selected-system comparisons and two native reranker replacement comparisons. Report all arms and contrasts, including regressions. A two-percentage-point recall gain is the prespecified practical target; neither significance nor a frontier result is guaranteed.

Historical registrations and results remain unchanged. Public reports contain scientific settings and aggregate results; private operational records remain separate.
