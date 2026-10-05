# Sample search for separation, acceptance and early stopping

This is a development-only proposal. It makes no claim that 0.90 is optimal or calibrated. The existing document-disjoint validation/test partitions remain closed. The prior and sudden-drop rule stay a combined method; this is parameter selection, not a separate prior ablation.

## A lower separation score favors a split

Using the proposed cross-minus-within term as the leaf/group signal, define:

\[
L=p_{\mathrm{leaf,cross}}-p_{\mathrm{leaf,within}},\qquad
R=\frac{L}{p_{\mathrm{parent,cross}}},\qquad
S=\min(1,\max(0,R)).
\]

When the leaf signal equals the parent reference, **S = 1**: the candidate split has no separation gain by this criterion. Lower values indicate better separation. Accept a candidate split when **S ≤ τ**, with τ below 1. A numerator above the parent reference saturates at 1; a negative numerator saturates at 0. Preserve the signed raw ratio and every input probability so clipping remains visible. This bounded normalization is an experimental score, not a probability or a Bayesian posterior.

For example, cross = 0.40, within = 0.30 and parent cross = 0.50 give S = 0.20. A cutoff of 0.30 accepts that candidate. Cross = 0.75, within = 0.25 and parent cross = 0.50 give S = 1, so the same cutoff rejects it.

If the parent reference is below 0.05, the helper returns `insufficient_parent_reference`, with no score or cut decision. It does not divide by a substituted epsilon or silently count the candidate as a good split. Report such cases. The 0.05 floor is a fixed starting guard, not a learned result.

```python
from zero_index import normalized_separation

decision = normalized_separation(
    cross_probability=0.40,
    within_probability=0.30,
    parent_cross_probability=0.50,
    threshold=0.30,
)
# Returns score, raw_ratio, all components and accept_split.
```

The helper is implemented, but does **not** automatically alter existing trees. Before live use, define which source spans form the child groups and parent reference, which Jev judgments produce the probabilities, and how multiple leaf judgments are aggregated. Use comparable tasks, evidence scopes and aggregation across candidate boundaries. Measure parent probabilities; do not copy unrelated query relevance or confidence scores. Native headings remain hard boundaries. Freeze the cut adapter and treatment of insufficient-reference cases before inference.

This split criterion is distinct from query routing acceptance, where a high relevance score justifies visiting a branch.

## Stop outside-in search once a strong candidate appears

Stopping is disabled by default. Enable it in Python or the CLI:

```python
from zero_index import Config, build_index

index = build_index(source, scorer=jev,
                    config=Config(sentence_budget=8, sentence_stop_threshold=0.90))
```

```sh
python -m zero_index build document.md --scorer jev --provider typesafe --sentence-budget 8 --sentence-stop-threshold 0.90 -o output/tree.json
```

All active paragraphs and topic sections send their current outside-in wave together. If a target's best returned score is at least the threshold, its later waves are omitted. Other targets continue. A section stopping does not stop its paragraphs. All candidates already sent in the wave still count and participate in choosing the winner. Ties retain outside-in order; candidate limits still apply.

A paragraph wave normally contains its next first/last pair. A topic-section wave contains the next outer candidates from each paragraph, up to that section's candidate budget. With stopping disabled, the existing implementation coalesces all depths. Enabling stopping can reduce decisions and tokens while requiring more sequential request rounds; measure latency rather than assuming it improves.

Saved representatives expose `candidate_indices`, `comparisons`, `exhaustive`, `stop_threshold`, `threshold_reached`, `early_stopped` and `stop_reason`. A threshold reached on the last available candidate is not labeled as saved work. Singleton nodes need no model call. Reaching 0.90 means accepting a model score; it does not prove that no better sentence exists or that correctness is 90%.

`reselect_representatives(..., stop_threshold=0.90)` supports fixed-tree controls. Embedding centrality can use the same stopping interface but still embeds every context sentence to compute its centroid. Jev and cosine-derived thresholds have different calibration and cost implications.

## A reproducible small search

The [sampled plan](iterations/v2/threshold-search-plan.json) contains **13 configurations**, chosen before outcomes: five anchors and eight samples without replacement, with seed **20261005**. It samples this 128-configuration space:

| Parameter | Values |
| --- | --- |
| Split-separation cutoff | Disabled, 0.15, 0.30, 0.50 |
| Representative stopping threshold | Disabled, 0.85, 0.90, 0.95 |
| Tree beam width per evidence need | 2, 4 |
| Tree routing acceptance threshold | Disabled, 0.30, 0.50, 0.70 |

The anchors are the unchanged reference, stopping-only at 0.90, separation-only at 0.30, routing-acceptance-only at 0.50, and separation 0.30 plus stopping 0.90. Other settings stay fixed: eight representative candidates per target, parent floor 0.05, at most 256 routed node scores, 8,192 routing-payload tokens and 2,048 final reader tokens. Disabled separation retains the existing prior/drop rule. Configurations enabling separation need the frozen split adapter before they can run.

```sh
python scripts/sample_search_parameters.py --seed 20261005 --sampled 8 --output output/threshold-search-plan.json
```

`TreeSearchConfig(acceptance_threshold=0.50)` rejects previews below 0.50 before beam selection. If every branch is rejected, search returns `no_accepted_branches`; it does not quietly choose a rejected branch. This option is separate from representative stopping and split acceptance.

## Compare quality with actual cost

Use document-grouped folds drawn only from exposed development data. Evaluate complete groups of cases for every configuration, retaining failures in their denominators. Inspect development failures and retain a small set offering different quality/cost tradeoffs. Measure official QASPER F1, evidence recall and QuALITY-HARD accuracy alongside Jev decisions, requests, tokens, embeddings and elapsed time. Report activation rates and lost evidence. Do not optimize merely for agreement with the full search's selected sentence.

The 13 configurations are a proposal, not an uncosted live sweep. Publish the complete run manifest and reserve requests before inference under the existing $30 Gemini cap. Reuse only exact compatible inputs. Fixed-score replay can cheaply estimate policy effects, but live batch composition can change Jev scores and replay cannot establish actual latency or prospective accuracy. Confirm finalists with their actual execution policy.

After development, freeze at most three candidates and the strong baselines under the [validation gate](ITERATION_PROTOCOL.md). Validation selects among them; test opens once afterward. Keep the fixed [component comparison](TREE_SYSTEM_PROTOCOL.md) distinct from tuning. The hybrid continues to combine complete Jev retrieval with independent direct embedding retrieval.
