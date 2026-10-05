# Query evidence planning and complementary Jev selection

This is a development experiment on the same 128 exposed questions used in the isolated-reader comparison. It opens no validation or test data. Register the manifest and implementation on GitHub before any model inference. Report both new methods and the historical no-planner controls, including regressions.

## Hypothesis from observed failures

The prior run omitted useful passages that were already in the 8,192-token candidate pool: a comparison-model paragraph, a character's presence during a plan, and the final transition of a ship's route. More candidate exposure alone will not recover evidence ranked too low to reach the reader. Explicit evidence needs may make those passages easier for Jev to recognize, and a coverage objective may prevent redundant passages from filling the final context.

The query-time LLM sees only the question, answer options and document title. It proposes one to three distinct evidence needs, each at most 300 characters, without answering the question or inventing facts. One plan is shared across the Jev and Codex methods. The planner cannot access document passages, gold answers, prior failure labels or other questions. Public-benchmark pretraining exposure cannot be ruled out.

## Registered comparison

| Component | Jev planned coverage | Codex planned reranking |
| --- | --- | --- |
| Source index | Existing Jev decision-defined topic chunks | Existing recursive chunks |
| Candidate pool | Frozen 8,192 source tokens, at most 128 candidates | Same allowance, frozen recursive pool |
| Query plan | Shared Codex plan | The identical plan |
| Ranking | Jev scores each need/passage pair independently | Codex ranks passages using the question and needs |
| Context | Greedy new-need coverage plus saved original-query relevance, with a passage-length penalty | Original whole-passage packing in the returned order |
| Final reader | Unchanged isolated Codex reader, original question and final source context | Identical reader and prompt |
| Final source budget | 2,048 rendered tokens, including headings and separators | Identical |

For Jev, maintain the largest selected score for each need. A candidate's coverage gain is the mean increase in these maxima. Greedy utility is `(0.6 * coverage_gain + 0.4 * original_relevance) / sqrt(source_tokens)`. Ties use original relevance, then the frozen candidate order. Consider only whole passages that fit the actual rendered context limit, and return selected passages in document order. Stop if nothing fits or every remaining utility is zero. Scores guide selection; the coverage objective is not a calibrated probability model.

Need/passage decisions share query, need and source text in bounded Jev requests, preserving independent parallel scoring. The topic-blocking prior and sudden-drop rule are unchanged. There are **zero new indexing, embedding or Gemini calls** in this trial.

The Codex ranking prompt already requests complementary evidence. It now also receives the same explicit plan, so the planner's extra information is available to both competitors. Candidate counts, source tokens, plans, Jev score matrices, selection traces, raw model responses, attempts and usage are retained. A single question is used in every planning, reranking and reader call.

## Integrity, scoring and cost

All 128 questions must receive both predictions: 256 method/question rows. Identical reader ID/query/context inputs share a prediction, including matches in the completed isolated-reader run. Otherwise, use one reader draw; no best-of sampling or answer-based retries are allowed. The reader prompt and schema remain byte-compatible with the previous run.

The new client records usage before validating output and preserves every raw JSON response. Invalid structure may be retried unchanged once, with the two-attempt limit retained across process resumes. Transport/provider failures stop the stage for inspection. Tool use invalidates a response. Rejected attempts are charged to the internal call reservation; no difficult question is removed.

The expected maximum without structural retries or cache reuse is 128 planner calls, 128 Codex ranking calls and 256 reader calls. The internal total Codex research ceiling increases from 1,000 to **1,300** calls, from 629 already reserved, and will stop the run if exhausted. This is an operational limit, not a dollar-cost estimate or a promise of account quota. The user's **$30 Gemini cap** and existing $5.410937 conservative reservation remain unchanged. Jev requests and individual decisions retain their existing limits and audited reservations.

Freeze the shared planner model/prompt/schema, source implementation, prepared pools, source documents and Codex binary checksum. Score QASPER with the official evaluator and QuALITY with exact option accuracy. Report evidence recall and paired document-bootstrap intervals alongside answer scores. These are development observations; they cannot establish confirmatory significance or authorize test access.

```sh
python scripts/planned_evidence_trial.py prepare
# Publish the manifest and frozen source before inference.
python scripts/planned_evidence_trial.py run
python scripts/planned_evidence_trial.py score
```

Runtime setup uses the same evaluation dependencies and authenticated Codex CLI as the previous run. Jev credentials come from `TYPESAFE_API_KEY`; credentials are removed from Codex child-process environments and never enter saved manifests.
