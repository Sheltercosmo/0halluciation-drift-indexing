# Development, validation and test for evidence retrieval

Updated 2026-10-05. The active design is [tree-retrieval-v3](TREE_SYSTEM_PROTOCOL.md): all eight split / central-sentence selection / search combinations, global embedding search across tree depths, and Jev exploration of promising nodes from root to leaves. Retrieval of source evidence is the primary outcome. The statistical prior and drop rule remain one splitting method.

**Status:** existing document assignments remain authoritative and held-out outcomes remain unopened. The current Python gate implements the earlier answer-based experiment. It must be extended and verified for this retrieval design before opening validation; changing this document does not change the executable gate.

## Preserve document-disjoint assignments

The published bounded-v1 questions and their documents are exposed development data. Their original benchmark split names do not make them held out after failure analysis. Additional QASPER training papers are development material. All QuALITY development articles are also development material.

The existing assignment uses seed 20261005, canonical document IDs, normalized text hashes and normalized titles. Documents matching development material are excluded from validation/test; validation duplicates take precedence over test. Keep the registry and source checksums unchanged. These checks establish detectable document separation, not absence of public benchmark material from model pretraining.

| Dataset | Development questions | Validation questions | Locked test questions |
| --- | --- | --- | --- |
| QASPER | 3,316 | 1,005 | 728 |
| QuALITY-HARD | 1,065 | 420 | 831 |

QASPER development combines training with exposed test papers; validation uses the official development release; locked test uses remaining unexposed test papers after duplicate removal. QuALITY uses a custom document-held-out allocation and has no equivalent paragraph evidence labels. It is reserved for optional downstream QA, not the retrieval selection objective.

HotpotQA sentence selection and BRIGHT-Pro corpus retrieval require their own frozen task/split registrations. A supplied-candidate task and a full-corpus task must not be pooled into one retrieval score.

## Develop and freeze the complete comparison

Inspect failures and tune only on development. Use document-grouped development folds. Record the sampled parameter configurations and tuning allowance before each sweep; keep all outcomes. Software pilots establish correctness and timing, not benchmark superiority.

The factorial study contains all eight EEE/EJE/EEJ/EJJ/JEE/JJE/JEJ/JJJ arms. Freeze shared stage settings and retain every arm as a fixed component comparator. Separately register at most three tuned complete-system finalists, including the independent-path hybrid where applicable. Register the required published baselines with upstream commits, model revisions and adapter hashes.

Before validation, freeze:

- Data registry, implementation and source hashes, source-ID mapping and scorer version.
- All eight component arms, required baselines and up to three selectable system finalists.
- Query planner, shared requests, failure fallback and generation settings.
- Source-access policy, global embedding search settings, Jev frontier/acceptance settings, and final extraction/packing policies.
- Primary retrieval score, denominators, acceptable-reference handling, secondary metrics, planned contrasts and tie-breaks.
- Output limits, resource limits, failure/truncation accounting and latency measurement procedure.

The proposed selection rule is highest mean QASPER evidence F1 for up to five original paragraphs among the selectable system finalists. Break exact ties by lower mean standalone retrieval model tokens, then configuration hash; publish actual latency and other work separately. All finalists use the same tokenizer and usage accounting. Empty-evidence cases are reported separately from evidence-bearing retrieval; answerability conventions and failure handling must be frozen before labels are scored.

Validation selects among already registered finalists. Do not inspect its individual failures, change candidates or retune thresholds after seeing aggregate outcomes. The fixed eight component arms are all retained regardless of validation ranking.

## Retrieval gate requirements

Extend the gate to accept fixed component arms separately from selectable finalists and to require retrieval predictions without requiring reader answers. It must verify complete coverage for all registered questions and methods before scoring, with explicit failure records, source IDs/spans, pre-packing ranks, delivered evidence and standalone usage.

Use the hashed evidence scorer to recompute primary validation scores from predictions; never trust caller-supplied aggregates. Freeze selection deterministically, recording its evidence and hashes. The test gate must verify the selected configuration and the complete validation artifacts before releasing test paths.

Do not run the existing answer-based `open-validation` or `freeze-winner` workflow for this study until these changes and checks are implemented. Existing test-access protections and document assignments still apply. Registration is an auditable workflow control, not filesystem access control.

Test is opened once after final freezing. No test failure inspection or tuning occurs until all registered predictions and reporting are complete. Once exposed, it is spent for later development; a subsequent research round needs new untouched evaluation data.

## Scoring and claims

Primary: QASPER evidence F1 for up to five ranked original paragraphs, with precision, recall and complete-support recovery reported separately. Preserve alternative valid evidence sets rather than treating every annotator's evidence as jointly required. Keep evidence-bearing and empty/unanswerable strata separate and report their counts. Failed predictions remain in the corresponding denominator.

Secondary: retrieval at other paragraph counts, delivered evidence under 512/1,024/2,048 source-token caps, supporting-fact accuracy on sentence-labeled data, corpus ranking metrics on the corpus track, and quality-versus-work curves. A sentence's membership in a gold paragraph does not establish that it is supporting evidence.

Use paired bootstrap confidence intervals clustered by document. Pre-register primary component contrasts and system comparisons, with multiplicity adjustment within the declared families. Estimate detectable effect sizes from development and available document counts before claiming the test can establish a small gain. Practical gain targets and any efficiency noninferiority margin must be fixed before validation; they must not be selected after inspecting test outcomes.

Report every frozen arm and losses. Compare complete systems on the same source collection and output limits, retaining upstream architectures and recording indexing/query work. A corpus method using an embedding first stage is reported as a composite method. A same-pool reranker win is not a full-corpus retrieval win.

Answer generation is optional after retrieval configurations are frozen and uses the same reader and output policy across systems. It does not select the primary retrieval winner. No frontier claim follows solely from improvement over the old implementation or one local baseline.

## Reproducibility and public reporting

Reuse only exact content-addressed inputs and compatible model outputs. Each run records configuration, code/data/model hashes, question IDs, source spans, traces, failures, metrics and actual usage. Software tests verify index independence, non-central evidence access, global embedding eligibility, Jev branch retention, token packing and complete resumes.

Keep historical registrations and scores immutable. Public reports identify the dataset scope, methods, metrics, uncertainty and implementation limits. Private operational ledgers and personal spending details do not belong in the scientific narrative.
