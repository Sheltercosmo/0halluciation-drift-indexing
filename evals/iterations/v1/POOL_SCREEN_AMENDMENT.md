# Candidate-pool screen: ranking format amendment

Registered on 2026-10-05, after reranking and **before any reader predictions**. The 128 development questions, all candidate pools, Jev outputs, reader prompt and token budgets remain fixed.

One two-case Codex reranking batch failed the complete-permutation validator twice. The last response omitted one of 14 candidate IDs for one case; the other case returned all 13 IDs. Both failed attempts remain counted. This is a formatting failure, not an answer-score-based retry or a reason to remove either question.

The deterministic recovery rule preserves returned valid unique IDs in their original order and appends missing IDs in the original hybrid retrieval order. Duplicates and out-of-range IDs remain errors. Apply the rule to all incomplete ranking responses in this screen. Record the raw order, completed order and appended IDs before answer generation. This uses no gold labels, question replacements or extra model call. Already valid permutations are unchanged.

The amended development comparison must disclose this recovery. It is not an untouched confirmatory run. Before validation, incorporate a bounded failure policy into the frozen harness and test it. The originally registered source and both rejected request records remain preserved.
