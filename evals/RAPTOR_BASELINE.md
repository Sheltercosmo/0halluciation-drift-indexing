# RAPTOR comparator: adapter verified, QA comparison pending

We use the [official RAPTOR implementation](https://github.com/parthsarthi03/raptor), pinned to commit `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`. Its [ICLR 2024 paper](https://arxiv.org/abs/2401.18059) studies recursive clustering, summaries and retrieval across abstraction levels. This is an applicable published-method comparator; beating it alone would not establish a current frontier result.

The adapter runs the upstream clustering, tree builder and collapsed retriever code. An offline integration check passed with a synthetic embedding callback and an extractive summary callback: 24 leaves became 28 total nodes, the fixed-seed build reproduced exactly, JSON serialization preserved the tree, and retrieval preserved the upstream ranking. These fixture callbacks are only for software verification. **No RAPTOR QA score or live-model comparison is available yet.** [Verification record](results/raptor-adapter-v1/verification.json)

## Fidelity and explicit adaptations

| Component | Implementation and status |
| --- | --- |
| Upstream source | Fifteen original files, including the MIT license; every byte length and Git blob hash is checked against the [source lock](frontier/raptor-source-lock.json). Downloaded code stays in `output/vendor/raptor`. |
| Clustering | Unmodified upstream global/local UMAP, GMM selection by BIC, soft assignment threshold 0.1, reduction dimension 10, recursive cluster token limit 3,500. The upstream cluster-count search excludes its upper endpoint. |
| Tree | Unmodified sentence splitter and recursive builder, 100-token leaf target, 100-token summary limit, at most five layers, and upstream stopping condition at 11 or fewer nodes. The splitter strips sentence delimiters and can exceed its target on unsplittable phrases; it is not our paragraph parser. |
| Providers | Original abstract embedding and summarization classes are loaded directly from their AST. Explicit audited callbacks replace providers. Unused SBERT/Torch/QA imports are skipped, and implicit default providers are rejected. Core algorithm modules are imported without edits. |
| Reproducibility | NumPy/Python seed 224, serial node construction, one numerical thread, original NumPy 1.26.3 and UMAP 0.5.5. Exact repeatability was checked within the recorded environment, not promised across hardware. |
| Tokenizer | `cl100k_base` through tiktoken 0.12.0, matching the main evaluation runtime; upstream originally pinned tiktoken 0.5.1. |
| Retrieval | Cosine ranking over every node in the collapsed tree, preserving upstream's stop-at-first-nonfitting-node rule. The adapter requests all nodes rather than accepting the convenience API's default top-10 cap. |
| Shared context limit | Upstream counts node tokens but omits separators. The wrapper measures the final rendered context and removes whole nodes from the end until it fits 2,048 tokens. Ranking is preserved; this budget correction is recorded per query. |
| Provenance | Trees store original node text, embeddings, children, layer IDs and selected node IDs in JSON. A generated summary's descendant coverage must not be reported as evidence actually present in the reader's context. |

Any live comparison will be labeled **RAPTOR with matched model adapters**, because replacing the paper's models changes the system. It will use the same evaluation reader, question set and final context budget as the Jev pipelines. Embedding and summarization models, preprocessing calls, token usage, failures and cache reuse must be recorded before running it. RAPTOR's generative indexing cost is part of that comparison.

## Reproduce the offline check

Use a separate Python 3.12 environment; these dependencies are evaluation-only.

```sh
python -m pip install -r evals/requirements-raptor-lock.txt
python scripts/fetch_raptor.py
python scripts/verify_raptor_adapter.py
```

The check exercises real upstream clustering with fixture vectors and summaries, exact tree replay, collapsed ranking, separator-aware context limits and rejection of implicit providers. It does not call a model API or open validation/test data. The resolved [dependency lock](requirements-raptor-lock.txt) and [direct requirements](requirements-raptor.txt) are separate from the project's dependency-free runtime.

Before a measured comparison, the remaining work is to connect audited live summarization/embedding callbacks, preregister the exposed-development run and its model substitutions, generate all predictions with the common isolated reader, and replay its metrics. Validation and the final test remain locked until candidate selection is registered and completed.
