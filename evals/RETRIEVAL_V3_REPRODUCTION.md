# Reproduce the paragraph retrieval study

Use the exact [registration](registrations/tree-retrieval-v3.json), [protocol](TREE_SYSTEM_PROTOCOL.md) and document registry in `evals/iterations/v1/registry.json`. All compared outputs are original paragraphs. Source hashes enforce the registered implementation; do not edit the gate, scorer or inference code in the middle of a run.

## Environment

Python 3.11+, the [evaluation requirements](requirements-retrieval-v3.txt), CUDA 12.8 and an NVIDIA GPU are required by the local Qwen/BGE adapters. The run used an RTX 5060 Laptop GPU with 8 GB memory. Inference uses FP16 and SDPA attention; normalized embedding vectors and reranker scores are converted to FP32. The main library remains independent of these evaluation dependencies.

Download these published checkpoints into the named directories under `output/retrieval-v3-models/`, at the immutable revisions in the registration:

| Directory | Hugging Face repository |
| --- | --- |
| `qwen-embed` | `Qwen/Qwen3-Embedding-0.6B` |
| `qwen-rerank` | `Qwen/Qwen3-Reranker-0.6B` |
| `bge-embed` | `BAAI/bge-m3` |
| `bge-rerank` | `BAAI/bge-reranker-v2-m3` |

Include tokenizer and model configuration files. Qwen uses `model.safetensors`; the pinned BGE embedding checkpoint uses `pytorch_model.bin`. No remote model code is enabled. The adapter rejects inputs longer than 8,192 model tokens instead of silently truncating them.

Set `GEMINI_API_KEY`, `TYPESAFE_API_KEY` and `CODEX_EVAL_BINARY` in the environment. The last variable points to an authenticated Codex CLI supporting `gpt-6.1-sol`. Shared planning runs in an isolated directory with no tools; it sees only the question, title and native headings. Original-question Qwen/BGE runs are separate native baselines, not proposal-augmented versions.

Prepare QASPER source releases through the existing `scripts/frontier_data.py` and `scripts/iteration_splits.py` pipeline, retaining the published registry and data hashes. Do not regenerate or overwrite an existing research allocation. These scripts also prepare QuALITY files for historical studies; QuALITY is not scored in this paragraph comparison.

The private `output/research-budget.json` tracks cumulative request reservations and an explicit API cap. `RetrievalBudget` checks it before every paid request, including retries. Keep one paid inference process at a time: its threads share the ledger lock, but separate processes do not. Local GPU baselines may run alongside that process because they do not write the API ledger. Never commit provider keys or private operational ledgers.

## Frozen execution

The published registration was produced after the fixed development pilot and before validation access. Its private working copy is `output/retrieval-v3-confirmatory/registration.json`; on a clean reproduction, copy the public registration there after restoring the matching split allocation. On the original research allocation:

```sh
python scripts/retrieval_v3_gate.py open-validation
python scripts/retrieval_v3.py index --output output/retrieval-v3-confirmatory/validation
python scripts/retrieval_v3.py plans --output output/retrieval-v3-confirmatory/validation --workers 2
python scripts/retrieval_v3.py retrieve --output output/retrieval-v3-confirmatory/validation
python scripts/retrieval_v3.py qwen --output output/retrieval-v3-confirmatory/validation
python scripts/retrieval_v3_bge.py output/retrieval-v3-confirmatory/validation
python scripts/retrieval_v3_jev_controls.py run --output output/retrieval-v3-confirmatory/validation
python scripts/retrieval_v3_jev_controls.py open-test
```

The amended run also requires the published `tree-retrieval-v3-jev-rerank-amendment.json` as `output/retrieval-v3-confirmatory/jev-rerank-amendment.json` and the matching amended run manifest. The original registration and access markers remain preserved; the amendment records that no held-out outcomes had been inspected. Do not rerun the original `amend` command to invent the original timestamp.

The last command verifies all question/method identities, identical dense candidate pools and whole-paragraph packs, recomputes validation scores, freezes the selected JJJ/hybrid system and then releases test inputs. Run the same six inference stages with `test` in place of `validation`. Checkpointed outputs and provider caches allow interrupted stages to resume. A failed stage must be completed or reported incomplete; it does not permit dropping questions.

```sh
python scripts/analyze_retrieval_v3.py output/retrieval-v3-confirmatory/test
python scripts/retrieval_v3_statistics.py output/retrieval-v3-confirmatory/test
python -m unittest discover -s tests -q
```

The dedicated integrity gate must pass before results are accepted. Use `analyze_retrieval_v3.py` for the registered aligned paragraph metrics; the legacy `score` CLI retained in `retrieval_v3.py` is a raw-string diagnostic and is not the primary scorer.

For independent replication, create a separate output allocation and record a new registration that preserves the protocol. Do not impersonate the original one-time access markers. Public datasets may have appeared in model training; document-disjoint tuning/evaluation only controls this experiment's access.

## Interpretation

Scores apply to evidence recovery inside the supplied paper. The run tests compact Qwen and BGE checkpoint pipelines, not every frontier RAG system. Source-token output budgets exclude headings and wrapper syntax and are sums over unique complete paragraphs. Shared caching means recorded stage times are not independent deployed-system latency measurements; no speed superiority follows from them.
