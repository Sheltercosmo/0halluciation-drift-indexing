<div align="center">
  <img src="assets/github-social-preview.jpg" width="960" alt="0halluciation drift indexing builds topic blocks with decision models and statistical priors" />
  <h1>0halluciation drift indexing</h1>
</div>

Our index is a pure decision model based method with **0 LLM and optional embedding intervention**.

<p align="center">
  <img src="https://img.shields.io/badge/indexing-0%20generative%20LLM-345847?style=flat-square" alt="Indexing uses zero generative LLM calls" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-53645b?style=flat-square" alt="Python 3.10 and newer" />
  <img src="https://img.shields.io/badge/runtime%20dependencies-0-ad8650?style=flat-square" alt="Zero third-party runtime dependencies" />
  <img src="https://img.shields.io/badge/status-research%20alpha-8b7355?style=flat-square" alt="Research alpha" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="evals/REPORT.md">Measured results</a> ·
  <a href="docs/algorithm.md">Algorithm</a> ·
  <a href="docs/retrieval.md">Retrieval</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

**Reliable decisions with a purely statistical prior.** Jev scores whether paragraphs share a topic; an explicit Bayesian prior and a probability-drop rule determine where to cut. This makes document blocking a combination of learned decisions and transparent statistical rules, with no generative LLM calls during indexing and no required embedding model.

Headings and contents supply the upper structure without model calls. Jev then selects representative sentences directly from each topic block and paragraph using parallel outside-in search. The resulting content tree supports retrieval-augmented generation (RAG): an LLM proposes the content it needs, Jev reranks source evidence, and the LLM reads upward for context. The contribution is how topic blocks are formed; the tree is how those blocks are organized for retrieval.

“0 LLM” means no **generative LLM calls during indexing**. Jev is a learned decision model; the query-time LLM is separate. The project name expresses the aim of source-grounded retrieval, not a guarantee of error-free decisions or answers.

![Indexing and retrieval pipeline](assets/pipeline.svg)

## Measured, with limits

**Competitive performance has not yet been established.** The larger evaluation now has verified, complete releases totaling **4,779 questions** across QASPER, QuALITY, LongBench v2 and Bright-Pro, including Bright-Pro's 526,319-document corpus. These are prepared inputs, not new results. The [research-based evaluation design](evals/FRONTIER_EVALUATION.md) isolates decision-based blocking and the statistical prior, specifies stronger baselines, and reports evidence and answer quality under matched token budgets.

A frozen live pilot on **six synthetic documents and 24 questions** produced these results:

| Measurement | Baseline | Jev method |
| --- | ---: | ---: |
| Boundary F1 | 0.727, fixed two-paragraph chunks | **0.800** |
| Paragraph representative agreement | 12/36, first sentence | **31/36** |
| Retrieval hit@1 on the same Jev tree | 15/24, BM25-style ranking | **24/24** |
| Requests for six representative questions | 6, serial | **1, batched** |

Jev still made **five extra topic cuts**. The labels are assistant-authored, every document has only 18 sentence candidates, and the pilot does not evaluate an LLM’s final answers or compare against PageIndex. Read the [full report](evals/REPORT.md) for failures, baselines, uncertainty, usage, batching score differences, and reproducible decision replay.

A separate **public SciFact check** used existing rationale annotations: Jev achieved **11/12 hit@1**, versus **7/12** for BM25-style ranking, with five improvements and one regression. Each query was given its evidence abstract and four lexical distractors (35–86 sentences), so this evaluates reranking within a supplied pool, not open-corpus retrieval. [Read the public-data report →](evals/SCIFACT_REPORT.md)

## Quick start

Python 3.10+. The runtime has **no third-party dependencies**. Run from the repository root, or install with `python -m pip install -e .` to enable `zero-index`.

```sh
# Offline example: no key, model, or embedding required.
python -m zero_index build examples/structured.md -o output/tree.json
python -m zero_index outline output/tree.json
python -m zero_index find output/tree.json "visitor opening and closing times"
python -m examples.bottom_up
```

The default scorer and reranker are explicitly labeled lexical baselines. To use Jev, set `TYPESAFE_API_KEY` in your environment and run:

```sh
python -m zero_index build examples/structured.md --scorer jev --provider typesafe --max-calls 100 -o output/jev-tree.json
python -m zero_index find output/jev-tree.json "visitor opening and closing times" --reranker jev --provider typesafe --max-calls 20
```

These commands send source text to [TypeSafe’s decision API](https://docs.typesafe.ai/api) and incur usage. The native adapter pins `jev-1.13.0`. OpenRouter is also supported: use `--provider openrouter` with `OPENROUTER_API_KEY`; its default model is `typesafe/jev-1.13`. Keys are never stored in the tree, and provider errors stop the operation.

## Methodology: how we create the content tree

The index is built from the document's existing structure and exact source text. Topic blocking is driven by Jev decisions and an explicit statistical prior.

1. **Separate titles and contents.** Parse headings, heading levels and contents links without Jev or an LLM. Headings form the upper tree and act as hard boundaries. Recognized contents entries become navigation links to those headings.
2. **Create paragraph blocks.** Split content on paragraph boundaries within each heading. Preserve original text, source offsets and line references; keep fenced code intact.
3. **Compare against one anchor.** Start with `p0` and ask Jev whether `p0-p1`, `p0-p2`, `p0-p3`, and subsequent pairs belong to the same topic. Keep the anchor fixed until a boundary is found, avoiding all-pairs paragraph comparisons.
4. **Apply the statistical prior and cut at a drop.** Adjust each same-topic probability using the configured Bayesian prior. Cut before a paragraph when its adjusted probability is low **and** falls sharply from the preceding comparison. Start the next topic block at that paragraph and make it the new anchor. Save the scores, prior, probability drop and cut decision for inspection.
5. **Select central sentences in parallel waves.** For paragraph 1, paragraph 2, paragraph 3 and the other paragraphs in a topic block, evaluate each first and last sentence together. Then evaluate each second and second-last sentence, continuing toward the middle. Each candidate is judged against its paragraph and the whole topic block, so Jev can batch independent judgments for paragraph and section representatives. Representatives are copied source sentences.
6. **Assemble the content tree.** Attach topic blocks below their headings, paragraphs below topic blocks, and sentences below paragraphs. Store each paragraph and topic block's representative sentence alongside its full source content and provenance.

```text
Document
├── Contents → links to existing headings
└── Heading (nested headings retain their source hierarchy)
    └── Topic block + central source sentence
        └── Paragraph + central source sentence
            └── Source sentences with exact offsets
```

At retrieval time, an LLM proposes the content it needs. Lexical shortlisting finds candidate evidence; Jev reranks it against that request. The LLM can then read the selected sentence or paragraph and move upward through its topic block, heading and document for context. Embeddings are optional; the default retrieval path requires no vector database.

Full sentence search is the default. `--sentence-budget 2` restricts each target to two candidates; `--sentence-budget 0` searches all. Any finite search can miss a better candidate. The pilot’s budget-two agreement was 23/36 versus 31/36 with full search.

The default topic prior is `0.7`, cutoff `0.5`, drop `0.2`, and assumed reference prior `0.5`. These are explicit experimental choices, **not empirically calibrated probabilities**. [Read the equations and boundary policy →](docs/algorithm.md)

## Ask for content, then read upward

```python
from pathlib import Path
from zero_index import JevScorer, build_index, retrieve

jev = JevScorer(provider="typesafe", max_calls=100)
index = build_index(
    Path("document.md").read_text(encoding="utf-8"),
    source_name="document.md",
    scorer=jev,
)

# Your application supplies the query-time LLM callback.
# It returns one action: find, read, up, or finish.
def choose(state):
    return your_llm_tool_call(state)

result = retrieve(index, "What evidence answers my question?", choose, reranker=jev)
```

The callback proposes an evidence need with `find`. A scoped BM25-style pass supplies candidate sentences; Jev reranks them using parallel relevance decisions. The callback then uses `read` for a selected leaf, `up` for its paragraph and ancestors, and `finish` for evidence it actually read. Returned citations include source name, node ID, exact offsets, line numbers and source hash.

Candidate, call, step and reading budgets are explicit. Set `candidate_limit=None` in Python or `--candidates 0` in the CLI to rerank all leaves in scope. A finite lexical shortlist can lose paraphrases before Jev sees them. The application supplies the LLM and final answer generator. [See the callback contract →](docs/retrieval.md)

## Reproduce the pilot

```sh
python -m unittest discover -s tests -v

# Offline replay verifies the archived source hashes and reproduces metrics.
python -m evals.run --replay evals/results/pilot-v1-live --output output/my-replay

# A new bounded live evaluation; requires TYPESAFE_API_KEY.
python -m evals.run --output output/my-new-live-run
```

The archive includes the exact runtime snapshot, source corpus, gold labels, request bodies, returned probabilities, usage, timing and failed launch. It contains no credentials. Each new live run uses a fresh output directory and is capped at 220 HTTP attempts and 2,000 decision questions. [Protocol →](evals/PROTOCOL.md) · [Results →](evals/REPORT.md) · [Raw audit →](evals/results/pilot-v1-live/decisions.jsonl)

## Design notes

| Topic | Where to read |
| --- | --- |
| Bayesian cuts and outside-in waves | [Algorithm](docs/algorithm.md) |
| Headings, contents and research precedents | [Structural design](docs/structure-and-retrieval-design.md) |
| LLM proposals, Jev reranking and upward reads | [Retrieval](docs/retrieval.md) |
| Similarities and differences with PageIndex | [Comparison](docs/pageindex-comparison.md) |
| Mascot and generation provenance | [Meet Folio](assets/README.md) |
| GitHub description, topics and cover | [Repository metadata](docs/github-discovery.md) |
| Public-data retrieval check | [SciFact micro-pilot](evals/SCIFACT_REPORT.md) |
| Larger benchmarks, fair controls and complete dataset manifests | [Research-based evaluation](evals/FRONTIER_EVALUATION.md) |

**Is it the same as PageIndex?** It shares hierarchical, embedding-free LLM retrieval. The specific indexing recipe differs: fixed-anchor Jev comparisons, prior-adjusted cuts, and extractive representatives evaluated in parallel waves. PageIndex Flash already derives structure from PDF layout before model-assisted summaries/refinement, so structural parsing without an LLM alone is not a sufficient distinction. See the [primary description](https://pageindex.ai/blog/pageindex-flash) and the comparison above. No performance advantage over PageIndex has been measured here.

Embeddings remain optional through `EmbeddingScorer(embed, model_name="your-model")`; no embedding model or vector database is bundled. Markdown parsing and sentence splitting are deliberately small. PDF/DOCX layout recovery and OCR belong upstream. Fixed anchors can over-split subtopics or miss gradual drift; source extraction cannot guarantee correct retrieval or faithful generated answers.

<p align="center"><sub>Topic blocking with Jev decisions and Bayesian statistical priors, with embeddings optional.</sub></p>
