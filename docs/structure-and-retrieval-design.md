# Structural indexing and evidence-first retrieval

Design revision: 4 October 2026. These references inform specific choices; they do not validate this implementation's retrieval quality.

## Lessons from papers and repositories

| Reference | Relevant idea | Adoption here |
| --- | --- | --- |
| [Docling document model](https://github.com/docling-project/docling/blob/main/docs/concepts/docling_document.md) and [technical report](https://arxiv.org/abs/2408.09869) | Explicit hierarchy, reading order, and provenance | Structural metadata, source-aligned heading hints, and parent/child relationships |
| [Docling heading hierarchy code](https://github.com/docling-project/docling/blob/main/docling/models/stages/heading_hierarchy/heading_hierarchy_model.py) | Its PDF path prioritizes matched bookmarks, then numbering, then style | Preserve native hierarchy and import upstream heading evidence with provenance, rather than asking Jev to reconstruct it |
| [RAPTOR, ICLR 2024](https://arxiv.org/abs/2401.18059) | Multiple granularities in a tree; construction uses recursive embedding, clustering, and summaries | Separately addressable sentences, paragraphs, sections, and headings. We retain source spans and extractive representatives instead of its generative summary construction |
| [LlamaIndex AutoMergingRetriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/) | Retrieve leaves, then expand using parent relationships | Read detailed matches first and expose immediate-parent expansion. Our LLM requests expansion rather than using a child-count merge threshold |
| [Query2doc, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.585/) | LLM-generated query expansion can help express information needs | Let the LLM propose desired content while retaining the original question. The proposal is a query artifact, never cited evidence |

This adapts selected principles, not entire systems. No benchmark figures from those systems transfer to this project. Jev's existing parallel sentence-selection waves are preserved.

## Structural upper tree, inferred lower tree

```text
document title / root
└── heading hierarchy             declared structure, no Jev
    ├── contents entries           navigation links, no Jev
    └── topic sections             Jev + Bayesian boundary rule
        └── paragraphs             source blocks
            └── sentences          source spans
```

Native ATX/Setext heading levels take precedence. Heading text is retained; a leading heading can supply the document display title, while the filename remains the source identity. Imported headings carry origin, source reference, optional page, level, and exact header range.

Contents entries link to the existing hierarchy rather than duplicating body sections. Under a heading named `Contents`, `Table of Contents`, or `目录`, blocks consisting entirely of local Markdown links or dot-leader entries become TOC entries. They bypass Jev. Mixed prose and fenced code stay ordinary content.

Local anchors use a documented slug convention and unique suffixes. Printed TOC labels require a unique normalized title match. Ambiguous/missing matches remain unresolved with diagnostics. Printed page labels are retained without guessing character positions. No target heading is fabricated.

Arbitrary short lines, uppercase words, or numbering alone are not automatically promoted: they can be ordinary text or list items. Upstream document parsers can supply stronger evidence through `HeadingHint`.

## Source-aligned parser input

```python
from zero_index import HeadingHint, build_index

text = "Manual\n\nSetup\nRead this first."
index = build_index(text, heading_hints=[
    HeadingHint(0, 6, "Manual", 1, origin="docling", source_ref="#/texts/0", page=1),
    HeadingHint(8, 14, "Setup", 2, origin="pdf-outline", page=1),
])
```

Hints must address whole source lines, match the text, avoid fenced code, and not overlap. Native Markdown levels win when a hint addresses the same title; contradictory titles fail. CLI `--headings-json` accepts a JSON array of those fields.

This is a normalized integration boundary, **not a bundled Docling/PDF converter**. The caller must map reading-order text and headings into the same decoded string passed to `build_index`. PDF coordinates, bytes, or JavaScript UTF-16 offsets cannot replace Python Unicode character offsets. Upstream extraction quality and model use are separate from this indexer's structural stage.

## Retrieval responsibilities

1. The LLM expresses an evidence need and may choose an outline scope.
2. Deterministic candidate generation supplies source-backed leaf descriptors.
3. Jev reranks relevance to the question and need in parallel batches.
4. The LLM reads selected sentences and requests parents for more context.
5. The application returns actual read spans, dropping redundant child evidence when its parent is selected.

Detailed matches can be reached without requiring correct root-to-leaf choices at every level. The outline still helps formulate needs and choose scopes. Repeated searches support multiple needs and branches. See [retrieval actions and limits](retrieval.md).

## Validation and evaluation

Tests cover hierarchy/provenance, TOC ambiguity, exclusion of TOC text from Jev, score-to-candidate mapping, batched relevance, read-before-expand behavior, source citations, and budgets. They do not establish live-model speed or accuracy gains.

Evaluate structure with heading-parent accuracy, TOC-link precision/unresolved rate, and source-span coverage. Evaluate retrieval with pre-rerank candidate recall, post-rerank ranking quality, evidence recall after expansion, irrelevant context, input tokens, and latency. Compare scoped/global search, lexical/all-leaf candidate pools, Jev/lexical ranking, and leaf-only/bottom-up reading. Include paraphrases, misleading proposals, repeated titles, and evidence split across paragraphs.
