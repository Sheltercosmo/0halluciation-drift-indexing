"""Embedding-free candidate generation and query-specific evidence reranking."""

from collections import Counter
import math
import re
from typing import Protocol

from .index import DocumentIndex


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, need: str, candidates: list[dict]) -> list[float]: ...


class LexicalReranker:
    """Offline smoke-test baseline, not Jev relevance probabilities."""

    name = "lexical-relevance-baseline"

    def rerank(self, question: str, need: str, candidates: list[dict]) -> list[float]:
        largest = max((item["lexical_score"] for item in candidates), default=0)
        return [item["lexical_score"] / largest if largest else 0.0 for item in candidates]


def _terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def candidates(index: DocumentIndex, question: str, need: str, *, scope_id: str = "root",
               limit: int | None = 64) -> dict:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("candidate limit must be a positive integer or None")
    scope = index._node(scope_id)
    if scope.kind == "toc_entry":
        target = scope.metadata.get("target_id")
        if target is None:
            raise ValueError("An unresolved contents entry cannot be used as a search scope")
        scope = index._node(target)
    cards, counters = [], []
    for node in scope.walk():
        if node.kind != "sentence":
            continue
        path = index.path(node.node_id)
        headings = [part for part in path if part["kind"] == "heading"]
        parent = index._node(index._parents[node.node_id])
        text = index.source[node.start:node.end]
        summary = parent.central["text"] if parent.central else ""
        cards.append({"node_id": node.node_id, "kind": node.kind, "text": text,
                      "heading_path": headings, "paragraph_representative": summary,
                      "start": node.start, "end": node.end})
        counters.append(Counter(_terms(" ".join([text, summary, *(part["title"] for part in headings)]))))
    frequencies = Counter(term for counter in counters for term in counter)
    lengths = [sum(counter.values()) for counter in counters]
    average = sum(lengths) / len(lengths) if lengths else 1
    # Stable summation order gives repeatable scores across Python hash seeds.
    query = sorted(set(_terms(question + " " + need)))
    for card, counter, length in zip(cards, counters, lengths):
        score = 0.0
        for term in query:
            count = counter[term]
            if count:
                idf = math.log(1 + (len(cards) - frequencies[term] + .5) / (frequencies[term] + .5))
                score += idf * (count * 2.2) / (count + 1.2 * (.25 + .75 * length / (average or 1)))
        card["lexical_score"] = score
    cards.sort(key=lambda card: (-card["lexical_score"], card["start"]))
    selected = cards if limit is None else cards[:limit]
    return {"candidates": selected, "scope_id": scope.node_id, "pool_size": len(cards),
            "candidate_count": len(selected), "prefiltered": len(selected) < len(cards),
            "lexical_overlap": any(card["lexical_score"] > 0 for card in selected)}


def find(index: DocumentIndex, question: str, need: str, *, reranker: Reranker | None = None,
         scope_id: str = "root", candidate_limit: int | None = 64, top_k: int = 5,
         minimum_relevance: float = 0.0) -> dict:
    """The proposal is a search request. Only candidate source text is evidence."""
    if not isinstance(need, str) or not need.strip():
        raise ValueError("need must describe the requested content")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(minimum_relevance) or not 0 <= minimum_relevance <= 1:
        raise ValueError("minimum_relevance must be finite and in [0, 1]")
    pool = candidates(index, question, need, scope_id=scope_id, limit=candidate_limit)
    return rank_candidates(question, need, pool, reranker=reranker, top_k=top_k,
                           minimum_relevance=minimum_relevance)


def rank_candidates(question: str, need: str, pool: dict, *, reranker: Reranker | None,
                    top_k: int, minimum_relevance: float = 0.0) -> dict:
    cards = pool["candidates"]
    reranker = reranker if reranker is not None else LexicalReranker()
    scores = list(reranker.rerank(question, need, cards)) if cards else []
    if len(scores) != len(cards):
        raise ValueError("Reranker must return one score per candidate")
    matches = []
    for card, score in zip(cards, scores):
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Relevance scores must be finite numbers in [0, 1]")
        if score >= minimum_relevance:
            matches.append({**card, "relevance": score})
    matches.sort(key=lambda card: (-card["relevance"], -card["lexical_score"], card["start"]))
    return {**{key: value for key, value in pool.items() if key != "candidates"},
            "need": need, "reranker": reranker.name, "matches": matches[:top_k]}
