"""Fuse independent Jev tree and direct embedding retrieval paths."""

import math

from .tree_search import EmbeddingTreeRouter, _count


def _validate(source, passage):
    start, end = passage["start"], passage["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source):
        raise ValueError("Passage must have valid source offsets")
    if passage["text"] != source[start:end]:
        raise ValueError("Passage must match exact source text")
    return start, end


class EmbeddingPassageRetriever:
    """Search every supplied source chunk directly, independently of tree routes.

    Supply canonical recursive/semantic chunks from the whole document. Query
    and document embeddings may use their respective model task roles.
    """

    def __init__(self, embed_document, *, model_name, embed_query=None):
        self.name = f"direct-embedding:{model_name}"
        self.scorer = EmbeddingTreeRouter(embed_document, model_name=model_name, embed_query=embed_query)

    def retrieve(self, source, question, need, passages, *, limit=64):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        seen = set()
        for passage in passages:
            key = _validate(source, passage)
            if key in seen:
                raise ValueError("Direct retrieval chunks must have unique spans")
            seen.add(key)
        cards = [{"text": p["text"], "heading_path": [p.get("heading", "")]} for p in passages]
        scores = self.scorer.route(question, need, cards)
        order = sorted(range(len(passages)), key=lambda i: (-scores[i], passages[i]["start"], passages[i]["end"]))
        if limit is not None:
            order = order[:limit]
        return [{**passages[i], "similarity": scores[i]} for i in order]


def tree_passages(index, search):
    """Convert reached sentence leaves to ranked, exact paragraph candidates."""
    if search["source_sha256"] != index.metadata["source_sha256"]:
        raise ValueError("Search and index source differ")
    needs = [[] for _ in search["needs"]]
    for item in search["leaves"]:
        node = index._node(item["node_id"])
        if node.kind != "sentence" or (node.start, node.end) != (item["start"], item["end"]):
            raise ValueError("Tree result must reference source leaves")
        parent = index._node(index._parents[node.node_id])
        if parent.kind != "paragraph":
            raise ValueError("Tree leaf must have a paragraph parent")
        for need, score in item["need_scores"].items():
            need = int(need)
            if not 0 <= need < len(needs) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid need or route score")
            needs[need].append((score, parent))
    for queue in needs:
        queue.sort(key=lambda item: (-item[0], item[1].start))
    ranked, seen = [], set()
    for position in range(max(map(len, needs), default=0)):
        for queue in needs:
            if position >= len(queue):
                continue
            node = queue[position][1]
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            ranked.append({"start": node.start, "end": node.end, "text": index.source[node.start:node.end],
                           "node_id": node.node_id})
    return ranked


def fuse_retrieval_paths(source, tree_ranking, embedding_ranking, *, token_count,
                         title="document", budget=2048, tree_weight=.5, rank_constant=60):
    """Fuse complete retrieval outputs, then pack their source union once.

    Rank fusion avoids averaging Jev scores with cosine values. Exact duplicate
    spans receive support from both paths. Overlap is merged for reading, not
    counted twice against the token budget. Unread gaps are never filled.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not math.isfinite(tree_weight) or not 0 < tree_weight < 1:
        raise ValueError("tree_weight must be strictly between zero and one")
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("rank_constant must be a positive integer")
    candidates = {}
    for path, ranking, weight in (("jev_tree", tree_ranking, tree_weight),
                                  ("direct_embedding", embedding_ranking, 1-tree_weight)):
        seen = set()
        for rank, passage in enumerate(ranking, 1):
            span = _validate(source, passage)
            if span in seen:
                raise ValueError("Each retrieval path must deduplicate its spans")
            seen.add(span)
            item = candidates.setdefault(span, {"start": span[0], "end": span[1], "score": 0.0, "ranks": {}})
            item["score"] += weight / (rank_constant+rank)
            item["ranks"][path] = rank
    ranking = sorted(candidates.values(), key=lambda p: (-p["score"], p["start"], p["end"]))
    def union(spans):
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start,end])
        return merged
    def render(spans):
        return "Title: " + title + "\n" + "\n\n".join(
            f"[{i+1}]\n{source[a:b]}" for i,(a,b) in enumerate(spans)) if spans else ""
    spans, skipped, selected = [], [], []
    for item in ranking:
        proposed = union([*spans, [item["start"], item["end"]]])
        if _count(token_count, render(proposed)) <= budget:
            spans = proposed
            selected.append(item)
        else:
            skipped.append(item)
    context = render(spans)
    return {"context": context, "context_tokens": _count(token_count, context), "spans": spans,
            "ranking": ranking, "selected": selected, "skipped": skipped,
            "fusion": {"method": "retrieval-path-rrf", "tree_weight": tree_weight, "rank_constant": rank_constant}}
