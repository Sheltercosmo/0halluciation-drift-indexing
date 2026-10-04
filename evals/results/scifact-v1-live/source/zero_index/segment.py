"""One-way anchor comparisons and outside-in representative selection."""

import math

from .model import Config, Similarity, adjust_probability, checked_score, posterior_same
from .parse import Block


def segment(
    source: str, blocks: list[Block], scorer: Similarity, config: Config
) -> tuple[list[list[Block]], list[dict]]:
    if not blocks:
        return [], []
    groups = [[blocks[0]]]
    anchor = blocks[0]
    previous = config.same_topic_prior
    trace: list[dict] = []
    for block in blocks[1:]:
        score = checked_score(scorer, source[anchor.start:anchor.end], source[block.start:block.end])
        is_probability = getattr(scorer, "score_kind", "similarity") == "probability"
        probability = adjust_probability(score, config) if is_probability else posterior_same(score, config)
        drop = previous - probability
        cut = probability <= config.posterior_cutoff and drop >= config.minimum_drop
        trace.append({
            "anchor_start": anchor.start, "candidate_start": block.start,
            "score": score, "prior": config.same_topic_prior,
            "evidence_mode": "prior-odds-adjustment" if is_probability else "beta-likelihood",
            "posterior_same": probability, "previous_probability": previous,
            "drop": drop, "cut": cut,
        })
        if cut:
            groups.append([block])
            anchor = block
            previous = config.same_topic_prior
        else:
            groups[-1].append(block)
            previous = probability
    return groups, trace


def outside_in(size: int):
    left, right = 0, size - 1
    while left <= right:
        yield left
        if left != right:
            yield right
        left += 1
        right -= 1


def central_sentence(
    source: str, spans: list[tuple[int, int]], scorer: Similarity, budget: int | None
) -> dict | None:
    return central_sentences(source, [spans], scorer, budget)[0]


def central_sentences(
    source: str, groups: list[list[tuple[int, int]]], scorer: Similarity,
    budget: int | None = None, *, include_section: bool = False,
) -> list[dict | None]:
    """Visit every paragraph's outer pair together, then move inward in waves.

    If requested, score each candidate against both its section and paragraph
    context in the same wave. These judgments have no dependency on each other.
    A budget limits candidates per target node; None covers every sentence.
    """
    if budget is not None and (type(budget) is not int or budget < 2):
        raise ValueError("budget must be an integer >= 2 or None")
    targets = ([[span for spans in groups for span in spans]] if include_section else []) + groups
    contexts = ["\n".join(source[start:end] for start, end in spans) for spans in targets]
    limits = [len(spans) if budget is None else min(budget, len(spans)) for spans in targets]
    visited: list[list[int]] = [[] for _ in targets]
    best: list[tuple[int, float] | None] = [None for _ in targets]
    comparisons = [0 for _ in targets]
    batch = getattr(scorer, "representatives", None)
    single = getattr(scorer, "representative", None)
    semantic = callable(batch) or callable(single)

    def record(target: int, candidate: int, value: float) -> None:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Representative scores must be finite and in [0, 1]")
        if best[target] is None or value > best[target][1]:
            best[target] = (candidate, value)

    depth = 0
    while True:
        jobs: list[tuple[int, int]] = []
        offset = 0
        for paragraph, spans in enumerate(groups):
            left, right = depth, len(spans) - 1 - depth
            if left <= right:
                for candidate in ([left] if left == right else [left, right]):
                    locations = [(paragraph + int(include_section), candidate)]
                    if include_section:
                        locations.insert(0, (0, offset + candidate))
                    for target, position in locations:
                        if len(visited[target]) < limits[target]:
                            visited[target].append(position)
                            if len(targets[target]) == 1:
                                record(target, position, 1.0)
                            else:
                                jobs.append((target, position))
            offset += len(spans)
        if jobs:
            if semantic:
                pairs = [(source[targets[t][c][0]:targets[t][c][1]], contexts[t]) for t, c in jobs]
                values = list(batch(pairs)) if callable(batch) else [single(*pair) for pair in pairs]
                if len(values) != len(jobs):
                    raise ValueError("Representative batch returned the wrong number of scores")
                for (target, candidate), value in zip(jobs, values):
                    comparisons[target] += 1
                    record(target, candidate, float(value))
            else:
                for target, candidate in jobs:
                    spans = targets[target]
                    start, end = spans[candidate]
                    total = sum(checked_score(scorer, source[start:end], source[a:b])
                                for other, (a, b) in enumerate(spans) if other != candidate)
                    comparisons[target] += len(spans) - 1
                    record(target, candidate, total / (len(spans) - 1))
        if all(len(indices) == limit for indices, limit in zip(visited, limits)):
            break
        depth += 1

    results = []
    for target, winner in enumerate(best):
        if winner is None:
            results.append(None)
            continue
        start, end = targets[target][winner[0]]
        results.append({
            "text": source[start:end], "start": start, "end": end,
            "centrality": winner[1], "candidate_indices": visited[target],
            "candidate_count": len(targets[target]), "comparisons": comparisons[target],
            "exhaustive": len(visited[target]) == len(targets[target]),
            "method": "context-representativeness" if semantic else "mean-pairwise-similarity",
        })
    return results
