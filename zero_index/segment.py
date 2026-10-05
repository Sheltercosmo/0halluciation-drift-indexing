"""One-way anchor comparisons and outside-in representative selection."""

import math
from dataclasses import dataclass

from .model import Config, Similarity, adjust_probability, checked_score, posterior_same
from .parse import Block


def segment(
    source: str, blocks: list[Block], scorer: Similarity, config: Config
) -> tuple[list[list[Block]], list[dict]]:
    return segment_runs(source, [blocks], scorer, config)[0]


def segment_runs(source: str, runs: list[list[Block]], scorer: Similarity, config: Config):
    """Advance independent heading runs together, with no speculative pairs."""
    runners = [_segment_steps(source, blocks, scorer, config) for blocks in runs]
    results = [None] * len(runs)
    ready = {}

    def advance(number, value=None):
        try:
            ready[number] = runners[number].send(value)
        except StopIteration as done:
            results[number] = done.value
            ready.pop(number, None)

    batch = getattr(scorer, "score_many", None)
    if not callable(batch):
        # Stateful third-party scalar scorers retain source-order invocation.
        for number in range(len(runs)):
            advance(number)
            while number in ready:
                advance(number, checked_score(scorer, *ready[number]))
        return results

    for number in range(len(runs)):
        advance(number)
    while ready:
        numbers = list(ready)
        values = list(batch([ready[number] for number in numbers]))
        if len(values) != len(numbers):
            raise ValueError("Topic batch returned the wrong number of scores")
        for number, value in zip(numbers, values):
            advance(number, float(value))
    return results


def _segment_steps(source, blocks, scorer, config):
    if not blocks:
        return [], []
    groups = [[blocks[0]]]
    anchor = blocks[0]
    previous = config.same_topic_prior
    trace: list[dict] = []
    for block in blocks[1:]:
        score = yield source[anchor.start:anchor.end], source[block.start:block.end]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Similarity scores must be finite and in [0, 1]")
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
    return central_sentences_many(
        source, [groups], scorer, budget, include_section=include_section,
    )[0]


@dataclass
class _CentralPlan:
    targets: list[list[tuple[int, int]]]
    contexts: list[str]
    visited: list[list[int]]
    jobs: list[tuple[int, int]]
    best: list[tuple[int, float] | None]


def central_sentences_many(source, sections, scorer, budget=None, *, include_section=True):
    """Plan outside-in candidates, then score all independent sections together.

    Waves define candidate order, not response dependencies. Flattening them
    lets the scorer fill batches across depths and sections. Reduce results in
    planned order, so completion timing cannot change ties or candidate limits.
    """
    plans = [_central_plan(source, groups, budget, include_section) for groups in sections]
    batch = getattr(scorer, "representatives", None)
    single = getattr(scorer, "representative", None)
    semantic = callable(batch) or callable(single)
    all_jobs = [(plan, target, candidate) for plan in plans for target, candidate in plan.jobs]
    if semantic:
        pairs = [(source[plan.targets[t][c][0]:plan.targets[t][c][1]], plan.contexts[t])
                 for plan, t, c in all_jobs]
        values = list(batch(pairs)) if callable(batch) and pairs else [single(*pair) for pair in pairs]
        if len(values) != len(all_jobs):
            raise ValueError("Representative batch returned the wrong number of scores")
    else:
        values = []
        for plan, target, candidate in all_jobs:
            spans = plan.targets[target]
            start, end = spans[candidate]
            total = sum(checked_score(scorer, source[start:end], source[a:b])
                        for other, (a, b) in enumerate(spans) if other != candidate)
            values.append(total / (len(spans) - 1))
    for (plan, target, candidate), value in zip(all_jobs, values):
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Representative scores must be finite and in [0, 1]")
        best = plan.best
        if best[target] is None or value > best[target][1]:
            best[target] = (candidate, value)

    results = []
    for plan in plans:
        section_results = []
        for target, winner in enumerate(plan.best):
            if winner is None:
                section_results.append(None)
                continue
            start, end = plan.targets[target][winner[0]]
            count = len(plan.targets[target])
            comparisons = len(plan.visited[target]) * (1 if semantic else count - 1) if count > 1 else 0
            section_results.append({
                "text": source[start:end], "start": start, "end": end,
                "centrality": winner[1], "candidate_indices": plan.visited[target],
                "candidate_count": count,
                "comparisons": comparisons,
                "exhaustive": len(plan.visited[target]) == count,
                "method": "context-representativeness" if semantic else "mean-pairwise-similarity",
            })
        results.append(section_results)
    return results


def _central_plan(source, groups, budget, include_section):
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
    jobs: list[tuple[int, int]] = []
    depth = 0
    while True:
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
                                best[target] = (position, 1.0)
                            else:
                                jobs.append((target, position))
            offset += len(spans)
        if all(len(indices) == limit for indices, limit in zip(visited, limits)):
            break
        depth += 1

    return _CentralPlan(targets, contexts, visited, jobs, best)
