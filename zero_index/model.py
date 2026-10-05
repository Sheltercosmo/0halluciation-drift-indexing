"""Replaceable evidence scores and an explicit Bayesian decision model."""

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Protocol


class Similarity(Protocol):
    """Higher scores mean stronger same-topic evidence, on [0, 1].

    Optional embedding adapters can implement this interface explicitly.
    Set score_kind='probability' when scores are already probabilities.
    """

    name: str

    def score(self, left: str, right: str) -> float: ...


@lru_cache(maxsize=4096)
def _tokens(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


class LexicalJaccard:
    """Untrained lexical baseline; neither JEV nor an embedding model."""

    name = "lexical-jaccard-baseline"
    score_kind = "similarity"

    def score(self, left: str, right: str) -> float:
        a, b = _tokens(left), _tokens(right)
        union = a | b
        return len(a & b) / len(union) if union else 0.0


def checked_score(scorer: Similarity, left: str, right: str) -> float:
    value = float(scorer.score(left, right))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Similarity scores must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class Config:
    same_topic_prior: float = 0.7
    posterior_cutoff: float = 0.5
    minimum_drop: float = 0.2
    same_alpha: float = 2.0
    same_beta: float = 1.0
    different_alpha: float = 1.0
    different_beta: float = 2.0
    probability_reference_prior: float = 0.5
    sentence_budget: int | None = None
    sentence_stop_threshold: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.same_topic_prior < 1:
            raise ValueError("same_topic_prior must be strictly between 0 and 1")
        if not 0 < self.probability_reference_prior < 1:
            raise ValueError("probability_reference_prior must be strictly between 0 and 1")
        for name in ("posterior_cutoff", "minimum_drop"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("same_alpha", "same_beta", "different_alpha", "different_beta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.sentence_budget is not None and (
            type(self.sentence_budget) is not int or self.sentence_budget < 2
        ):
            raise ValueError("sentence_budget must be an integer >= 2 or None")
        if self.sentence_stop_threshold is not None and (
            type(self.sentence_stop_threshold) not in (int, float)
            or not math.isfinite(self.sentence_stop_threshold)
            or not 0 <= self.sentence_stop_threshold <= 1
        ):
            raise ValueError("sentence_stop_threshold must be finite in [0, 1] or None")


def _log_beta_density(score: float, alpha: float, beta: float) -> float:
    return (
        (alpha - 1) * math.log(score)
        + (beta - 1) * math.log1p(-score)
        + math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    )


def posterior_same(score: float, config: Config) -> float:
    """Bayes' rule with two assumed Beta likelihoods; not calibrated accuracy."""
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("score must be finite and in [0, 1]")
    score = min(1 - 1e-9, max(1e-9, score))
    prior = config.same_topic_prior
    log_odds = (
        math.log(prior) - math.log1p(-prior)
        + _log_beta_density(score, config.same_alpha, config.same_beta)
        - _log_beta_density(score, config.different_alpha, config.different_beta)
    )
    if log_odds >= 0:
        return 1 / (1 + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def adjust_probability(probability: float, config: Config) -> float:
    """Prior-odds adjustment under an explicit reference-prior assumption.

    Exact only if the input is calibrated under that reference prior and
    class-conditional evidence stays fixed. Jev does not establish that here.
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if probability in (0, 1):
        return probability
    prior, reference = config.same_topic_prior, config.probability_reference_prior
    log_odds = (
        math.log(probability) - math.log1p(-probability)
        + math.log(prior) - math.log1p(-prior)
        - math.log(reference) + math.log1p(-reference)
    )
    if log_odds >= 0:
        return 1 / (1 + math.exp(-log_odds))
    odds = math.exp(log_odds)
    return odds / (1 + odds)
