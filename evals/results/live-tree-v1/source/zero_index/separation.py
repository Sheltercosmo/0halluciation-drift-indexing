"""Experimental parent-normalized split evidence; lower scores favor a cut."""

import math


def _probability(name, value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")


def normalized_separation(cross_probability, within_probability, parent_cross_probability, *,
                          threshold, minimum_parent_probability=.05):
    """Compare a leaf/group contrast with a parent reference.

    The leaf signal is cross_probability - within_probability. Its raw ratio
    to parent_cross_probability is retained; the decision score is clipped to
    [0, 1]. A signal equal to the parent reference scores 1. Scores at or below
    threshold favor separation. The returned decision does not mutate a tree.

    A weak/zero parent reference yields no decision, rather than manufacturing
    a favorable score through epsilon division. These supplied probabilities
    must describe registered, comparable group judgments, not arbitrary query
    relevance or confidence values. This ratio is a heuristic, not Bayes' rule.
    """
    for name,value in (("cross_probability",cross_probability),("within_probability",within_probability),
                       ("parent_cross_probability",parent_cross_probability),("threshold",threshold),
                       ("minimum_parent_probability",minimum_parent_probability)):
        _probability(name,value)
    if threshold == 1:
        raise ValueError("threshold must be below 1; equality means no separation gain")
    if minimum_parent_probability == 0:
        raise ValueError("minimum_parent_probability must be positive")
    signal = cross_probability-within_probability
    result = {"cross_probability":cross_probability,"within_probability":within_probability,
              "parent_cross_probability":parent_cross_probability,"leaf_signal":signal,
              "threshold":threshold,"minimum_parent_probability":minimum_parent_probability,
              "raw_ratio":None,"score":None,"accept_split":None,"clipped":False}
    if parent_cross_probability < minimum_parent_probability:
        return {**result,"status":"insufficient_parent_reference"}
    raw = signal / parent_cross_probability
    score = min(1.0,max(0.0,raw))
    return {**result,"raw_ratio":raw,"score":score,"clipped":raw != score,
            "accept_split":score <= threshold,"status":"scored"}
