"""Provider-neutral indexing adapter for the standard EEJ/JJJ pipeline."""

from dataclasses import asdict
import math

from scripts.bounded_eval import sections
from scripts.live_tree_eval import tree
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.configuration import RetrievalConfig
from zero_index.embeddings import CentroidRepresentatives
from zero_index.index import reselect_representatives
from zero_index.parse import Block
from zero_index.segment import segment_runs


def validate_document(doc):
    """Check paragraph identity and spans before any billable provider callback."""
    if not isinstance(doc.get("text"), str) or not isinstance(doc.get("title"), str) or not doc.get("id"):
        raise ValueError("Document requires id, title and source text")
    if not isinstance(doc.get("units"), list):
        raise ValueError("Document units must be a list")
    end = 0
    completed = set()
    active = None
    headings = {}
    for i, unit in enumerate(doc["units"]):
        if (type(unit.get("paragraph")) is not int or unit["paragraph"] != i
                or type(unit.get("start")) is not int or type(unit.get("end")) is not int
                or not end <= unit["start"] < unit["end"] <= len(doc["text"])):
            raise ValueError("Paragraph IDs must be sequential with ordered, nonoverlapping source spans")
        section = unit.get("section")
        if type(section) is not int or section < 0 or not isinstance(unit.get("heading"), str):
            raise ValueError("Each unit requires a nonnegative section number and heading")
        if active != section:
            if section in completed:
                raise ValueError("Section occurrences must have distinct IDs")
            completed.add(section)
            active = section
        if section in headings and headings[section] != unit["heading"]:
            raise ValueError("A section must have a consistent heading")
        headings[section] = unit["heading"]
        end = unit["end"]


def _vectors(embed, texts, purpose):
    if not texts:
        return []
    rows = list(embed(texts, purpose))
    if len(rows) != len(texts):
        raise ValueError("Embedding response must contain one vector per input")
    normalized = []
    for row in rows:
        vector = tuple(float(v) for v in row)
        norm = math.hypot(*vector)
        if not vector or not all(math.isfinite(v) for v in vector) or not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding vectors must be finite and nonzero")
        if normalized and len(vector) != len(normalized[0]):
            raise ValueError("Embedding dimensions must match")
        normalized.append(tuple(v / norm for v in vector))
    return normalized


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def build_standard_index(doc, *, config=None, embed=None, embedding_model=None, jev=None):
    """Build only the two configured indexing stages; EEJ makes no Jev calls.

    embed(texts, purpose) returns vectors in input order. Purpose is 'splitting'
    or 'central-sentences'. Jev implements score/score_many and representatives.
    Native headings are structural; whole paragraph IDs remain p0, p1, ... .
    """
    config = config or RetrievalConfig()
    validate_document(doc)
    if "E" in config.variant[:2] and (not callable(embed) or not isinstance(embedding_model, str) or not embedding_model.strip()):
        raise ValueError("E indexing requires an embed callback and explicit embedding_model")
    if "J" in config.variant[:2] and jev is None:
        raise ValueError("J indexing requires an explicit Jev scorer")
    native = sections(doc)
    groups, trace = [], []
    if config.variant[0] == "E":
        vectors = _vectors(embed, [config.indexing.paragraph_embedding_prefix + doc["text"][u["start"]:u["end"]]
                                  for u in doc["units"]], "splitting")
        for run in native:
            distances = [1 - min(1., max(-1., math.fsum(a * b for a, b in zip(
                vectors[left["paragraph"]], vectors[right["paragraph"]])))) for left, right in zip(run, run[1:])]
            threshold = _quantile(distances, config.indexing.embedding_split_quantile) if distances else 2.
            current = [run[0]]
            for distance, unit in zip(distances, run[1:]):
                cut = distance > threshold
                trace.append({"candidate_start": unit["start"], "distance": distance,
                              "threshold": threshold, "cut": cut, "method": "embedding_distance_quantile"})
                if cut:
                    groups.append(current)
                    current = []
                current.append(unit)
            groups.append(current)
    else:
        runs = [[Block("paragraph", u["start"], u["end"]) for u in run] for run in native]
        by_start = {u["start"]: u for u in doc["units"]}
        for cuts, decisions in segment_runs(doc["text"], runs, jev, config.indexing.decision_config()):
            groups.extend([[by_start[b.start] for b in group] for group in cuts])
            trace.extend(decisions)
    index = tree(doc, groups, config.variant[0])
    index.decisions = trace
    central = jev
    if config.variant[1] == "E":
        sentences = list(dict.fromkeys(n.title for n in index.root.walk() if n.kind == "sentence"))
        lookup = dict(zip(sentences, _vectors(embed, sentences, "central-sentences")))
        central = CentroidRepresentatives(lookup.__getitem__, model_name=embedding_model)
    index = reselect_representatives(index, central, sentence_budget=config.indexing.sentence_budget,
                                    stop_threshold=config.indexing.sentence_stop_threshold)
    index.metadata["standard_index"] = {"factors": config.variant[:2], "config": asdict(config.indexing),
                                        "embedding_model": embedding_model if "E" in config.variant[:2] else None}
    return restore_heading_hierarchy(index)


def check_index_factors(index, config):
    factors = index.metadata.get("standard_index", {}).get("factors")
    if factors is None:
        split = index.metadata.get("partition")
        central = index.metadata.get("representative_scorer", "")
        kind = "E" if central.startswith("embedding-centroid:") else "J" if central == "typesafe-jev" else "?"
        factors = str(split) + kind
    if factors != config.variant[:2]:
        raise ValueError(f"Index factors {factors!r} do not match {config.variant}; rebuild or choose the matching configuration")
    recorded = index.metadata.get("standard_index", {}).get("config")
    if recorded is not None and recorded != asdict(config.indexing):
        raise ValueError("Indexing controls changed; rebuild the index before retrieval")
