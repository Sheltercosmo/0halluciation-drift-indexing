"""Validated, serializable controls for the standard evidence pipeline."""

from dataclasses import asdict, dataclass, field, fields, replace
import json
import math
from pathlib import Path

from .evidence_search import EvidenceSearchConfig
from .model import Config


def _number(name, value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")


@dataclass(frozen=True)
class IndexingConfig:
    """E splits use a distance quantile; J splits use Config's probability drop."""

    same_topic_prior: float = 0.7
    posterior_cutoff: float = 0.5
    minimum_drop: float = 0.2
    probability_reference_prior: float = 0.5
    sentence_budget: int | None = 8
    sentence_stop_threshold: float | None = None
    embedding_split_quantile: float = 0.85
    paragraph_embedding_prefix: str = "task: sentence similarity | query: "

    def __post_init__(self):
        for name in ("same_topic_prior", "posterior_cutoff", "minimum_drop",
                     "probability_reference_prior", "embedding_split_quantile"):
            _number(name, getattr(self, name), 0, 1)
        self.decision_config()
        if not isinstance(self.paragraph_embedding_prefix, str):
            raise ValueError("paragraph_embedding_prefix must be a string")

    def decision_config(self):
        names = {f.name for f in fields(Config)}
        return Config(**{f.name: getattr(self, f.name) for f in fields(self) if f.name in names})


@dataclass(frozen=True)
class SearchConfig:
    """Controls used by hierarchical Jev search (no global embedding knobs)."""

    beam: int = 5
    acceptance: float = 0.2
    child_cues: int = 6
    extra_sentences: int = 8
    refine_below: float = 0.85
    max_decisions: int = 4096

    def __post_init__(self):
        self.as_evidence_config()
        _number("acceptance", self.acceptance, 0, 1)
        _number("refine_below", self.refine_below, 0, 1)

    def as_evidence_config(self):
        return EvidenceSearchConfig(**asdict(self))


@dataclass(frozen=True)
class RetrievalConfig:
    """EEJ = embedding split, embedding central sentence, Jev tree search.

    Presets are effort controls, not measured accuracy tiers. Query-time Jev
    decisions are bounded separately from provider requests and indexing work.
    """

    variant: str = "EEJ"
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    deferred_search_fraction: float = 0.25
    pairwise_candidates: int = 30
    pairwise_threshold: float = 0.5
    shared_targets: int = 12
    output_paragraphs: int = 5
    dense_candidates: int | None = None

    def __post_init__(self):
        if self.variant not in ("EEJ", "EJJ", "JEJ", "JJJ"):
            raise ValueError("variant must be EEJ, EJJ, JEJ or JJJ; standard search uses Jev")
        if not isinstance(self.indexing, IndexingConfig) or not isinstance(self.search, SearchConfig):
            raise ValueError("Use IndexingConfig and SearchConfig for nested controls")
        _number("acceptance", self.search.acceptance, 0, 1)
        _number("refine_below", self.search.refine_below, 0, 1)
        _number("deferred_search_fraction", self.deferred_search_fraction, 0, 1)
        _number("pairwise_threshold", self.pairwise_threshold, 0.5, 1)
        for name in ("pairwise_candidates", "shared_targets", "output_paragraphs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dense_candidates is not None and (type(self.dense_candidates) is not int or self.dense_candidates < 1):
            raise ValueError("dense_candidates must be a positive integer or None")
        if not self.output_paragraphs <= self.shared_targets <= self.pairwise_candidates:
            raise ValueError("Require output_paragraphs <= shared_targets <= pairwise_candidates")

    @classmethod
    def for_effort(cls, effort="standard", **overrides):
        presets = {
            "low": dict(search=SearchConfig(beam=3, max_decisions=1024),
                        deferred_search_fraction=0.1, pairwise_candidates=16, shared_targets=8),
            "standard": {},
            "high": dict(search=SearchConfig(beam=8, max_decisions=8192),
                         deferred_search_fraction=0.5, pairwise_candidates=40, shared_targets=16),
        }
        if effort not in presets:
            raise ValueError("effort must be low, standard or high")
        return cls(**(presets[effort] | overrides))

    def with_search(self, **changes):
        """Change beam, acceptance, refinement, source cues or decision ceiling."""
        return replace(self, search=replace(self.search, **changes))

    def to_dict(self):
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Configuration must be an object")
        data = dict(value)
        version = data.pop("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported configuration schema_version")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
        try:
            for key, factory in (("indexing", IndexingConfig), ("search", SearchConfig)):
                if key in data:
                    if not isinstance(data[key], dict):
                        raise ValueError(f"{key} must be an object")
                    data[key] = factory(**data[key])
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid configuration: {error}") from error

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def decision_ceiling(self):
        """Query decision bound, excluding indexing, retries and LLM planning."""
        return (self.search.max_decisions + self.pairwise_candidates * (self.pairwise_candidates - 1)
                + 2 * self.shared_targets)


# The scored archive used JJ indexing. Keep its configuration explicit when
# replaying it; changing the deployment default cannot relabel historical scores.
MEASURED_JJJ_CONFIG = RetrievalConfig(variant="JJJ")
