"""Optional caller-supplied embeddings; never enabled implicitly."""

import math
from typing import Callable, Iterable


class EmbeddingScorer:
    score_kind = "similarity"

    def __init__(self, embed: Callable[[str], Iterable[float]], *, model_name: str):
        self._embed = embed
        self.name = f"embedding-cosine:{model_name}"
        self._cache: dict[str, tuple[float, ...]] = {}

    def _vector(self, text: str) -> tuple[float, ...]:
        if text not in self._cache:
            vector = tuple(float(value) for value in self._embed(text))
            if not vector or not all(math.isfinite(value) for value in vector):
                raise ValueError("Embeddings must be nonempty finite vectors")
            norm = math.hypot(*vector)
            if not math.isfinite(norm):
                raise ValueError("Embedding norm overflow")
            self._cache[text] = tuple(value / norm for value in vector) if norm else vector
        return self._cache[text]

    def score(self, left: str, right: str) -> float:
        a, b = self._vector(left), self._vector(right)
        if len(a) != len(b):
            raise ValueError("Embedding dimensions must match")
        return min(1.0, max(0.0, sum(x * y for x, y in zip(a, b))))


class CentroidRepresentatives(EmbeddingScorer):
    """Exact mean cosine to the other sentences, computed through their sum.

    Each source sentence is embedded once; candidate scoring is O(n*d), rather
    than materializing all pairs. Scores are mapped from [-1, 1] to [0, 1] for
    the selection interface, and are not calibrated probabilities.
    """

    representative_method = "leave-one-out-cosine-centroid"

    def __init__(self, embed, *, model_name):
        super().__init__(embed, model_name=model_name)
        self.name = f"embedding-centroid:{model_name}"

    def score_representative_groups(self, groups, candidate_indices):
        if len(groups) != len(candidate_indices):
            raise ValueError("Groups and candidate indices must align")
        results = []
        for sentences, indices in zip(groups, candidate_indices):
            vectors = [self._vector(text) for text in sentences]
            if not vectors or any(not any(v) for v in vectors):
                raise ValueError("Centrality requires nonzero sentence embeddings")
            dimension = len(vectors[0])
            if any(len(v) != dimension for v in vectors):
                raise ValueError("Embedding dimensions must match")
            if any(type(i) is not int or not 0 <= i < len(vectors) for i in indices):
                raise ValueError("Invalid representative candidate index")
            total = [math.fsum(v[d] for v in vectors) for d in range(dimension)]
            scores = []
            for i in indices:
                if len(vectors) == 1:
                    scores.append(1.0)
                    continue
                dot = math.fsum(a * (b - a) for a, b in zip(vectors[i], total)) / (len(vectors) - 1)
                scores.append(min(1.0, max(0.0, (dot + 1) / 2)))
            results.append(scores)
        return results
