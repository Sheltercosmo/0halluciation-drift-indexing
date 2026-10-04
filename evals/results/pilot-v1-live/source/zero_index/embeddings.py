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
