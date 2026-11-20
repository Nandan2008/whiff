"""Vector index.

Brute force over normalized vectors: exact, trivially correct, and fast
enough for the corpus sizes a debugger runs on. Exactness matters here more
than speed - a debugger that blames your chunking when the real culprit was
its own approximate index would be worse than useless.

The `search` signature is deliberately the only thing the pipeline depends
on, so a hand-written ANN index can be dropped in behind it. That is the
natural extension to this project if you want to go deeper on algorithms.
"""

from __future__ import annotations

import numpy as np


class BruteForceIndex:
    def __init__(self, vectors: np.ndarray, ids: list[str]):
        if vectors.shape[0] != len(ids):
            raise ValueError("vectors and ids length mismatch")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ids = list(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Similarity against every row. Needed for hybrid blending, which
        has to combine full score vectors before taking a top-k."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        return self.vectors @ q

    @staticmethod
    def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k of an arbitrary score vector, best first."""
        if scores.size == 0 or k <= 0:
            return []
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k by cosine similarity, as (row_index, score), best first.

        Inputs are already L2-normalized, so the dot product *is* cosine.
        """
        if len(self) == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        scores = self.vectors @ q
        k = min(k, scores.shape[0])
        # argpartition for the top-k, then sort just those: O(n + k log k).
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]
