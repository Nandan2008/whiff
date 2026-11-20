"""Pytest configuration and shared fixtures.

Living at the project root so that pytest puts the root on sys.path and
`import whiff` resolves without an install step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whiff.config import RetrievalConfig  # noqa: E402
from whiff.embed import EmbeddingCache  # noqa: E402
from whiff.pipeline import Corpus, IndexStore, Query  # noqa: E402


@pytest.fixture
def cache() -> EmbeddingCache:
    """In-memory cache: no file, no cross-test contamination."""
    return EmbeddingCache(None)


@pytest.fixture
def ranked_corpus() -> Corpus:
    """Two documents where `strong` reliably outranks `gold` for "zebra".

    `strong` is short and made almost entirely of the query term, so after
    L2 normalization its vector points nearly straight at the query. `gold`
    dilutes the term with other content. That gives a deterministic
    rank-1/rank-2 pair to test every cutoff boundary against.
    """
    return Corpus({
        "strong": "zebra zebra zebra zebra",
        "gold": "zebra stripes. The answer token is kangaroo.",
    })


@pytest.fixture
def zebra_query() -> Query:
    return Query(id="z1", text="zebra", gold_doc="gold",
                 gold_span="answer token is kangaroo")


@pytest.fixture
def store(ranked_corpus: Corpus, cache: EmbeddingCache) -> IndexStore:
    return IndexStore(ranked_corpus, cache)


def cfg(**kw) -> RetrievalConfig:
    """A config with test-friendly defaults: big chunks, tiny pools."""
    base = dict(chunker="fixed", chunk_size=400, overlap=0, embedder="hash",
                k=5, candidate_k=20, context_budget=None)
    base.update(kw)
    return RetrievalConfig(**base)
