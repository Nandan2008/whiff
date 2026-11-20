"""Hybrid lexical+dense retrieval.

Hybrid exists to fix vocabulary mismatch: a lexical retriever misses a
paraphrase, a dense one can miss an exact rare token. These tests use a
second *hash* embedder with different settings as the stand-in "other"
retriever, so the whole file runs offline with no model download - the
blending maths is what is under test, not any particular model.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import cfg
from whiff.index import BruteForceIndex
from whiff.pipeline import Corpus, IndexStore, Query, _minmax
from whiff.triage import triage

# A genuinely different vector space, still offline: fewer hash features
# means different collisions, so the two retrievers disagree about ranking.
SECOND = "hash:512"


def test_minmax_scales_to_unit_range() -> None:
    out = _minmax(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_minmax_of_flat_scores_is_all_zeros() -> None:
    """A retriever with no opinion must not dominate the blend."""
    out = _minmax(np.array([3.0, 3.0, 3.0], dtype=np.float32))
    assert np.allclose(out, 0.0)


def test_minmax_of_empty_is_empty() -> None:
    assert _minmax(np.zeros(0, dtype=np.float32)).size == 0


def test_index_scores_returns_one_per_row() -> None:
    vecs = np.eye(3, dtype=np.float32)
    idx = BruteForceIndex(vecs, ["a", "b", "c"])
    s = idx.scores(np.array([1, 0, 0], dtype=np.float32))
    assert s.shape == (3,)
    assert s[0] == pytest.approx(1.0)


def test_top_k_matches_search(ranked_corpus: Corpus, cache) -> None:
    """top_k on a full score vector must agree with the optimized search."""
    store = IndexStore(ranked_corpus, cache)
    retr = store.get(cfg())
    qv = retr._query_vec("zebra")
    via_search = retr.index.search(qv, 2)
    via_scores = BruteForceIndex.top_k(retr.index.scores(qv), 2)
    assert [i for i, _ in via_search] == [i for i, _ in via_scores]


def test_hybrid_builds_a_secondary_index(ranked_corpus: Corpus, cache) -> None:
    store = IndexStore(ranked_corpus, cache)
    assert store.get(cfg()).secondary is None
    assert store.get(cfg(hybrid_with=SECOND)).secondary is not None


def test_hybrid_shares_the_chunk_set(ranked_corpus: Corpus, cache) -> None:
    """Only the vectors are extra work; chunks are not re-chunked."""
    store = IndexStore(ranked_corpus, cache)
    plain = store.get(cfg())
    hybrid = store.get(cfg(hybrid_with=SECOND))
    assert [c.id for c in plain.chunks] == [c.id for c in hybrid.chunks]


def test_hybrid_changes_the_index_key(ranked_corpus: Corpus, cache) -> None:
    """Otherwise the store would hand back a non-hybrid retriever."""
    assert cfg().index_key() != cfg(hybrid_with=SECOND).index_key()


def test_index_key_stays_sortable_with_and_without_hybrid() -> None:
    """None coerced to "" - a space mixing both must not raise on sort."""
    keys = [cfg(hybrid_with=None).index_key(),
            cfg(hybrid_with=SECOND).index_key()]
    sorted(keys)   # must not raise TypeError
    assert all(isinstance(k[-1], str) for k in keys)


def test_hybrid_requires_reembedding() -> None:
    base = cfg()
    assert base.with_(hybrid_with=SECOND).requires_reembedding(base)


def test_hybrid_weight_zero_matches_primary_ranking(ranked_corpus, cache) -> None:
    """w=0 must reduce exactly to the primary retriever."""
    store = IndexStore(ranked_corpus, cache)
    plain = store.get(cfg())
    hybrid = store.get(cfg(hybrid_with=SECOND, hybrid_weight=0.0))
    assert ([r.chunk.id for r in plain.candidates("zebra")]
            == [r.chunk.id for r in hybrid.candidates("zebra")])


def test_hybrid_weight_one_matches_secondary_ranking(ranked_corpus, cache) -> None:
    """w=1 must reduce exactly to the secondary retriever."""
    store = IndexStore(ranked_corpus, cache)
    only_second = store.get(cfg(embedder=SECOND))
    hybrid = store.get(cfg(hybrid_with=SECOND, hybrid_weight=1.0))
    assert ([r.chunk.id for r in only_second.candidates("zebra")]
            == [r.chunk.id for r in hybrid.candidates("zebra")])


def test_hybrid_scores_are_in_unit_range(ranked_corpus, cache) -> None:
    store = IndexStore(ranked_corpus, cache)
    retr = store.get(cfg(hybrid_with=SECOND, hybrid_weight=0.5))
    for r in retr.candidates("zebra"):
        assert -1e-6 <= r.score <= 1.0 + 1e-6


def test_hybrid_still_triages(ranked_corpus, cache, zebra_query: Query) -> None:
    """End to end: the verdict machinery works through the blended path."""
    store = IndexStore(ranked_corpus, cache)
    t = triage(store.get(cfg(hybrid_with=SECOND, k=5)), zebra_query)
    assert t.ok
    t2 = triage(store.get(cfg(hybrid_with=SECOND, k=1)), zebra_query)
    assert not t2.ok


def test_describe_mentions_hybrid() -> None:
    assert SECOND in cfg(hybrid_with=SECOND).describe()
    assert "+" in cfg(hybrid_with=SECOND).describe()
