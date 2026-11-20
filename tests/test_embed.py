"""Cache accounting and index correctness.

The cache's hit/miss numbers are quoted in the README as evidence that the
sweep is cheap, so they have to be right. A cache that silently reported
hits it did not have would make the project's central claim false.
"""

from __future__ import annotations

import numpy as np
import pytest

from whiff.embed import EmbeddingCache, HashEmbedder, get_embedder
from whiff.index import BruteForceIndex


@pytest.fixture
def emb() -> HashEmbedder:
    return HashEmbedder(n_features=256)


def test_vectors_are_l2_normalized(emb: HashEmbedder) -> None:
    vecs = emb.encode(["alpha beta", "gamma"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_input_returns_empty_matrix(emb: HashEmbedder) -> None:
    assert emb.encode([]).shape[0] == 0


def test_embedder_is_deterministic(emb: HashEmbedder) -> None:
    a = HashEmbedder(n_features=256).encode(["alpha beta"])
    b = HashEmbedder(n_features=256).encode(["alpha beta"])
    assert np.allclose(a, b)


@pytest.mark.parametrize("spec", ["gpt-9", "hash:", "hash:abc", "hash:4", "st:"])
def test_bad_embedder_specs_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        get_embedder(spec)


def test_hash_spec_sets_dimensionality() -> None:
    assert get_embedder("hash:512").dim == 512
    assert get_embedder("hash").dim == 2048


def test_hash_specs_are_different_vector_spaces() -> None:
    """What makes offline hybrid testing possible."""
    a, b = get_embedder("hash:512"), get_embedder("hash:1024")
    assert a.id != b.id
    assert a.encode(["alpha beta"]).shape[1] != b.encode(["alpha beta"]).shape[1]


def test_first_call_is_all_misses(cache: EmbeddingCache, emb) -> None:
    cache.encode(emb, ["one", "two", "three"])
    assert (cache.stats.hits, cache.stats.misses) == (0, 3)


def test_second_call_is_all_hits(cache: EmbeddingCache, emb) -> None:
    cache.encode(emb, ["one", "two"])
    cache.reset_stats()
    cache.encode(emb, ["one", "two"])
    assert (cache.stats.hits, cache.stats.misses) == (2, 0)
    assert cache.stats.hit_rate == 100.0


def test_duplicate_texts_in_one_call_are_embedded_once(cache, emb) -> None:
    """Dedup within a call - otherwise the same string costs twice."""
    cache.encode(emb, ["same", "same", "same"])
    assert cache.stats.misses == 1
    assert cache.stats.hits == 2


def test_row_order_matches_input_order(cache, emb) -> None:
    texts = ["alpha", "beta", "alpha", "gamma"]
    out = cache.encode(emb, texts)
    assert out.shape[0] == len(texts)
    assert np.allclose(out[0], out[2])       # same text, same vector
    assert not np.allclose(out[0], out[1])


def test_cache_result_equals_uncached_embedding(cache, emb) -> None:
    direct = emb.encode(["alpha beta gamma"])
    through_cache = cache.encode(emb, ["alpha beta gamma"])
    assert np.allclose(direct, through_cache, atol=1e-6)


def test_different_embedders_do_not_share_entries(cache) -> None:
    a, b = HashEmbedder(n_features=256), HashEmbedder(n_features=512)
    cache.encode(a, ["text"])
    cache.reset_stats()
    cache.encode(b, ["text"])
    assert cache.stats.misses == 1, "embedder id must be part of the cache key"


class PersistingHash(HashEmbedder):
    """Stands in for an expensive embedder (a sentence-transformer) so the
    persistence path is testable without a model download."""

    @property
    def id(self) -> str:
        return "persisting-" + super().id

    @property
    def persist(self) -> bool:
        return True


def test_cache_persists_across_instances(tmp_path) -> None:
    emb = PersistingHash(n_features=256)
    path = tmp_path / "vecs.sqlite"
    first = EmbeddingCache(path)
    first.encode(emb, ["persisted"])
    first.close()

    second = EmbeddingCache(path)
    second.encode(emb, ["persisted"])
    assert (second.stats.hits, second.stats.misses) == (1, 0)
    second.close()


def test_hash_embedder_does_not_touch_disk(tmp_path, emb) -> None:
    """The policy: cheap, high-dimensional vectors stay in memory.

    Persisting them wrote a 352 MB file on a SQuAD-sized run to save work
    that costs about a second to redo.
    """
    assert emb.persist is False
    path = tmp_path / "vecs.sqlite"
    cache = EmbeddingCache(path)
    cache.encode(emb, ["one", "two", "three"])
    cache.close()

    reopened = EmbeddingCache(path)
    reopened.encode(emb, ["one", "two", "three"])
    assert reopened.stats.misses == 3, "nothing should have been persisted"
    reopened.close()


def test_in_memory_reuse_still_works_for_non_persisted(cache, emb) -> None:
    """Skipping disk must not disable the layer that actually won the speedup."""
    cache.encode(emb, ["repeated"])
    cache.reset_stats()
    cache.encode(emb, ["repeated"])
    assert (cache.stats.hits, cache.stats.misses) == (1, 0)


def test_persisting_embedder_has_a_distinct_cache_namespace(tmp_path) -> None:
    """Same text, different embedder id, must not collide."""
    cheap = HashEmbedder(n_features=256)
    pricey = PersistingHash(n_features=256)
    cache = EmbeddingCache(tmp_path / "v.sqlite")
    cache.encode(cheap, ["shared text"])
    cache.reset_stats()
    cache.encode(pricey, ["shared text"])
    assert cache.stats.misses == 1
    cache.close()


def test_hit_rate_of_empty_cache_is_zero(cache) -> None:
    assert cache.stats.hit_rate == 0.0
    assert cache.stats.total == 0


# --- index -----------------------------------------------------------------

def _unit(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def test_index_returns_best_first() -> None:
    vecs = np.array([_unit([1, 0]), _unit([0.7, 0.7]), _unit([0, 1])],
                    dtype=np.float32)
    idx = BruteForceIndex(vecs, ["a", "b", "c"])
    hits = idx.search(np.array(_unit([1, 0]), dtype=np.float32), k=3)
    assert [i for i, _ in hits] == [0, 1, 2]
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_k_larger_than_corpus_is_clamped() -> None:
    vecs = np.array([_unit([1, 0])], dtype=np.float32)
    idx = BruteForceIndex(vecs, ["only"])
    assert len(idx.search(np.array(_unit([1, 0]), dtype=np.float32), k=50)) == 1


def test_index_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        BruteForceIndex(np.zeros((2, 3), dtype=np.float32), ["one"])


def test_empty_index_and_zero_k_return_nothing() -> None:
    empty = BruteForceIndex(np.zeros((0, 3), dtype=np.float32), [])
    assert empty.search(np.zeros(3, dtype=np.float32), k=5) == []

    one = BruteForceIndex(np.array([_unit([1, 0])], dtype=np.float32), ["a"])
    assert one.search(np.array(_unit([1, 0]), dtype=np.float32), k=0) == []
