"""Minimality, the degenerate guard, and regression detection.

These three are what separate the tool from a parameter sweep that prints
whatever flatters it:

  * minimality has to be provable, not approximate
  * a config returning most of the corpus is not a fix
  * a fix that breaks other queries has to say so
"""

from __future__ import annotations

from conftest import cfg
from whiff.config import RetrievalConfig, enumerate_space
from whiff.counterfactual import (DEGENERATE_K_FRACTION, find_minimal_fix,
                                  headroom, is_degenerate, retrieves,
                                  search_fix)
from whiff.pipeline import Corpus, IndexStore, Query
from whiff.triage import Verdict, triage


# --- the ordering that makes the search minimal ----------------------------

def test_space_is_ordered_by_distance_from_baseline() -> None:
    base = cfg(k=5, chunk_size=150)
    space = enumerate_space(base)
    distances = [c.distance_from(base) for c in space]
    assert distances == sorted(distances), (
        "enumerate_space must yield non-decreasing distance, or the first "
        "hit is not a minimum")


def test_baseline_is_distance_zero_and_comes_first() -> None:
    base = cfg(k=5, chunk_size=150, chunker="fixed", overlap=0)
    space = enumerate_space(base)
    assert space[0].distance_from(base) == (0, 0)


def test_reranking_changes_rank_before_reembedding_changes() -> None:
    """A k change costs less than a chunk_size change at equal field count."""
    base = cfg(k=5, chunk_size=150)
    assert base.with_(k=10).distance_from(base) == (1, 0)
    assert base.with_(chunk_size=300).distance_from(base) == (1, 1)


def test_requires_reembedding_only_for_chunk_or_model_changes() -> None:
    base = cfg()
    assert not base.with_(k=20).requires_reembedding(base)
    assert not base.with_(context_budget=10).requires_reembedding(base)
    assert base.with_(chunk_size=99).requires_reembedding(base)
    assert base.with_(chunker="sentence").requires_reembedding(base)
    assert base.with_(embedder="st:x").requires_reembedding(base)


def test_max_changed_fields_is_respected() -> None:
    base = cfg(k=5, chunk_size=150)
    for c in enumerate_space(base, max_changed_fields=1):
        assert len(c.changed_fields(base)) <= 1


def test_index_key_ignores_k() -> None:
    base = cfg(k=5)
    assert base.index_key() == base.with_(k=50).index_key()
    assert base.index_key() != base.with_(chunk_size=1).index_key()


# --- finding the fix -------------------------------------------------------

def test_minimal_fix_for_below_cutoff_is_a_k_change(store, zebra_query) -> None:
    base = cfg(k=1)
    assert triage(store.get(base), zebra_query).verdict is Verdict.BELOW_CUTOFF

    fix = find_minimal_fix(store, base, zebra_query,
                           space={"k": [1, 5, 10]}, max_changed_fields=1)
    assert fix is not None
    assert set(fix.changed) == {"k"}
    assert fix.distance[0] == 1
    assert not fix.requires_reembedding
    assert retrieves(store, fix.config, zebra_query)


def test_minimal_fix_stops_at_the_first_success(store, zebra_query) -> None:
    """Cheapest-first ordering means very few configs get evaluated."""
    fix = find_minimal_fix(store, cfg(k=1), zebra_query,
                           space={"k": [1, 5, 10]}, max_changed_fields=1)
    assert fix is not None
    assert fix.configs_tried <= 2


def test_no_fix_returns_none(store) -> None:
    """A document that is not in the corpus cannot be retrieved by tuning."""
    q = Query(id="imp", text="zebra", gold_doc="absent", gold_span="x")
    assert find_minimal_fix(store, cfg(), q, max_changed_fields=2) is None


def test_fix_describe_lists_the_change(store, zebra_query) -> None:
    fix = find_minimal_fix(store, cfg(k=1), zebra_query, space={"k": [1, 5]})
    assert fix is not None
    assert "k" in fix.describe() and "5" in fix.describe()


# --- the degenerate guard --------------------------------------------------

def test_k_above_the_fraction_is_degenerate(store) -> None:
    n_chunks = len(store.get(cfg()).chunks)
    assert n_chunks == 2            # two short docs, one chunk each
    # The floor of 5 keeps tiny corpora usable; beyond it the fraction bites.
    assert not is_degenerate(store, cfg(k=5))
    assert is_degenerate(store, cfg(k=6))


def test_degenerate_fraction_scales_with_corpus(cache) -> None:
    corpus = Corpus({f"d{i}": f"document number {i} about zebras"
                     for i in range(100)})
    store = IndexStore(corpus, cache)
    n = len(store.get(cfg()).chunks)
    assert n == 100
    limit = DEGENERATE_K_FRACTION * n
    assert not is_degenerate(store, cfg(k=int(limit)))
    assert is_degenerate(store, cfg(k=int(limit) + 1))


def test_degenerate_configs_are_not_offered_as_fixes(store, zebra_query) -> None:
    """On a 2-chunk corpus, k=20 'retrieves' everything. It is not a fix."""
    assert retrieves(store, cfg(k=20), zebra_query), (
        "premise: k=20 does technically retrieve it")
    fix = find_minimal_fix(store, cfg(k=1), zebra_query,
                           space={"k": [1, 20]}, max_changed_fields=1)
    assert fix is None, "a config returning the whole corpus is not a fix"


def test_degenerate_skips_are_counted_even_when_no_fix_is_found(
        store, zebra_query) -> None:
    """"No fix, and I refused 1 degenerate one" must be distinguishable from
    "no fix, nothing came close"."""
    res = search_fix(store, cfg(k=1), zebra_query,
                     space={"k": [1, 20]}, max_changed_fields=1)
    assert not res.found
    assert res.skipped_degenerate == 1
    assert res.configs_evaluated == 0


def test_cheapest_legitimate_fix_wins_over_a_degenerate_one(
        store, zebra_query) -> None:
    """k=5 sorts before k=20, so the guard never even has to refuse."""
    res = search_fix(store, cfg(k=1), zebra_query,
                     space={"k": [1, 20, 5]}, max_changed_fields=1)
    assert res.found
    assert res.fix.config.k == 5
    assert res.skipped_degenerate == 0


# --- regressions -----------------------------------------------------------

def test_regression_is_reported_when_a_fix_breaks_another_query(cache) -> None:
    """Smaller chunks raise term density (fixing a diluted document) while
    cutting through a long answer span (breaking another). The fix must own
    up to that."""
    long_span = ("the complete answer runs across quite a lot of characters "
                 "and will not survive a small chunk size")
    corpus = Corpus({
        "dense": "zebra zebra zebra zebra",
        "diluted": "padding text here. " * 12 + "zebra " * 8,
        "spanned": "quokka " * 2 + long_span,
    })
    store = IndexStore(corpus, cache)

    q_dilute = Query(id="a", text="zebra", gold_doc="diluted", gold_span=None)
    q_span = Query(id="b", text="quokka", gold_doc="spanned",
                   gold_span=long_span)
    base = cfg(chunk_size=400, k=1)

    # Premise: the spanned query works at the baseline, the diluted one does not.
    assert retrieves(store, base, q_span)
    assert not retrieves(store, base, q_dilute)

    fix = find_minimal_fix(store, base, q_dilute, queries=[q_dilute, q_span],
                           space={"chunk_size": [400, 30]},
                           max_changed_fields=1)
    assert fix is not None
    assert set(fix.changed) == {"chunk_size"}
    assert fix.regressions == ["b"], "shrinking chunks must be flagged as breaking b"
    assert not fix.safe
    assert fix.net < 0


def test_safe_fix_reports_no_regressions(store, zebra_query) -> None:
    other = Query(id="o", text="zebra", gold_doc="strong", gold_span=None)
    fix = find_minimal_fix(store, cfg(k=1), zebra_query,
                           queries=[zebra_query, other], space={"k": [1, 5]})
    assert fix is not None
    assert fix.safe
    assert fix.regressions == []


# --- headroom --------------------------------------------------------------

def test_headroom_reports_gap_and_unfixable(store, zebra_query) -> None:
    impossible = Query(id="imp", text="zebra", gold_doc="absent", gold_span="x")
    h = headroom(store, cfg(k=1), [zebra_query, impossible],
                 space={"k": [1, 5]}, max_changed_fields=1)
    assert h.n_queries == 2
    assert h.baseline_recall == 0.0
    assert h.oracle_recall == 50.0
    assert h.gap == 50.0
    assert h.unfixable == ["imp"]


def test_headroom_gap_is_zero_when_baseline_is_already_best(store, zebra_query) -> None:
    h = headroom(store, cfg(k=5), [zebra_query], space={"k": [5]},
                 max_changed_fields=1)
    assert h.baseline_recall == 100.0
    assert h.gap == 0.0
    assert "not retrieval tuning" in h.verdict
