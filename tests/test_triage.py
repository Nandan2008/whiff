"""Verdict boundaries.

Every verdict is produced by holding the corpus fixed and moving exactly one
knob, so each test pins one boundary. The `strong`/`gold` fixture gives a
deterministic rank-1/rank-2 pair, which is all you need to walk a document
across the candidate pool, the cutoff, and the context budget in turn.
"""

from __future__ import annotations

from conftest import cfg
from whiff.pipeline import Corpus, IndexStore, Query, contains_span
from whiff.triage import Verdict, summarize, triage, triage_all


def test_fixture_ranks_as_expected(store: IndexStore, zebra_query) -> None:
    """Guard the premise the other tests rest on."""
    retr = store.get(cfg(k=5))
    ranked = retr.candidates(zebra_query.text)
    assert ranked[0].chunk.doc_id == "strong"
    assert ranked[1].chunk.doc_id == "gold"


def test_retrieved_when_k_is_generous(store, zebra_query) -> None:
    t = triage(store.get(cfg(k=5)), zebra_query)
    assert t.verdict is Verdict.RETRIEVED
    assert t.ok
    assert t.best_rank == 2


def test_below_cutoff_when_k_excludes_rank_two(store, zebra_query) -> None:
    t = triage(store.get(cfg(k=1)), zebra_query)
    assert t.verdict is Verdict.BELOW_CUTOFF
    assert t.best_rank == 2
    assert t.cutoff_k == 1
    assert "ranking" in t.verdict.blame


def test_absent_from_candidates_when_pool_excludes_it(store, zebra_query) -> None:
    """Below-cutoff and absent-from-candidates differ only by which stage
    dropped the document, and they blame different components."""
    t = triage(store.get(cfg(k=1, candidate_k=1)), zebra_query)
    assert t.verdict is Verdict.ABSENT_FROM_CANDIDATES
    assert t.best_rank is None
    assert "embedding" in t.verdict.blame


def test_dropped_in_packing_when_budget_cuts_it(store, zebra_query) -> None:
    """Retrieved within k, then discarded by context assembly."""
    retr = store.get(cfg(k=2))
    gold_chunk = retr.gold_chunks(zebra_query)[0]
    rank_one = retr.final(zebra_query.text)[0].chunk

    # A budget that fits the rank-1 chunk but not the gold chunk as well.
    budget = len(rank_one.text) + len(gold_chunk.text) - 1
    t = triage(store.get(cfg(k=2, context_budget=budget)), zebra_query)
    assert t.verdict is Verdict.DROPPED_IN_PACKING
    assert t.best_rank == 2
    assert "context assembly" in t.verdict.blame


def test_packing_keeps_gold_when_budget_is_ample(store, zebra_query) -> None:
    t = triage(store.get(cfg(k=2, context_budget=10_000)), zebra_query)
    assert t.verdict is Verdict.RETRIEVED


def test_not_indexed_when_document_is_missing(store) -> None:
    q = Query(id="x", text="zebra", gold_doc="nonexistent", gold_span="n/a")
    t = triage(store.get(cfg()), q)
    assert t.verdict is Verdict.NOT_INDEXED
    assert "not in the corpus" in " ".join(t.notes)


def test_span_fragmented_when_chunker_cuts_the_answer(cache) -> None:
    span = "the answer token is kangaroo and nothing else"
    corpus = Corpus({"gold": "zebra " * 3 + span})
    store = IndexStore(corpus, cache)
    q = Query(id="f", text="zebra", gold_doc="gold", gold_span=span)

    # Small chunks guarantee a boundary inside a 44-character span.
    t = triage(store.get(cfg(chunk_size=20)), q)
    assert t.verdict is Verdict.SPAN_FRAGMENTED
    assert "chunking" in t.verdict.blame

    # The same corpus and query succeed once a chunk can hold the span.
    assert triage(store.get(cfg(chunk_size=400)), q).ok


def test_wrong_span_text_is_blamed_on_the_span_not_the_retriever(cache) -> None:
    """A span that is not in the gold document at all is a data bug.

    Reporting this as a retrieval failure would send someone tuning a
    retriever that was never given a chance.
    """
    corpus = Corpus({"gold": "zebra stripes and nothing more"})
    store = IndexStore(corpus, cache)
    q = Query(id="w", text="zebra", gold_doc="gold",
              gold_span="a span that simply is not present")
    t = triage(store.get(cfg()), q)
    assert t.verdict is Verdict.NOT_INDEXED
    assert "check the span text" in " ".join(t.notes)


def test_gold_chunk_must_contain_the_span_not_just_the_document(cache) -> None:
    """A chunk from the right document that omits the answer is not a hit."""
    corpus = Corpus({"gold": "zebra " * 20 + "the buried answer token"})
    store = IndexStore(corpus, cache)
    q = Query(id="s", text="zebra", gold_doc="gold",
              gold_span="the buried answer token")
    retr = store.get(cfg(chunk_size=40))
    gold = retr.gold_chunks(q)
    in_doc = [c for c in retr.chunks if c.doc_id == "gold"]
    assert len(in_doc) > len(gold) >= 1


def test_no_span_means_any_chunk_of_the_document_counts(store) -> None:
    q = Query(id="n", text="zebra", gold_doc="gold", gold_span=None)
    retr = store.get(cfg())
    assert len(retr.gold_chunks(q)) == len(
        [c for c in retr.chunks if c.doc_id == "gold"])


def test_contains_span_ignores_whitespace_and_case() -> None:
    assert contains_span("The  Answer\nIs Here", "the answer is here")
    assert not contains_span("the answer", "")
    assert not contains_span("alpha beta", "gamma")


def test_summarize_counts_and_recall(store, zebra_query) -> None:
    missing = Query(id="m", text="zebra", gold_doc="gone", gold_span="x")
    results = triage_all(store.get(cfg(k=5)), [zebra_query, missing])
    s = summarize(results)
    assert s["n_queries"] == 2
    assert s["n_retrieved"] == 1
    assert s["recall_at_k"] == 50.0
    assert s["verdicts"]["not_indexed"] == 1


def test_index_store_reuses_indexes_across_k(store, zebra_query) -> None:
    """Configs differing only in k must not rebuild - this is the reuse the
    README claims."""
    store.get(cfg(k=1))
    builds_after_first = store.builds
    store.get(cfg(k=2))
    store.get(cfg(k=3))
    assert store.builds == builds_after_first

    store.get(cfg(chunk_size=50))   # changes the chunk set
    assert store.builds == builds_after_first + 1
