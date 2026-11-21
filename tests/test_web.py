"""The web layer's payload builders.

These are plain functions with no HTTP in them, which is the point: the API
shape is testable without binding a port, starting a thread, or waiting on a
browser.
"""

from __future__ import annotations

import json

import pytest

from conftest import cfg
from whiff.pipeline import Corpus, IndexStore, Query
from whiff.web import (STATIC, config_dict, config_from_params,
                       explain_payload, overview_payload)


@pytest.fixture
def ctx(ranked_corpus: Corpus, cache):
    store = IndexStore(ranked_corpus, cache)
    queries = [
        Query(id="z1", text="zebra", gold_doc="gold",
              gold_span="answer token is kangaroo"),
        Query(id="miss", text="zebra", gold_doc="absent", gold_span="nope"),
    ]
    return store, queries


# --- config parsing --------------------------------------------------------

def test_params_round_trip() -> None:
    c = config_from_params({
        "chunker": ["recursive"], "chunk_size": ["420"], "overlap": ["12"],
        "k": ["7"], "candidate_k": ["33"], "embedder": ["hash:512"],
    })
    assert (c.chunker, c.chunk_size, c.overlap) == ("recursive", 420, 12)
    assert (c.k, c.candidate_k, c.embedder) == (7, 33, "hash:512")


def test_missing_params_fall_back_to_defaults() -> None:
    c = config_from_params({})
    assert (c.chunker, c.chunk_size, c.k) == ("fixed", 150, 5)


def test_garbage_params_do_not_raise() -> None:
    """A half-typed number in the browser must not 500 the server."""
    c = config_from_params({"chunk_size": ["abc"], "k": [""], "overlap": ["-"]})
    assert (c.chunk_size, c.k, c.overlap) == (150, 5, 0)


def test_blank_hybrid_means_none() -> None:
    assert config_from_params({"hybrid_with": [""]}).hybrid_with is None
    assert config_from_params(
        {"hybrid_with": ["hash:512"]}).hybrid_with == "hash:512"


def test_zero_context_budget_means_no_packing_stage() -> None:
    assert config_from_params({"context_budget": ["0"]}).context_budget is None
    assert config_from_params({"context_budget": ["800"]}).context_budget == 800


def test_config_dict_is_json_serializable() -> None:
    json.dumps(config_dict(cfg()))


# --- overview --------------------------------------------------------------

def test_overview_lists_every_query(ctx) -> None:
    store, queries = ctx
    data = overview_payload(store, cfg(k=5), queries)
    assert [q["id"] for q in data["queries"]] == ["z1", "miss"]
    assert data["corpus"]["docs"] == 2
    assert data["summary"]["n_queries"] == 2


def test_overview_marks_pass_and_fail(ctx) -> None:
    store, queries = ctx
    rows = {q["id"]: q for q in
            overview_payload(store, cfg(k=5), queries)["queries"]}
    assert rows["z1"]["ok"] is True
    assert rows["miss"]["ok"] is False
    assert rows["miss"]["verdict"] == "not_indexed"


def test_overview_is_json_serializable(ctx) -> None:
    store, queries = ctx
    json.dumps(overview_payload(store, cfg(), queries))


# --- explain ---------------------------------------------------------------

def test_explain_unknown_id_returns_error_not_exception(ctx) -> None:
    store, queries = ctx
    assert "error" in explain_payload(store, cfg(), queries, "nope")


def test_explain_of_a_passing_query_has_no_fix(ctx) -> None:
    store, queries = ctx
    d = explain_payload(store, cfg(k=5), queries, "z1")
    assert d["triage"]["ok"] is True
    assert d["fix"] is None


def test_explain_of_a_failing_query_carries_a_fix(ctx) -> None:
    store, queries = ctx
    d = explain_payload(store, cfg(k=1), queries, "z1")
    assert d["triage"]["verdict"] == "below_cutoff"
    assert d["triage"]["best_rank"] == 2
    assert d["fix"]["found"] is True
    assert d["fix"]["changes"][0]["field"] == "k"


def test_explain_marks_the_gold_row(ctx) -> None:
    store, queries = ctx
    d = explain_payload(store, cfg(k=5), queries, "z1")
    gold = [r for r in d["results"] if r["is_gold"]]
    assert len(gold) == 1
    assert gold[0]["doc_id"] == "gold"


def test_explain_reports_the_cutoff_for_the_rank_strip(ctx) -> None:
    """The strip needs the cutoff and pool size, not just the rank."""
    store, queries = ctx
    t = explain_payload(store, cfg(k=3, candidate_k=9), queries, "z1")["triage"]
    assert t["cutoff_k"] == 3 and t["candidate_k"] == 9


def test_explain_unfixable_query_says_so(ctx) -> None:
    store, queries = ctx
    d = explain_payload(store, cfg(), queries, "miss", max_changed=1)
    assert d["fix"]["found"] is False
    assert "configs_evaluated" in d["fix"]


def test_explain_is_json_serializable(ctx) -> None:
    store, queries = ctx
    json.dumps(explain_payload(store, cfg(k=1), queries, "z1"))


# --- static assets ---------------------------------------------------------

@pytest.mark.parametrize("name", ["index.html", "app.css", "app.js"])
def test_static_files_are_shipped(name: str) -> None:
    """They are served by path lookup, so a missing one is a 404 at runtime
    rather than an import error."""
    assert (STATIC / name).is_file()


def test_page_references_its_assets() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "/app.css" in html and "/app.js" in html
