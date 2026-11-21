"""Uploading documents, and asking questions with no known answer.

Upload parsing is where malformed input arrives, so it is tested hard: a
half-written .jsonl or an empty file must produce a complaint, never an
exception that takes the server down.
"""

from __future__ import annotations

import json

import pytest

from conftest import cfg
from whiff.pipeline import Corpus, IndexStore
from whiff.web import docs_from_uploads, search_payload


# --- plain text files ------------------------------------------------------

def test_filename_becomes_the_document_id() -> None:
    docs, problems = docs_from_uploads([{"name": "returns.txt", "text": "body"}])
    assert docs == {"returns": "body"}
    assert problems == []


@pytest.mark.parametrize("name,expected", [
    ("notes.md", "notes"),
    ("notes.markdown", "notes"),
    ("notes.text", "notes"),
    ("notes.TXT", "notes"),
    ("house rules.md", "house rules"),
    ("no-extension", "no-extension"),
])
def test_extensions_are_stripped(name: str, expected: str) -> None:
    docs, _ = docs_from_uploads([{"name": name, "text": "body"}])
    assert list(docs) == [expected]


def test_directory_prefixes_are_dropped() -> None:
    """Browsers send a relative path for a dropped folder."""
    docs, _ = docs_from_uploads([{"name": "docs/sub/policy.txt", "text": "b"}])
    assert list(docs) == ["policy"]
    docs, _ = docs_from_uploads([{"name": r"docs\sub\policy.txt", "text": "b"}])
    assert list(docs) == ["policy"]


def test_duplicate_names_do_not_overwrite_each_other() -> None:
    docs, _ = docs_from_uploads([
        {"name": "a/notes.txt", "text": "first"},
        {"name": "b/notes.txt", "text": "second"},
    ])
    assert len(docs) == 2
    assert set(docs.values()) == {"first", "second"}


def test_empty_files_are_reported_not_silently_dropped() -> None:
    docs, problems = docs_from_uploads([
        {"name": "good.txt", "text": "something"},
        {"name": "blank.txt", "text": "   \n "},
    ])
    assert list(docs) == ["good"]
    assert any("blank.txt" in p for p in problems)


def test_nothing_usable_yields_no_docs_and_a_complaint() -> None:
    docs, problems = docs_from_uploads([{"name": "x.txt", "text": ""}])
    assert docs == {}
    assert problems


# --- jsonl -----------------------------------------------------------------

def test_jsonl_is_read_as_records() -> None:
    body = "\n".join(json.dumps({"id": f"d{i}", "text": f"body {i}"})
                     for i in range(3))
    docs, problems = docs_from_uploads([{"name": "corpus.jsonl", "text": body}])
    assert list(docs) == ["d0", "d1", "d2"]
    assert problems == []


def test_broken_jsonl_lines_are_reported_with_line_numbers() -> None:
    body = '{"id": "ok", "text": "fine"}\nnot json at all\n{"missing": "id"}'
    docs, problems = docs_from_uploads([{"name": "c.jsonl", "text": body}])
    assert list(docs) == ["ok"]
    assert len(problems) == 2
    assert any("line 2" in p for p in problems)
    assert any("line 3" in p for p in problems)


def test_blank_lines_in_jsonl_are_not_errors() -> None:
    body = '{"id":"a","text":"x"}\n\n\n{"id":"b","text":"y"}\n'
    docs, problems = docs_from_uploads([{"name": "c.jsonl", "text": body}])
    assert list(docs) == ["a", "b"]
    assert problems == []


def test_jsonl_with_no_usable_lines_complains() -> None:
    docs, problems = docs_from_uploads([{"name": "c.jsonl", "text": "junk\n"}])
    assert docs == {}
    assert any("c.jsonl" in p for p in problems)


def test_missing_name_or_text_does_not_raise() -> None:
    docs, problems = docs_from_uploads([{}, {"name": "a.txt"}, {"text": "b"}])
    assert isinstance(docs, dict) and isinstance(problems, list)


# --- free-text search ------------------------------------------------------

@pytest.fixture
def store(cache) -> IndexStore:
    return IndexStore(Corpus({
        "rules": "No shoes indoors. Bins go out on Tuesday night.",
        "pets": "The dog eats twice a day. The cat is fed in the morning.",
    }), cache)


def test_search_ranks_the_relevant_document_first(store) -> None:
    d = search_payload(store, cfg(k=2), "when do the bins go out")
    assert d["results"][0]["doc_id"] == "rules"


def test_search_marks_the_cutoff(store) -> None:
    d = search_payload(store, cfg(k=1), "bins")
    assert d["cutoff"] == 1
    assert d["results"][0]["inside_cutoff"] is True
    assert all(not r["inside_cutoff"] for r in d["results"][1:])


def test_search_has_no_verdict(store) -> None:
    """Nobody said which document is right, so there must be no judgement."""
    d = search_payload(store, cfg(), "bins")
    assert "verdict" not in d
    assert all("is_gold" not in r for r in d["results"])


def test_blank_question_returns_nothing(store) -> None:
    for q in ("", "   ", None):
        assert search_payload(store, cfg(), q)["results"] == []


def test_search_is_json_serializable(store) -> None:
    json.dumps(search_payload(store, cfg(), "bins"))
