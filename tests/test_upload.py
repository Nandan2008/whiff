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


# --- PDF and Word ----------------------------------------------------------

def b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def make_pdf(lines: list[str]) -> bytes:
    """A real PDF with a real text layer, built in memory."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for line in lines:
        page = doc.new_page()
        page.insert_text((72, 96), line, fontsize=12)
    return doc.tobytes()


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    from io import BytesIO
    docx = pytest.importorskip("docx")
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    if table:
        t = d.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    buf = BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_pdf_text_is_extracted() -> None:
    data = make_pdf(["The bins go out on Tuesday night.",
                     "The spare key is under the blue pot."])
    docs, problems = docs_from_uploads([{"name": "house.pdf", "b64": b64(data)}])
    assert problems == []
    assert list(docs) == ["house"]
    assert "Tuesday night" in docs["house"]
    assert "blue pot" in docs["house"], "every page must be read, not just one"


def test_pdf_with_no_text_layer_is_reported_not_silently_empty() -> None:
    """A scanned PDF is page images with no text. Say so rather than
    pretending the document was blank."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()                               # a page with nothing on it
    docs, problems = docs_from_uploads(
        [{"name": "scan.pdf", "b64": b64(doc.tobytes())}])
    assert docs == {}
    assert any("scan.pdf" in p and "no text" in p for p in problems)


def test_docx_paragraphs_are_extracted() -> None:
    data = make_docx(["Returns are accepted for fourteen days.",
                      "Damaged goods must be photographed."])
    docs, problems = docs_from_uploads([{"name": "policy.docx", "b64": b64(data)}])
    assert problems == []
    assert "fourteen days" in docs["policy"]
    assert "photographed" in docs["policy"]


def test_docx_tables_are_extracted() -> None:
    """Policy documents put the actual numbers in tables."""
    data = make_docx(["Rates:"], table=[["Region", "Rate"], ["North", "45p"]])
    docs, _ = docs_from_uploads([{"name": "rates.docx", "b64": b64(data)}])
    assert "45p" in docs["rates"]


def test_corrupt_binary_is_a_complaint_not_a_crash() -> None:
    docs, problems = docs_from_uploads(
        [{"name": "broken.pdf", "b64": b64(b"this is not a pdf at all")}])
    assert docs == {}
    assert any("broken.pdf" in p for p in problems)


def test_unsupported_binary_extension_is_rejected_clearly() -> None:
    docs, problems = docs_from_uploads(
        [{"name": "photo.png", "b64": b64(b"\x89PNG\r\n")}])
    assert docs == {}
    assert any("unsupported" in p for p in problems)


def test_binary_and_text_files_mix_in_one_upload() -> None:
    docs, problems = docs_from_uploads([
        {"name": "notes.txt", "text": "plain text notes"},
        {"name": "house.pdf", "b64": b64(make_pdf(["pdf contents here"]))},
        {"name": "policy.docx", "b64": b64(make_docx(["word contents here"]))},
    ])
    assert problems == []
    assert set(docs) == {"notes", "house", "policy"}


def test_binary_extensions_are_stripped_from_ids() -> None:
    docs, _ = docs_from_uploads([
        {"name": "reports/Q3 Summary.PDF", "b64": b64(make_pdf(["x"]))}])
    assert list(docs) == ["Q3 Summary"]


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
