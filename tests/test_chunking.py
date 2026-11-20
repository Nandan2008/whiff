"""Chunking invariants.

The offset invariant is the one that matters most in the whole codebase:
localization asks "which chunk contains the gold span", so an offset that is
wrong by a few characters produces a confident, plausible, wrong diagnosis -
the worst failure mode a debugger can have.
"""

from __future__ import annotations

import pytest

from conftest import cfg
from whiff.chunking import (CHUNKERS, chunk_corpus, chunk_fixed,
                            chunk_recursive, chunk_sentence, chunk_document)

DOC = (
    "First paragraph about alpha. It has two sentences.\n\n"
    "Second paragraph about beta, which is considerably longer than the "
    "first one and will need splitting at small chunk sizes.\n\n"
    "Third paragraph. Short."
)


@pytest.mark.parametrize("name", sorted(CHUNKERS))
@pytest.mark.parametrize("size", [20, 50, 150, 10_000])
def test_offsets_match_source_text(name: str, size: int) -> None:
    """chunk.text must equal doc[start:end] for every chunker and size."""
    chunks = CHUNKERS[name]("d", DOC, size, 0)
    for c in chunks:
        assert c.text == DOC[c.start:c.end], (
            f"{name}/{size} chunk {c.index}: offsets disagree with text")


@pytest.mark.parametrize("name", sorted(CHUNKERS))
def test_chunks_are_ordered_and_indexed(name: str) -> None:
    chunks = CHUNKERS[name]("d", DOC, 60, 0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(a.start <= b.start for a, b in zip(chunks, chunks[1:]))
    assert all(c.doc_id == "d" for c in chunks)


def test_fixed_covers_whole_document_without_overlap() -> None:
    chunks = chunk_fixed("d", DOC, 40, 0)
    assert "".join(c.text for c in chunks) == DOC
    assert chunks[0].start == 0
    assert chunks[-1].end == len(DOC)


def test_fixed_respects_size_bound() -> None:
    for size in (10, 37, 200):
        for c in chunk_fixed("d", DOC, size, 0):
            assert len(c.text) <= size


def test_fixed_overlap_advances_and_duplicates() -> None:
    """Overlap must still terminate, and must actually overlap."""
    plain = chunk_fixed("d", DOC, 50, 0)
    lapped = chunk_fixed("d", DOC, 50, 25)
    assert len(lapped) > len(plain)
    assert lapped[1].start < plain[1].start


def test_overlap_equal_to_size_does_not_hang() -> None:
    """step = max(1, size - overlap) is what prevents an infinite loop."""
    chunks = chunk_fixed("d", DOC, 30, 30)
    assert len(chunks) <= len(DOC) + 1
    assert chunks[-1].end == len(DOC)


def test_zero_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_fixed("d", DOC, 0, 0)


def test_recursive_respects_size_bound() -> None:
    for c in chunk_recursive("d", DOC, 80):
        assert len(c.text) <= 80


def test_recursive_prefers_paragraph_breaks_when_they_fit() -> None:
    """With room for whole paragraphs, chunks should not cut mid-paragraph."""
    chunks = chunk_recursive("d", DOC, 10_000)
    assert len(chunks) == 1  # everything fits in one


def test_sentence_chunker_ignores_size() -> None:
    a = chunk_sentence("d", DOC, 10)
    b = chunk_sentence("d", DOC, 9999)
    assert [c.text for c in a] == [c.text for c in b]
    assert len(a) > 1


def test_blank_chunks_are_dropped() -> None:
    doc = "alpha" + " " * 200 + "beta"
    chunks = chunk_fixed("d", doc, 50, 0)
    assert all(c.text.strip() for c in chunks)


def test_unknown_chunker_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_document("d", DOC, cfg(chunker="nope"))


def test_chunk_corpus_is_deterministic_and_sorted_by_doc() -> None:
    docs = {"b": DOC, "a": DOC}
    first = chunk_corpus(docs, cfg(chunk_size=60))
    second = chunk_corpus(docs, cfg(chunk_size=60))
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.doc_id for c in first][0] == "a"


def test_chunk_ids_are_unique() -> None:
    chunks = chunk_corpus({"a": DOC, "b": DOC}, cfg(chunk_size=30))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
