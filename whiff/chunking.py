"""Chunking, with offsets that are actually correct.

Every chunk carries the character span it came from, and `text` must equal
`document[start:end]`. That invariant is load-bearing: localization asks
"which chunk contains the gold span", and an off-by-a-few offset silently
turns a correct diagnosis into a plausible wrong one. It is tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int
    end: int

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.index}"


def _emit(doc_id: str, doc: str, spans: list[tuple[int, int]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(spans):
        text = doc[s:e]
        if text.strip():
            chunks.append(Chunk(doc_id, len(chunks), text, s, e))
    return chunks


def chunk_fixed(doc_id: str, doc: str, size: int, overlap: int) -> list[Chunk]:
    """Fixed character windows. Overlap must be < size or we never advance."""
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    step = max(1, size - max(0, overlap))
    spans = []
    pos = 0
    n = len(doc)
    while pos < n:
        spans.append((pos, min(pos + size, n)))
        if pos + size >= n:
            break
        pos += step
    return _emit(doc_id, doc, spans)


_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_spans(doc: str, pattern: re.Pattern, start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) on `pattern`, returning spans that tile the range."""
    spans = []
    cursor = start
    for m in pattern.finditer(doc, start, end):
        if m.start() > cursor:
            spans.append((cursor, m.start()))
        cursor = m.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans or [(start, end)]


def chunk_recursive(doc_id: str, doc: str, size: int, overlap: int = 0) -> list[Chunk]:
    """Paragraph-aware packing: fill up to `size`, preferring clean breaks.

    Paragraphs that already exceed `size` are split on sentences, and
    sentences that still exceed it are hard-cut.
    """
    units: list[tuple[int, int]] = []
    for ps, pe in _split_spans(doc, _PARA_RE, 0, len(doc)):
        if pe - ps <= size:
            units.append((ps, pe))
            continue
        for ss, se in _split_spans(doc, _SENT_RE, ps, pe):
            if se - ss <= size:
                units.append((ss, se))
            else:
                for cut in range(ss, se, size):
                    units.append((cut, min(cut + size, se)))

    # Greedily pack adjacent units into chunks of at most `size`.
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    for s, e in units:
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        if e - cur_start <= size:
            cur_end = e
        else:
            spans.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return _emit(doc_id, doc, spans)


def chunk_sentence(doc_id: str, doc: str, size: int = 0, overlap: int = 0) -> list[Chunk]:
    """One chunk per sentence. `size` is ignored, by design."""
    spans = _split_spans(doc, _SENT_RE, 0, len(doc))
    return _emit(doc_id, doc, spans)


CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "sentence": chunk_sentence,
}


def chunk_document(doc_id: str, doc: str, cfg) -> list[Chunk]:
    try:
        fn = CHUNKERS[cfg.chunker]
    except KeyError:
        raise ValueError(f"unknown chunker: {cfg.chunker}") from None
    return fn(doc_id, doc, cfg.chunk_size, cfg.overlap)


def chunk_corpus(docs: dict[str, str], cfg) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id in sorted(docs):
        out.extend(chunk_document(doc_id, docs[doc_id], cfg))
    return out
