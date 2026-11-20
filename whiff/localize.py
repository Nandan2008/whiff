"""Stage 2: where does the answer actually live, and how is it ranked?

The headline output is the pair (chunk rank, document rank). When a chunk
ranks well but the document it belongs to ranks badly, you are looking at
dilution: one decisive sentence averaged away by a long document's worth of
unrelated text. That is a chunking problem wearing a retrieval problem's
clothes, and the two numbers side by side make it obvious.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import Query, Retriever, contains_span


@dataclass
class Localization:
    query: Query
    gold_doc: str
    doc_chunk_count: int = 0
    gold_chunk_id: str | None = None
    gold_chunk_position: int | None = None     # which chunk of the document
    chunk_rank: int | None = None              # rank among all chunks
    chunk_score: float | None = None
    doc_rank: int | None = None                # rank among whole documents
    doc_score: float | None = None
    n_docs: int = 0
    span_offset_in_chunk: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def dilution(self) -> bool:
        """Chunk findable, document not."""
        if self.chunk_rank is None or self.doc_rank is None:
            return False
        return self.chunk_rank <= self.query_k and self.doc_rank > self.query_k

    query_k: int = 5

    def summary(self) -> str:
        if self.gold_chunk_id is None:
            return "no single chunk contains the answer"
        bits = [
            f"answer is in chunk {self.gold_chunk_position} of "
            f"{self.doc_chunk_count}",
            f"that chunk ranks {self.chunk_rank}",
        ]
        if self.doc_rank is not None:
            bits.append(f"the whole document ranks {self.doc_rank} of {self.n_docs}")
        return "; ".join(bits)


def localize(retriever: Retriever, q: Query) -> Localization:
    cfg = retriever.cfg
    loc = Localization(query=q, gold_doc=q.gold_doc, query_k=cfg.k)

    doc_chunks = [c for c in retriever.chunks if c.doc_id == q.gold_doc]
    loc.doc_chunk_count = len(doc_chunks)
    loc.n_docs = len(retriever.corpus.docs)

    gold = retriever.gold_chunks(q)
    if gold:
        best = gold[0]
        loc.gold_chunk_id = best.id
        loc.gold_chunk_position = best.index + 1
        if q.gold_span:
            from .pipeline import normalize
            loc.span_offset_in_chunk = normalize(best.text).find(
                normalize(q.gold_span))

        gold_ids = {c.id for c in gold}
        hit = next((r for r in retriever.candidates(q.text)
                    if r.chunk.id in gold_ids), None)
        if hit is not None:
            loc.chunk_rank, loc.chunk_score = hit.rank, hit.score
        else:
            loc.notes.append("gold chunk exists but is outside the candidate pool")
    else:
        loc.notes.append("no chunk fully contains the gold span")

    # Document-level ranking, for the dilution comparison. Embedding whole
    # documents is exactly what a naive "one vector per document" setup does.
    doc_ids = sorted(retriever.corpus.docs)
    if doc_ids:
        mat = retriever.cache.encode(
            retriever.embedder, [retriever.corpus.docs[d] for d in doc_ids])
        qv = retriever.cache.encode(retriever.embedder, [q.text])[0]
        scores = mat @ qv
        order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])
        for rank, i in enumerate(order, start=1):
            if doc_ids[i] == q.gold_doc:
                loc.doc_rank, loc.doc_score = rank, float(scores[i])
                break

    if loc.dilution:
        loc.notes.append(
            f"dilution: the right chunk ranks {loc.chunk_rank} but the "
            f"document ranks {loc.doc_rank} - chunk-level retrieval is "
            f"doing the work here, document-level would miss it")

    if gold and len(gold) > 1:
        loc.notes.append(f"{len(gold)} chunks contain the span (overlap is "
                         f"duplicating it)")

    return loc


def fragmentation_report(retriever: Retriever, q: Query) -> str | None:
    """If the chunker split the answer, say where."""
    if not q.gold_span:
        return None
    doc = retriever.corpus_text(q.gold_doc)
    if not contains_span(doc, q.gold_span):
        return None
    if retriever.gold_chunks(q):
        return None
    doc_chunks = [c for c in retriever.chunks if c.doc_id == q.gold_doc]
    for a, b in zip(doc_chunks, doc_chunks[1:]):
        joined = doc[a.start:b.end]
        if contains_span(joined, q.gold_span):
            return (f"the answer straddles the boundary between chunks "
                    f"{a.index + 1} and {b.index + 1} (at character {b.start} "
                    f"of the document); overlap would fix this")
    return "the answer spans more than two chunks at this chunk size"
