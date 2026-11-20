"""Stage 1: what kind of failure is this?

These five verdicts have completely different fixes, and the single most
common waste of time in RAG work is applying one bucket's fix to another
bucket's problem - tuning the reranker when the document was never indexed,
or swapping embedding models when the answer was split down the middle by
the chunker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .pipeline import Query, Retrieved, Retriever, contains_span


class Verdict(str, Enum):
    RETRIEVED = "retrieved"                        # no failure
    NOT_INDEXED = "not_indexed"                    # ingestion
    SPAN_FRAGMENTED = "span_fragmented"            # chunker split the answer
    ABSENT_FROM_CANDIDATES = "absent_from_candidates"  # embedding / index
    BELOW_CUTOFF = "below_cutoff"                  # ranking
    DROPPED_IN_PACKING = "dropped_in_packing"      # context assembly

    @property
    def blame(self) -> str:
        return {
            Verdict.RETRIEVED: "nothing - retrieval succeeded",
            Verdict.NOT_INDEXED: "ingestion: the document is not in the corpus",
            Verdict.SPAN_FRAGMENTED: "chunking: no single chunk contains the answer",
            Verdict.ABSENT_FROM_CANDIDATES: "embedding or index: gold never "
                                            "reached the candidate pool",
            Verdict.BELOW_CUTOFF: "ranking or k: gold was a candidate but "
                                  "ranked too low",
            Verdict.DROPPED_IN_PACKING: "context assembly: gold was retrieved, "
                                        "then discarded by the budget",
        }[self]


@dataclass
class Triage:
    query: Query
    verdict: Verdict
    gold_chunk_count: int = 0
    best_rank: int | None = None          # rank of best gold chunk in candidates
    best_score: float | None = None
    cutoff_k: int = 0
    candidate_k: int = 0
    top_results: list[Retrieved] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.RETRIEVED


def triage(retriever: Retriever, q: Query) -> Triage:
    cfg = retriever.cfg
    t = Triage(query=q, verdict=Verdict.RETRIEVED,
               cutoff_k=cfg.k, candidate_k=cfg.candidate_k)

    if q.gold_doc not in retriever.index_docs():
        t.verdict = Verdict.NOT_INDEXED
        t.notes.append(f"document {q.gold_doc!r} is not in the corpus")
        return t

    gold = retriever.gold_chunks(q)
    t.gold_chunk_count = len(gold)

    if not gold:
        # The document is present, so either the span text is wrong or the
        # chunker cut through it. Distinguish those - they are different bugs.
        doc = retriever.corpus_text(q.gold_doc)
        if q.gold_span and not contains_span(doc, q.gold_span):
            t.verdict = Verdict.NOT_INDEXED
            t.notes.append("gold span does not appear in the gold document - "
                           "check the span text, not the retriever")
        else:
            t.verdict = Verdict.SPAN_FRAGMENTED
            t.notes.append(f"the answer spans a chunk boundary at "
                           f"{cfg.chunker}/{cfg.chunk_size}+{cfg.overlap}")
        return t

    gold_ids = {c.id for c in gold}
    candidates = retriever.candidates(q.text)
    hit = next((r for r in candidates if r.chunk.id in gold_ids), None)

    if hit is None:
        t.verdict = Verdict.ABSENT_FROM_CANDIDATES
        t.top_results = candidates[:5]
        t.notes.append(f"no gold chunk in the top {cfg.candidate_k} candidates")
        return t

    t.best_rank, t.best_score = hit.rank, hit.score
    t.top_results = candidates[: max(cfg.k, 5)]

    if hit.rank > cfg.k:
        t.verdict = Verdict.BELOW_CUTOFF
        t.notes.append(f"gold chunk ranks {hit.rank}, cutoff is k={cfg.k}")
        return t

    if cfg.context_budget is not None:
        kept = {r.chunk.id for r in retriever.packed(q.text)}
        if not gold_ids.intersection(kept):
            t.verdict = Verdict.DROPPED_IN_PACKING
            t.notes.append(f"gold chunk ranked {hit.rank} but did not fit the "
                           f"{cfg.context_budget}-char budget")
            return t

    return t


def triage_all(retriever: Retriever, queries: list[Query]) -> list[Triage]:
    return [triage(retriever, q) for q in queries]


def summarize(triages: list[Triage]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for t in triages:
        counts[t.verdict.value] = counts.get(t.verdict.value, 0) + 1
    n = len(triages) or 1
    return {
        "n_queries": len(triages),
        "n_retrieved": sum(t.ok for t in triages),
        "recall_at_k": round(100.0 * sum(t.ok for t in triages) / n, 1),
        "verdicts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
