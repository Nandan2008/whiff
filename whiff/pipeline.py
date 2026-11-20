"""Corpus, queries, and building a retriever for a given config.

Two layers of reuse make the sweep cheap, and they are worth separating:

  * `IndexStore` - configs that differ only in `k` or `context_budget` share
    one built index outright, keyed on `cfg.index_key()`.
  * `EmbeddingCache` - configs with different chunk settings still share every
    byte-identical chunk's vector.

The first avoids rebuilding; the second avoids re-embedding. Together they
turn a quadratic-feeling sweep into something that runs in seconds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_corpus
from .config import RetrievalConfig
from .embed import EmbeddingCache, Embedder, get_embedder
from .index import BruteForceIndex

_WS = re.compile(r"\s+")


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A flat score vector maps to all zeros."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi <= lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def contains_span(haystack: str, needle: str) -> bool:
    """Whitespace- and case-insensitive containment.

    Chunk text is an exact substring of the document, but a gold span typed
    into a JSON file rarely matches the document's whitespace exactly, so
    comparing raw strings produces false negatives that look like real
    retrieval failures.
    """
    if not needle:
        return False
    return normalize(needle) in normalize(haystack)


@dataclass
class Query:
    id: str
    text: str
    gold_doc: str
    gold_span: str | None = None

    @staticmethod
    def from_dict(d: dict) -> "Query":
        return Query(
            id=str(d["id"]), text=str(d["text"]),
            gold_doc=str(d["gold_doc"]), gold_span=d.get("gold_span"),
        )


@dataclass
class Corpus:
    docs: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_jsonl(path: Path | str) -> "Corpus":
        docs: dict[str, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                docs[str(rec["id"])] = str(rec["text"])
        return Corpus(docs)

    def __len__(self) -> int:
        return len(self.docs)


def load_queries(path: Path | str) -> list[Query]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Query.from_dict(json.loads(line)))
    return out


@dataclass
class Retrieved:
    rank: int           # 1-based
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, cfg: RetrievalConfig, chunks: list[Chunk],
                 index: BruteForceIndex, embedder: Embedder,
                 cache: EmbeddingCache, corpus: "Corpus",
                 secondary: tuple[Embedder, BruteForceIndex] | None = None):
        self.cfg = cfg
        self.chunks = chunks
        self.index = index
        self.embedder = embedder
        self.cache = cache
        self.corpus = corpus
        self.secondary = secondary
        self._qcache: dict[str, np.ndarray] = {}

    def index_docs(self) -> set[str]:
        """Document ids actually present in the corpus."""
        return set(self.corpus.docs)

    def corpus_text(self, doc_id: str) -> str:
        return self.corpus.docs.get(doc_id, "")

    def _query_vec(self, text: str) -> np.ndarray:
        if text not in self._qcache:
            self._qcache[text] = self.cache.encode(self.embedder, [text])[0]
        return self._qcache[text]

    def candidates(self, text: str) -> list[Retrieved]:
        """First-stage pool. Models what an ANN index would hand a reranker."""
        if self.secondary is None:
            hits = self.index.search(self._query_vec(text), self.cfg.candidate_k)
        else:
            hits = BruteForceIndex.top_k(self._blended_scores(text),
                                         self.cfg.candidate_k)
        return [Retrieved(i + 1, self.chunks[row], score)
                for i, (row, score) in enumerate(hits)]

    def _blended_scores(self, text: str) -> np.ndarray:
        """Min-max normalize each retriever's scores, then blend.

        Normalizing per query is not optional. Lexical cosines on a hashing
        vectorizer cluster near zero while dense cosines spread across most of
        [0, 1], so a raw weighted sum is dominated by whichever embedder
        happens to have the larger numeric range - the weight would not mean
        what it says. Normalizing first makes `hybrid_weight` an actual
        mixing fraction.
        """
        emb2, index2 = self.secondary
        s1 = _minmax(self.index.scores(self._query_vec(text)))
        q2 = self.cache.encode(emb2, [text])[0]
        s2 = _minmax(index2.scores(q2))
        w = float(self.cfg.hybrid_weight)
        return (1.0 - w) * s1 + w * s2

    def final(self, text: str) -> list[Retrieved]:
        return self.candidates(text)[: self.cfg.k]

    def packed(self, text: str) -> list[Retrieved]:
        """Apply the context budget, greedily, in rank order."""
        results = self.final(text)
        if self.cfg.context_budget is None:
            return results
        kept, used = [], 0
        for r in results:
            cost = len(r.chunk.text)
            if used + cost > self.cfg.context_budget:
                continue
            kept.append(r)
            used += cost
        return kept

    def gold_chunks(self, q: Query) -> list[Chunk]:
        """Chunks that should satisfy the query.

        With a gold span, only chunks that fully contain it count - a chunk
        from the right document that happens to omit the answer is not a hit,
        and conflating the two is the most common way RAG evaluation flatters
        itself.
        """
        in_doc = [c for c in self.chunks if c.doc_id == q.gold_doc]
        if not q.gold_span:
            return in_doc
        return [c for c in in_doc if contains_span(c.text, q.gold_span)]


class IndexStore:
    """Memoizes built indexes within one process, by index_key()."""

    def __init__(self, corpus: Corpus, cache: EmbeddingCache):
        self.corpus = corpus
        self.cache = cache
        self._built: dict[tuple, tuple] = {}
        self.builds = 0

    def get(self, cfg: RetrievalConfig) -> Retriever:
        key = cfg.index_key()
        if key not in self._built:
            self.builds += 1
            embedder = get_embedder(cfg.embedder)
            chunks = chunk_corpus(self.corpus.docs, cfg)
            texts = [c.text for c in chunks]
            vecs = self.cache.encode(embedder, texts)
            index = BruteForceIndex(vecs, [c.id for c in chunks])

            secondary = None
            if cfg.hybrid_with:
                # Same chunk set, second vector space - only the vectors are
                # extra work, and the embedding cache covers those too.
                emb2 = get_embedder(cfg.hybrid_with)
                vecs2 = self.cache.encode(emb2, texts)
                secondary = (emb2, BruteForceIndex(vecs2, [c.id for c in chunks]))

            self._built[key] = (chunks, index, embedder, secondary)
        chunks, index, embedder, secondary = self._built[key]
        return Retriever(cfg, chunks, index, embedder, self.cache, self.corpus,
                         secondary)


def build(corpus: Corpus, cfg: RetrievalConfig,
          cache: EmbeddingCache | None = None) -> Retriever:
    """Convenience for one-off builds outside a sweep."""
    return IndexStore(corpus, cache or EmbeddingCache()).get(cfg)
