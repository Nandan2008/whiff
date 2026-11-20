"""Embedding backends and the content-addressed cache.

The cache is the reason a config sweep is feasible at all. A naive sweep
re-embeds the whole corpus for every configuration; but two configs with
different `k` produce *identical chunk text*, and even different chunk sizes
produce many byte-identical chunks. Keying stored vectors on
sha256(embedder_id + chunk_text) means each distinct string is embedded
exactly once, ever - across configs, across runs, across projects.

All vectors are L2-normalized on the way in, so cosine similarity is a dot
product and the index never has to normalize again.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def dim(self) -> int: ...
    @property
    def persist(self) -> bool: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class HashEmbedder:
    """Stateless hashing vectorizer. The default, and the reason this tool
    runs with no model download, no API key and no network.

    Being stateless matters: a TF-IDF embedder would have to be *fitted* on
    the chunk set, which differs per config, so its vectors could not be
    shared between configs and the cache would be worthless. Statelessness is
    what makes content-addressing sound.

    It is lexical, so it will not reproduce dense-retrieval phenomena
    faithfully. Use `st:<model>` for real findings.
    """

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.n_features = n_features
        self._vec = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None,
            lowercase=True, stop_words="english", ngram_range=(1, 2),
        )

    @property
    def id(self) -> str:
        return f"hash-{self.n_features}-ng2"

    @property
    def dim(self) -> int:
        return self.n_features

    @property
    def persist(self) -> bool:
        """Never worth writing to disk.

        A cache earns its keep on (cost to recompute) / (bytes to store).
        This embedder is the worst possible ratio: 2048 float32s is 8 KB per
        vector, and producing one takes microseconds. Persisting a SQuAD-sized
        run wrote a 352 MB file to save work that costs about a second to
        redo. The in-memory layer still applies, which is where the real win
        was anyway.
        """
        return False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(self._vec.transform(list(texts)).toarray())


class SentenceTransformerEmbedder:
    """Real dense embeddings. Needs the model downloaded once."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            # Hub progress bars write to stdout/stderr and interleave with the
            # diagnosis report, which is the actual product here. Set before
            # the import, because the flag is read at import time.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def id(self) -> str:
        return f"st:{self.model_name}"

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    @property
    def persist(self) -> bool:
        """Always worth writing to disk - the inverse of the hash embedder.

        Vectors are smaller (384 floats, ~1.5 KB) and cost milliseconds each,
        so a cached SQuAD run is ~8 MB and saves minutes of CPU.
        """
        return True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._load().encode(list(texts), convert_to_numpy=True,
                                  show_progress_bar=False)
        return _l2(vecs)


_REGISTRY: dict[str, Embedder] = {}


def get_embedder(spec: str) -> Embedder:
    """`hash`, `hash:<n_features>`, or `st:<model-name>`.

    The `hash:<n>` form exists so there are two genuinely different vector
    spaces available offline, which makes hybrid retrieval testable without
    downloading a model. It is also a real knob: fewer features means more
    hash collisions and blurrier matching.

    Instances are reused within a process.
    """
    if spec in _REGISTRY:
        return _REGISTRY[spec]
    if spec == "hash":
        emb: Embedder = HashEmbedder()
    elif spec.startswith("hash:"):
        tail = spec[5:]
        if not tail.isdigit() or int(tail) < 16:
            raise ValueError(f"bad hash spec {spec!r}: want hash:<n>, n >= 16")
        emb = HashEmbedder(n_features=int(tail))
    elif spec.startswith("st:"):
        name = spec[3:]
        if not name:
            raise ValueError("st: needs a model name, e.g. st:all-MiniLM-L6-v2")
        emb = SentenceTransformerEmbedder(name)
    else:
        raise ValueError(
            f"unknown embedder: {spec!r} "
            f"(want 'hash', 'hash:<n>', or 'st:<model>')")
    _REGISTRY[spec] = emb
    return emb


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.total else round(100.0 * self.hits / self.total, 1)

    def __str__(self) -> str:
        return (f"{self.hits} hits / {self.misses} embedded "
                f"({self.hit_rate}% reuse)")


class EmbeddingCache:
    """sha256(embedder_id + text) -> float32 vector, in SQLite."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.stats = CacheStats()
        self._mem: dict[str, np.ndarray] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(str(self.path))
            # WAL + synchronous=NORMAL drops the durability guarantee from
            # "survives power loss" to "survives process crash", which is the
            # right trade for a cache whose worst case is re-embedding some
            # text.
            #
            # Honest note on magnitude: when the cold sweep looked slow, these
            # pragmas were the obvious suspect and they bought almost nothing
            # (2.05s -> 1.95s, within noise). Profiling showed the real cost
            # was a one-time `import sklearn`. Kept because the trade is still
            # correct for this data, not because it was the fix.
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS vectors ("
                "  key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
            )
            self._con.commit()
        else:
            self._con = None

    @staticmethod
    def key(embedder_id: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(embedder_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _get_many(self, keys: list[str], persist: bool = True) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        missing = []
        for k in keys:
            if k in self._mem:
                found[k] = self._mem[k]
            else:
                missing.append(k)
        if persist and self._con is not None and missing:
            # Chunk the IN clause; SQLite caps variables per statement.
            for i in range(0, len(missing), 500):
                batch = missing[i:i + 500]
                q = "SELECT key, dim, vec FROM vectors WHERE key IN (%s)" % (
                    ",".join("?" * len(batch))
                )
                for key, dim, blob in self._con.execute(q, batch):
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if vec.size != dim:
                        continue  # corrupt row; treat as a miss
                    self._mem[key] = vec
                    found[key] = vec
        return found

    def _put_many(self, items: list[tuple[str, np.ndarray]],
                  persist: bool = True) -> None:
        for key, vec in items:
            self._mem[key] = vec
        if persist and self._con is not None and items:
            self._con.executemany(
                "INSERT OR REPLACE INTO vectors (key, dim, vec) VALUES (?,?,?)",
                [(k, int(v.size), v.astype(np.float32).tobytes()) for k, v in items],
            )
            self._con.commit()

    def encode(self, embedder: Embedder, texts: Sequence[str]) -> np.ndarray:
        """Return one row per input text, in order, using the cache."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        # The embedder decides whether its vectors are worth disk. Cheap,
        # high-dimensional vectors stay in memory only; expensive ones persist.
        persist = getattr(embedder, "persist", True)

        keys = [self.key(embedder.id, t) for t in texts]
        cached = self._get_many(keys, persist=persist)

        # Deduplicate misses: the same string may appear twice in one call.
        todo_keys: list[str] = []
        todo_texts: list[str] = []
        seen: set[str] = set()
        for k, t in zip(keys, texts):
            if k not in cached and k not in seen:
                seen.add(k)
                todo_keys.append(k)
                todo_texts.append(t)

        if todo_texts:
            fresh = embedder.encode(todo_texts)
            self._put_many(list(zip(todo_keys, fresh)), persist=persist)
            for k, v in zip(todo_keys, fresh):
                cached[k] = v

        self.stats.hits += len(texts) - len(todo_texts)
        self.stats.misses += len(todo_texts)
        return np.vstack([cached[k] for k in keys]).astype(np.float32)

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
