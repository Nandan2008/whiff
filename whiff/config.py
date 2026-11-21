"""Retrieval configuration and the space of configurations to search.

A `RetrievalConfig` is one complete, reproducible retrieval setup. The
counterfactual search walks a space of these looking for the smallest change
that recovers a document the baseline missed.

"Smallest" needs a definition, and this is it:

  1. fewest changed fields, then
  2. lowest cost, where a change that forces re-embedding costs more than one
     that only re-ranks what is already embedded.

That ordering is a deliberate choice, not an obvious truth. Changing `k` from
5 to 20 touches one field and needs no new vectors, so it is "smaller" than
switching embedding model even if a user might find the latter more
acceptable. The ordering is also what makes the search provably minimal:
candidates are evaluated in non-decreasing order of that key, so the first
success is a true minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Iterator

# Fields whose change invalidates stored vectors, because they change the
# chunk text itself or the model that encodes it.
REEMBED_FIELDS = frozenset({"chunker", "chunk_size", "overlap", "embedder",
                            "hybrid_with"})


@dataclass(frozen=True, order=True)
class RetrievalConfig:
    chunker: str = "fixed"            # fixed | recursive | sentence
    chunk_size: int = 400             # characters (fixed/recursive)
    overlap: int = 0                  # characters of overlap (fixed)
    embedder: str = "hash"            # hash | st:<model-name>
    k: int = 5                        # final result count
    candidate_k: int = 50             # first-stage candidate pool
    context_budget: int | None = None  # chars of context; None = no packing
    # Hybrid retrieval: score with a second embedder and blend. The standard
    # answer to vocabulary mismatch, where a lexical retriever misses a
    # paraphrase and a dense one misses an exact rare term.
    hybrid_with: str | None = None     # second embedder spec, or None
    hybrid_weight: float = 0.5         # weight on the SECOND embedder

    def changed_fields(self, other: "RetrievalConfig") -> tuple[str, ...]:
        return tuple(
            f for f in self.__dataclass_fields__
            if getattr(self, f) != getattr(other, f)
        )

    def requires_reembedding(self, other: "RetrievalConfig") -> bool:
        return bool(REEMBED_FIELDS.intersection(self.changed_fields(other)))

    def distance_from(self, baseline: "RetrievalConfig") -> tuple[int, int]:
        """The minimality key: (fields changed, cost tier)."""
        n = len(self.changed_fields(baseline))
        cost = 1 if self.requires_reembedding(baseline) else 0
        return (n, cost)

    def index_key(self) -> tuple:
        """Identity of the built index - fields that change the chunk set.

        `hybrid_with` is coerced to "" rather than left as None so the tuple
        stays sortable: a space containing both None and a model name would
        otherwise raise TypeError when enumerate_space sorts on this.
        """
        return (self.chunker, self.chunk_size, self.overlap, self.embedder,
                self.hybrid_with or "")

    def with_(self, **kw) -> "RetrievalConfig":
        return replace(self, **kw)

    def describe(self) -> str:
        base = (f"{self.chunker}/{self.chunk_size}+{self.overlap} "
                f"{self.embedder} k={self.k}")
        if self.hybrid_with:
            base += f" +{self.hybrid_with}@{self.hybrid_weight:g}"
        return base


# The v1 search space. Deliberately four axes, not eight: a space you can
# actually enumerate beats one you have to sample.
DEFAULT_SPACE: dict[str, list] = {
    "chunker": ["fixed", "recursive", "sentence"],
    "chunk_size": [150, 300, 600, 1200],
    "overlap": [0, 75],
    "k": [5, 10, 20],
}


def enumerate_space(
    baseline: RetrievalConfig,
    space: dict[str, list] | None = None,
    max_changed_fields: int | None = None,
) -> list[RetrievalConfig]:
    """All configs in the space, ordered cheapest-first from `baseline`.

    Ordering is the whole trick. Because candidates come back sorted by
    `distance_from(baseline)`, a caller can stop at the first config that
    fixes the query and know nothing smaller exists.
    """
    space = space or DEFAULT_SPACE
    keys = sorted(space)

    # Every axis must be able to hold its CURRENT value, or the baseline is
    # not representable in its own search space: with chunk_size=400 and a
    # grid of [150, 300, 600, 1200], every candidate differs in chunk_size,
    # so a one-field `k` fix can never be found and the reported minimum
    # overstates what actually changed.
    grid = {}
    for key in keys:
        values = list(space[key])
        current = getattr(baseline, key)
        if current not in values:
            values.insert(0, current)
        grid[key] = values

    seen: set[RetrievalConfig] = set()
    out: list[RetrievalConfig] = []

    for values in product(*(grid[k] for k in keys)):
        cfg = baseline.with_(**dict(zip(keys, values)))
        if cfg in seen:
            continue
        # `overlap` is only meaningful for the fixed chunker; collapse the
        # duplicates so the space does not contain equivalent configs.
        if cfg.chunker != "fixed" and cfg.overlap != baseline.overlap:
            continue
        if max_changed_fields is not None:
            if len(cfg.changed_fields(baseline)) > max_changed_fields:
                continue
        seen.add(cfg)
        out.append(cfg)

    out.sort(key=lambda c: (c.distance_from(baseline), c.index_key(), c.k))
    return out


def iter_space(
    baseline: RetrievalConfig,
    space: dict[str, list] | None = None,
) -> Iterator[RetrievalConfig]:
    yield from enumerate_space(baseline, space)
