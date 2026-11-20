"""Stage 3 and 4: the minimal fix, and how much headroom is left.

Stage 3 answers "what is the smallest change that would have retrieved this?"
Stage 4 answers "and is tuning retrieval even worth it, or am I near the
ceiling?"

Two things make this a tool rather than a toy:

  * **Provable minimality.** Candidates are evaluated in non-decreasing order
    of distance from the baseline, so the first config that works is a true
    minimum under that ordering. No search heuristics, no "good enough".
  * **Regression checking.** A fix that recovers this query while breaking
    four others is not a fix. Every proposed change is re-run against the
    whole query set before it is reported, and the net effect is printed
    whether it flatters the fix or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RetrievalConfig, enumerate_space
from .pipeline import IndexStore, Query
from .triage import Verdict, triage


def retrieves(store: IndexStore, cfg: RetrievalConfig, q: Query) -> bool:
    return triage(store.get(cfg), q).ok


# A config that returns a large fraction of the corpus "retrieves" everything
# and explains nothing. Without this guard the search happily reports
# `k: 5 -> 20` as a fix on a corpus that only has 22 chunks, which is a
# degenerate win: recall 100%, precision junk. The optimizer will cheat if
# you let it, and noticing that is most of the work.
DEGENERATE_K_FRACTION = 0.25


def is_degenerate(store: IndexStore, cfg: RetrievalConfig) -> bool:
    n_chunks = len(store.get(cfg).chunks)
    if n_chunks == 0:
        return True
    return cfg.k > max(5, DEGENERATE_K_FRACTION * n_chunks)


@dataclass
class Fix:
    query_id: str
    config: RetrievalConfig
    changed: dict[str, tuple[object, object]]
    distance: tuple[int, int]
    configs_tried: int
    requires_reembedding: bool
    skipped_degenerate: int = 0
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    @property
    def net(self) -> int:
        return len(self.improvements) - len(self.regressions)

    @property
    def safe(self) -> bool:
        return not self.regressions

    def describe(self) -> str:
        parts = [f"{f}: {a!r} -> {b!r}" for f, (a, b) in sorted(self.changed.items())]
        return ", ".join(parts) if parts else "(no change)"


@dataclass
class Headroom:
    n_queries: int
    baseline_recall: float
    oracle_recall: float          # any config in the space retrieves it
    best_single_config: RetrievalConfig | None
    best_single_recall: float
    unfixable: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float:
        return round(self.oracle_recall - self.baseline_recall, 1)

    @property
    def verdict(self) -> str:
        if self.gap <= 0:
            return ("no configuration in this space does better - the problem "
                    "is not retrieval tuning")
        if self.best_single_recall < self.oracle_recall:
            return (f"no single config reaches the ceiling: best is "
                    f"{self.best_single_recall}% vs {self.oracle_recall}% "
                    f"oracle, so different queries want different settings")
        return (f"one config reaches the ceiling at {self.best_single_recall}%")


@dataclass
class FixSearch:
    """The outcome of one search, including what it refused to consider.

    `fix` is None when nothing legitimate worked - but the counts still
    matter then, because "no fix, and I rejected 3 degenerate ones" is a very
    different report from "no fix, nothing came close". Returning them only
    on success would hide the guard exactly when it mattered most.
    """

    fix: Fix | None
    configs_evaluated: int
    skipped_degenerate: int

    @property
    def found(self) -> bool:
        return self.fix is not None


def search_fix(
    store: IndexStore,
    baseline: RetrievalConfig,
    q: Query,
    queries: list[Query] | None = None,
    space: dict[str, list] | None = None,
    max_changed_fields: int = 2,
) -> FixSearch:
    """Smallest legitimate change to `baseline` that retrieves `q`.

    Because `enumerate_space` yields candidates ordered by
    (fields changed, cost), the first hit is minimal and we stop there.
    """
    candidates = enumerate_space(baseline, space,
                                 max_changed_fields=max_changed_fields)
    tried = 0
    skipped_degenerate = 0
    for cfg in candidates:
        if cfg == baseline:
            continue
        if is_degenerate(store, cfg):
            skipped_degenerate += 1
            continue
        tried += 1
        if not retrieves(store, cfg, q):
            continue

        changed = {f: (getattr(baseline, f), getattr(cfg, f))
                   for f in cfg.changed_fields(baseline)}
        fix = Fix(
            query_id=q.id, config=cfg, changed=changed,
            distance=cfg.distance_from(baseline), configs_tried=tried,
            requires_reembedding=cfg.requires_reembedding(baseline),
            skipped_degenerate=skipped_degenerate,
        )
        if queries:
            _check_regressions(store, baseline, cfg, q, queries, fix)
        return FixSearch(fix=fix, configs_evaluated=tried,
                         skipped_degenerate=skipped_degenerate)

    return FixSearch(fix=None, configs_evaluated=tried,
                     skipped_degenerate=skipped_degenerate)


def find_minimal_fix(
    store: IndexStore,
    baseline: RetrievalConfig,
    q: Query,
    queries: list[Query] | None = None,
    space: dict[str, list] | None = None,
    max_changed_fields: int = 2,
) -> Fix | None:
    """Convenience wrapper: just the fix, or None."""
    return search_fix(store, baseline, q, queries, space,
                      max_changed_fields).fix


def _check_regressions(store, baseline, cfg, target, queries, fix: Fix) -> None:
    for other in queries:
        if other.id == target.id:
            continue
        before = retrieves(store, baseline, other)
        after = retrieves(store, cfg, other)
        if before and not after:
            fix.regressions.append(other.id)
        elif after and not before:
            fix.improvements.append(other.id)


def headroom(
    store: IndexStore,
    baseline: RetrievalConfig,
    queries: list[Query],
    space: dict[str, list] | None = None,
    max_changed_fields: int = 2,
) -> Headroom:
    """Oracle bound over the config space, plus the best single config.

    The difference between those two is the interesting part: when the oracle
    is much higher than any single config, no amount of global tuning will
    get you there and the answer is a better chunking strategy, not a better
    parameter.
    """
    candidates = enumerate_space(baseline, space,
                                 max_changed_fields=max_changed_fields)
    n = len(queries) or 1

    fixable: set[str] = set()
    per_config: dict[RetrievalConfig, int] = {}
    for cfg in candidates:
        if is_degenerate(store, cfg):
            continue
        wins = 0
        for q in queries:
            if retrieves(store, cfg, q):
                wins += 1
                fixable.add(q.id)
        per_config[cfg] = wins

    base_wins = sum(retrieves(store, baseline, q) for q in queries)
    best_cfg = max(per_config, key=lambda c: per_config[c]) if per_config else None

    return Headroom(
        n_queries=len(queries),
        baseline_recall=round(100.0 * base_wins / n, 1),
        oracle_recall=round(100.0 * len(fixable) / n, 1),
        best_single_config=best_cfg,
        best_single_recall=round(100.0 * per_config[best_cfg] / n, 1) if best_cfg else 0.0,
        unfixable=sorted(q.id for q in queries if q.id not in fixable),
    )
