"""Command line interface.

    python -m whiff.cli triage
    python -m whiff.cli explain q3
    python -m whiff.cli headroom
    python -m whiff.cli bench
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import DEFAULT_SPACE, RetrievalConfig
from .counterfactual import (DEGENERATE_K_FRACTION, headroom, retrieves,
                             search_fix)
from .embed import EmbeddingCache
from .localize import fragmentation_report, localize
from .pipeline import Corpus, IndexStore, load_queries
from .triage import summarize, triage, triage_all

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RULE = "-" * 72


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--corpus", default=str(DATA / "sample_corpus.jsonl"))
    ap.add_argument("--queries", default=str(DATA / "sample_queries.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / ".whiff-cache.sqlite"))
    ap.add_argument("--embedder", default="hash",
                    help="hash (default, offline) or st:<model-name>")
    ap.add_argument("--chunker", default="fixed",
                    choices=["fixed", "recursive", "sentence"])
    ap.add_argument("--chunk-size", type=int, default=150)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("-k", "--top-k", dest="k", type=int, default=5,
                    help="final result count")
    ap.add_argument("--candidate-k", type=int, default=20)
    ap.add_argument("--context-budget", type=int, default=None)
    ap.add_argument("--hybrid-with", default=None,
                    help="blend a second embedder's scores, e.g. "
                         "st:all-MiniLM-L6-v2")
    ap.add_argument("--hybrid-weight", type=float, default=0.5,
                    help="weight on the second embedder (0..1)")
    ap.add_argument("--max-changed", type=int, default=2,
                    help="largest number of fields a proposed fix may touch")


def baseline_from(args) -> RetrievalConfig:
    return RetrievalConfig(
        chunker=args.chunker, chunk_size=args.chunk_size, overlap=args.overlap,
        embedder=args.embedder, k=args.k, candidate_k=args.candidate_k,
        context_budget=args.context_budget,
        hybrid_with=args.hybrid_with, hybrid_weight=args.hybrid_weight,
    )


def setup(args):
    corpus = Corpus.from_jsonl(args.corpus)
    queries = load_queries(args.queries)
    cache = EmbeddingCache(args.cache)
    store = IndexStore(corpus, cache)
    return corpus, queries, cache, store


def cmd_triage(args) -> int:
    corpus, queries, cache, store = setup(args)
    cfg = baseline_from(args)
    retr = store.get(cfg)

    print(f"corpus: {len(corpus)} docs, {len(retr.chunks)} chunks")
    print(f"config: {cfg.describe()}  candidate_k={cfg.candidate_k}")
    print(RULE)

    results = triage_all(retr, queries)
    for t in results:
        rank = f"rank {t.best_rank}" if t.best_rank else "-"
        flag = "ok  " if t.ok else "MISS"
        print(f"{flag} {t.query.id:<5} {t.verdict.value:<24} {rank:<9} "
              f"{t.query.text[:34]}")

    s = summarize(results)
    print(RULE)
    print(f"recall@{cfg.k}: {s['recall_at_k']}%  "
          f"({s['n_retrieved']}/{s['n_queries']})")
    for verdict, n in s["verdicts"].items():
        print(f"  {verdict:<24} {n}")
    print(f"\nembedding cache: {cache.stats}")
    return 0


def cmd_explain(args) -> int:
    corpus, queries, cache, store = setup(args)
    cfg = baseline_from(args)
    retr = store.get(cfg)

    by_id = {q.id: q for q in queries}
    if args.query_id not in by_id:
        print(f"unknown query id {args.query_id!r}; have: "
              f"{', '.join(sorted(by_id))}", file=sys.stderr)
        return 2
    q = by_id[args.query_id]

    print(f"query {q.id}: {q.text}")
    print(f"expected: {q.gold_doc}"
          + (f"  span: {q.gold_span[:60]!r}" if q.gold_span else ""))
    print(f"config: {cfg.describe()}")
    print(RULE)

    t = triage(retr, q)
    print(f"STAGE 1  verdict: {t.verdict.value}")
    print(f"         blame:   {t.verdict.blame}")
    for n in t.notes:
        print(f"         note:    {n}")

    loc = localize(retr, q)
    print(f"\nSTAGE 2  {loc.summary()}")
    for n in loc.notes:
        print(f"         note:    {n}")
    frag = fragmentation_report(retr, q)
    if frag:
        print(f"         note:    {frag}")

    if t.top_results:
        print("\n         what came back instead:")
        for r in t.top_results[:5]:
            mark = "<-- gold" if r.chunk.doc_id == q.gold_doc else ""
            head = " ".join(r.chunk.text.split())[:46]
            print(f"           {r.rank}. {r.score:.3f} {r.chunk.id:<14} "
                  f"{head} {mark}")

    if t.ok:
        print("\nSTAGE 3  nothing to fix.")
        return 0

    print("\nSTAGE 3  searching for the minimal fix...")
    t0 = time.perf_counter()
    res = search_fix(store, cfg, q, queries=queries,
                     max_changed_fields=args.max_changed)
    dt = time.perf_counter() - t0
    fix = res.fix

    if res.skipped_degenerate:
        print(f"         refused {res.skipped_degenerate} degenerate config(s) "
              f"that would return over "
              f"{int(DEGENERATE_K_FRACTION * 100)}% of the corpus")

    if fix is None:
        print(f"         no config within {args.max_changed} changed field(s) "
              f"retrieves this. Try --max-changed 3, a different embedder, or "
              f"accept that this query needs a better retriever.")
    else:
        print(f"         minimal fix: {fix.describe()}")
        print(f"         resulting config: {fix.config.describe()}")
        print(f"         fields changed: {fix.distance[0]}, "
              f"re-embedding required: {fix.requires_reembedding}")
        print(f"         configs evaluated: {fix.configs_tried} in {dt:.2f}s")
        if fix.regressions:
            print(f"         WARNING regressions: {', '.join(fix.regressions)}")
        if fix.improvements:
            print(f"         also fixes: {', '.join(fix.improvements)}")
        print(f"         net queries gained: {fix.net:+d}"
              + ("  (safe)" if fix.safe else "  (NOT safe)"))

    print(f"\nembedding cache: {cache.stats}")
    return 0


def cmd_headroom(args) -> int:
    corpus, queries, cache, store = setup(args)
    cfg = baseline_from(args)

    print(f"sweeping config space (max {args.max_changed} changed fields)...")
    t0 = time.perf_counter()
    h = headroom(store, cfg, queries, max_changed_fields=args.max_changed)
    dt = time.perf_counter() - t0

    print(RULE)
    print(f"queries:              {h.n_queries}")
    print(f"baseline recall:      {h.baseline_recall}%   ({cfg.describe()})")
    print(f"best single config:   {h.best_single_recall}%   "
          f"({h.best_single_config.describe() if h.best_single_config else '-'})")
    print(f"oracle (any config):  {h.oracle_recall}%")
    print(f"headroom:             {h.gap:+.1f} points")
    print(f"\n{h.verdict}")
    if h.unfixable:
        print(f"\nno config retrieves: {', '.join(h.unfixable)}")
        print("these are genuine retriever limits, not tuning problems")
    print(f"\nswept in {dt:.2f}s; indexes built: {store.builds}; "
          f"embedding cache: {cache.stats}")
    return 0


def cmd_bench(args) -> int:
    """Measure what the content-addressed cache actually buys.

    Warmup is not optional here. The first run to touch an embedder pays a
    one-time `import sklearn`, which profiling showed to be ~3.2s - an order
    of magnitude more than the work being measured. Without the warmup below
    this benchmark reported the cold-cache run as 17x *slower* than no cache
    at all, purely because it ran first and absorbed the import.
    """
    corpus = Corpus.from_jsonl(args.corpus)
    queries = load_queries(args.queries)
    cfg = baseline_from(args)
    from .config import enumerate_space
    from .embed import get_embedder

    configs = enumerate_space(cfg, max_changed_fields=args.max_changed)
    print(f"corpus: {len(corpus)} docs | configs in sweep: {len(configs)} | "
          f"queries: {len(queries)}")

    t0 = time.perf_counter()
    get_embedder(cfg.embedder).encode(["warmup"])
    print(f"warmup (one-time embedder import + init): "
          f"{time.perf_counter() - t0:.2f}s - excluded from the rows below")
    print(RULE)

    rows = []
    with TemporaryDirectory() as tmp:
        for label, path in (("cold cache", Path(tmp) / "cold.sqlite"),
                            ("warm cache", Path(tmp) / "cold.sqlite")):
            cache = EmbeddingCache(path)
            cache.reset_stats()
            store = IndexStore(corpus, cache)
            t0 = time.perf_counter()
            for c in configs:
                r = store.get(c)
                for q in queries:
                    triage(r, q)
            dt = time.perf_counter() - t0
            rows.append((label, dt, cache.stats.hits, cache.stats.misses,
                         cache.stats.hit_rate, store.builds))
            cache.close()

        # And a no-cache control: every config re-embeds from scratch.
        t0 = time.perf_counter()
        total_embedded = 0
        for c in configs:
            cache = EmbeddingCache(None)
            store = IndexStore(corpus, cache)
            r = store.get(c)
            for q in queries:
                triage(r, q)
            total_embedded += cache.stats.misses
        dt_nocache = time.perf_counter() - t0

    print(f"{'run':<14}{'seconds':>9}{'reused':>9}{'embedded':>10}"
          f"{'reuse':>8}{'builds':>8}")
    for label, dt, hits, misses, rate, builds in rows:
        print(f"{label:<14}{dt:>9.2f}{hits:>9}{misses:>10}{rate:>7.1f}%{builds:>8}")
    print(f"{'no cache':<14}{dt_nocache:>9.2f}{0:>9}{total_embedded:>10}"
          f"{0.0:>7.1f}%{len(configs):>8}")

    best = rows[1][1]
    if best > 0:
        print(f"\ncold vs no-cache: {dt_nocache / rows[0][1]:.1f}x faster")
        print(f"warm vs no-cache: {dt_nocache / best:.1f}x faster")
    print(f"texts embedded without cache: {total_embedded}; "
          f"with cache (cold): {rows[0][3]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whiff",
                                 description="debug why retrieval missed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("triage", help="classify every query's failure mode")
    add_common(p)
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("explain", help="full diagnosis for one query")
    p.add_argument("query_id")
    add_common(p)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("headroom", help="oracle bound over the config space")
    add_common(p)
    p.set_defaults(fn=cmd_headroom)

    p = sub.add_parser("bench", help="measure the embedding cache speedup")
    add_common(p)
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
