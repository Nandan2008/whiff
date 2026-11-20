"""The study: which single configuration change fixes each failure?

This answers the question the README poses - of the retrieval failures on a
real dataset, what fraction are fixable by one change, and which knob does
the work? Reported per original verdict, because the useful claim is not
"chunking matters" but "chunking fixes fragmentation and does nothing for
vocabulary mismatch".

Usage
-----
    python scripts/finding.py --limit 80
    python scripts/finding.py --limit 80 --chunking-only

Degenerate configs (those returning over 25% of the corpus) are excluded by
the search, so "fixable" here means fixable without cheating.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from whiff.config import RetrievalConfig  # noqa: E402
from whiff.counterfactual import search_fix  # noqa: E402
from whiff.embed import EmbeddingCache, get_embedder  # noqa: E402
from whiff.pipeline import Corpus, IndexStore, load_queries  # noqa: E402
from whiff.triage import triage  # noqa: E402

CHUNKING_SPACE = {
    "chunker": ["fixed", "recursive", "sentence"],
    "chunk_size": [150, 300, 600, 1200],
    "overlap": [0, 75],
}
FULL_SPACE = dict(CHUNKING_SPACE, k=[5, 10, 20])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "squad_corpus.jsonl"))
    ap.add_argument("--queries", default=str(ROOT / "data" / "squad_queries.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / ".squad-cache.sqlite"))
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("-k", "--top-k", dest="k", type=int, default=5)
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--chunking-only", action="store_true",
                    help="restrict the space to chunking knobs (no k)")
    ap.add_argument("--embedder", default="hash",
                    help="hash (lexical, offline) or st:<model-name> (dense)")
    args = ap.parse_args()

    corpus = Corpus.from_jsonl(args.corpus)
    queries = load_queries(args.queries)[: args.limit]
    cache = EmbeddingCache(args.cache)
    store = IndexStore(corpus, cache)
    space = CHUNKING_SPACE if args.chunking_only else FULL_SPACE

    base = RetrievalConfig(chunk_size=args.chunk_size, k=args.k,
                           candidate_k=args.candidate_k,
                           embedder=args.embedder)

    get_embedder(base.embedder).encode(["warmup"])   # keep timing honest

    print(f"corpus: {len(corpus)} docs | queries: {len(queries)}")
    print(f"baseline: {base.describe()} candidate_k={base.candidate_k}")
    print(f"space: {'chunking only' if args.chunking_only else 'chunking + k'}"
          f" | single-change fixes only")
    print("-" * 72)

    t0 = time.perf_counter()
    retr = store.get(base)
    failures = [(q, triage(retr, q)) for q in queries]
    failures = [(q, t) for q, t in failures if not t.ok]
    print(f"baseline failures: {len(failures)} of {len(queries)}")

    fixed_by: Counter[str] = Counter()
    by_verdict: dict[str, Counter[str]] = defaultdict(Counter)
    unfixable: list[str] = []
    evaluated = 0
    refused = 0

    for i, (q, t) in enumerate(failures, 1):
        res = search_fix(store, base, q, space=space, max_changed_fields=1)
        evaluated += res.configs_evaluated
        refused += res.skipped_degenerate
        if res.fix is None:
            fixed_by["(none)"] += 1
            by_verdict[t.verdict.value]["(none)"] += 1
            unfixable.append(q.id)
        else:
            field = next(iter(res.fix.changed))
            fixed_by[field] += 1
            by_verdict[t.verdict.value][field] += 1
        if i % 10 == 0:
            print(f"  ...{i}/{len(failures)} "
                  f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    dt = time.perf_counter() - t0
    n = len(failures) or 1
    fixable = n - fixed_by["(none)"]

    print("-" * 72)
    print(f"FIXABLE BY ONE CHANGE: {fixable}/{n} "
          f"({100.0 * fixable / n:.1f}%)")
    print("\nwhich knob did the work:")
    for field, count in fixed_by.most_common():
        print(f"  {field:<16} {count:>4}  ({100.0 * count / n:>5.1f}%)")

    print("\nby original verdict:")
    for verdict in sorted(by_verdict):
        total = sum(by_verdict[verdict].values())
        inner = ", ".join(f"{f}={c}" for f, c in by_verdict[verdict].most_common())
        print(f"  {verdict:<24} n={total:<4} {inner}")

    print(f"\nconfigs evaluated: {evaluated}; degenerate refused: {refused}")
    print(f"elapsed: {dt:.1f}s; indexes built: {store.builds}; "
          f"cache: {cache.stats}")
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
