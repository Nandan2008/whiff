"""Convert SQuAD v1.1/v2.0 into whiff's corpus + queries format.

SQuAD is a good first real dataset for this tool because every question
already carries an exact answer span and the paragraph it came from - which
is precisely the (gold_doc, gold_span) pair whiff needs, with no labelling
work.

Each SQuAD paragraph becomes one document. The answer text becomes the gold
span, so a retrieval "hit" requires a chunk that actually contains the
answer, not merely the right paragraph.

Usage
-----
Download once (about 30 MB), then convert:

    curl -L -o data/squad-dev.json \\
      https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

    python data/load_squad.py data/squad-dev.json --max-queries 200

Then point whiff at the result:

    python -m whiff.cli triage \\
      --corpus data/squad_corpus.jsonl --queries data/squad_queries.jsonl \\
      --chunk-size 300 --k 5 --candidate-k 50

Note on scale: a few hundred queries against a few thousand paragraphs is
the right size for a first run. The `headroom` sweep is quadratic in
(configs x queries), so start small and grow once you know the runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def convert(src: Path, max_queries: int | None, max_docs: int | None,
            skip_unanswerable: bool = True) -> tuple[dict, list]:
    raw = json.loads(src.read_text(encoding="utf-8"))
    if "data" not in raw:
        raise SystemExit(f"{src} does not look like SQuAD (no 'data' key)")

    docs: dict[str, str] = {}
    queries: list[dict] = []
    seen_spans: set[tuple[str, str]] = set()

    for article in raw["data"]:
        title = str(article.get("title", "untitled"))
        for p_i, para in enumerate(article.get("paragraphs", [])):
            context = para.get("context", "")
            if not context.strip():
                continue
            doc_id = f"{title}#{p_i}".replace(" ", "_")

            if max_docs is not None and doc_id not in docs and len(docs) >= max_docs:
                continue
            docs[doc_id] = context

            for qa in para.get("qas", []):
                if max_queries is not None and len(queries) >= max_queries:
                    continue
                if skip_unanswerable and qa.get("is_impossible"):
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    continue
                span = str(answers[0].get("text", "")).strip()
                # Very short spans ("1998", "he") match almost any chunk and
                # make the gold-span check meaningless, so require some length.
                if len(span) < 8:
                    continue
                if span not in context:
                    continue  # malformed record; drop rather than mislabel
                key = (doc_id, span)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                # Sequential ids, NOT SQuAD's own. Truncating SQuAD ids to a
                # readable width collides (several share a 16-char prefix),
                # which makes `explain <id>` ambiguous and silently breaks
                # regression checking, since that matches queries by id.
                queries.append({
                    "id": f"q{len(queries):04d}",
                    "text": str(qa.get("question", "")).strip(),
                    "gold_doc": doc_id,
                    "gold_span": span,
                    "squad_id": str(qa.get("id", "")),
                })

    # Keep only documents that are actually referenced or needed as
    # distractors; every paragraph is a legitimate distractor, so keep all.
    return docs, queries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SQuAD -> whiff JSONL")
    ap.add_argument("src", help="path to dev-v1.1.json or train-v2.0.json")
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--out-prefix", default="squad")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        print(f"not found: {src}\n\nDownload it first:\n"
              f"  curl -L -o {src} https://rajpurkar.github.io/"
              f"SQuAD-explorer/dataset/dev-v1.1.json", file=sys.stderr)
        return 2

    docs, queries = convert(src, args.max_queries, args.max_docs)
    if not queries:
        print("no usable queries extracted", file=sys.stderr)
        return 1

    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        raise SystemExit("query ids are not unique - refusing to write")

    corpus_path = HERE / f"{args.out_prefix}_corpus.jsonl"
    queries_path = HERE / f"{args.out_prefix}_queries.jsonl"

    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc_id in sorted(docs):
            fh.write(json.dumps({"id": doc_id, "text": docs[doc_id]}) + "\n")
    with queries_path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")

    chars = sum(len(t) for t in docs.values())
    print(f"wrote {len(docs)} docs ({chars:,} chars) -> {corpus_path.name}")
    print(f"wrote {len(queries)} queries -> {queries_path.name}")
    print(f"at chunk_size=300 that is roughly {chars // 300:,} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
