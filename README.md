# whiff

**Why didn't my retriever find the obvious document?**

Give it your corpus, a query, and the document you expected back. It tells
you which *kind* of failure this is, where the answer actually lives, and the
smallest configuration change that would have retrieved it.

Runs fully offline with no API key, no model download and no network.

```
$ python -m whiff.cli explain q2

query q2: how long must refrigerated containers be pre-cooled before loading
expected: handbook-coldchain
config: fixed/150+0 hash k=5
------------------------------------------------------------------------
STAGE 1  verdict: span_fragmented
         blame:   chunking: no single chunk contains the answer
STAGE 2  note:    the answer straddles the boundary between chunks 3 and 4
                  (at character 450 of the document); overlap would fix this
STAGE 3  minimal fix: chunk_size: 150 -> 300
         fields changed: 1, re-embedding required: True
         configs evaluated: 4 in 0.02s
         net queries gained: +0  (safe)
```

> **Status: working.** All four stages run end to end, 76 tests pass, and the
> tool has been run against SQuAD v1.1 dev with both a lexical and a dense
> embedder — see *Finding* below.

## Why this exists

Evaluation frameworks score a pipeline — they tell you you're at 0.62. They
don't explain one query. When the answer is sitting in a document and your
retriever returns three irrelevant chunks, you're left changing three things
at once and hoping.

The four stages are deliberately separate because **these failures have
completely different fixes**, and applying one bucket's fix to another
bucket's problem is the most common way to waste a day:

| Verdict | Blame |
|---|---|
| `not_indexed` | ingestion — the document isn't in the corpus |
| `span_fragmented` | chunking — no single chunk contains the answer |
| `absent_from_candidates` | embedding or index — gold never reached the pool |
| `below_cutoff` | ranking or `k` — gold was a candidate, ranked too low |
| `dropped_in_packing` | context assembly — retrieved, then discarded |

## Usage

```bash
pip install -r requirements.txt
python data/make_sample.py            # generate the sample corpus

python -m whiff.cli triage            # classify every query
python -m whiff.cli explain q3        # full diagnosis for one query
python -m whiff.cli headroom          # oracle bound over the config space
python -m whiff.cli bench             # measure the embedding cache speedup
```

Point it at your own data with `--corpus` and `--queries` (JSONL; see
`data/`). For real dense embeddings, `--embedder st:all-MiniLM-L6-v2`.

## The three ideas worth knowing

**1. Minimality is defined, not guessed.** A fix's distance from the baseline
is `(fields changed, cost)`, where a change forcing re-embedding costs more
than one that only re-ranks existing vectors. Candidates are evaluated in
non-decreasing order of that key, so **the first config that works is a
provable minimum** — no search heuristics. In practice `below_cutoff` fixes
are found in one evaluation.

**2. The optimizer will cheat, so it's fenced.** At `chunk_size=600` the
sample corpus has ~22 chunks, so `k=20` "retrieves" nearly everything —
recall 100%, precision junk. Configs returning more than 25% of the corpus
are excluded (`DEGENERATE_K_FRACTION`). Turning the guard on dropped the
sample's oracle from 85.7% to 71.4% and correctly reclassified one query from
"fixable" to a genuine retriever limit. **A diagnosis that flatters your
pipeline is worse than none.**

**3. Two layers of reuse make the sweep cheap.** `IndexStore` shares a built
index across configs differing only in `k`; `EmbeddingCache` keys vectors on
`sha256(embedder_id + chunk_text)` so each distinct string is embedded once
ever, across configs and across runs. This is also why the default embedder
is *stateless* — a fitted TF-IDF vectorizer would produce different vectors
per config and make content-addressing unsound.

Measured on the sample corpus (`python -m whiff.cli bench`), 30 configs × 7
queries:

| run | seconds | texts embedded | reuse | indexes built |
|---|---|---|---|---|
| no cache | 0.12 | 1621 | 0% | 30 |
| cold cache | 0.07 | 353 | 77.2% | 16 |
| warm cache | 0.04 | 0 | 100% | 16 |

1.8x faster cold, 2.9x warm, and **4.6x fewer embeddings**. The gap widens
enormously with a real model: the `hash` embedder costs microseconds per
text, a sentence-transformer costs milliseconds, so the cache's value scales
directly with embedding cost.

### What the benchmark taught me

The first version of that table was wrong, and interestingly wrong. It
reported the cold cache at **2.05s — 17x slower than no cache at all.** The
obvious culprit was SQLite's per-commit `fsync`, so I added `journal_mode=WAL`
and `synchronous=NORMAL`. That bought 2.05s → 1.95s. Within noise.

Profiling instead of guessing showed the actual cost: **3.25 of 3.76 seconds
was a one-time `import sklearn`**, paid lazily by whichever run touched an
embedder first. The benchmark had no warmup, so the first row absorbed the
import and the last row ran against a hot interpreter. Nothing to do with
caching, SQLite, or fsync.

Adding an explicit warmup took the cold row from 2.05s to 0.07s. The pragmas
are still there — the durability trade is right for a recomputable cache —
but they are documented as *not* having been the fix, because a comment
claiming a measurement it never made is worse than no comment.

## Finding: what fraction of retrieval failures are fixable by tuning?

Run on **SQuAD v1.1 dev** — 2,067 paragraphs, 80 questions, each with an
exact answer span — at a baseline of `fixed/300+0`, `hash` embedder, `k=5`,
`candidate_k=50`. Reproduce with:

```bash
curl -L -o /tmp/dev.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
python data/load_squad.py /tmp/dev.json --max-queries 300
python scripts/finding.py --limit 80
python scripts/finding.py --limit 80 --chunking-only
```

At the baseline, **48 of 80 queries fail**. Of those 48:

| Single change | Fixes | Share of failures |
|---|---|---|
| `k` | 11 | 22.9% |
| `chunk_size` | 6 | 12.5% |
| `overlap` | 2 | 4.2% |
| `chunker` | 1 | 2.1% |
| **nothing in the space** | **28** | **58.3%** |

**41.7% of failures are fixable by one change; 33.3% by a chunking change
alone** (chunking-only run: `chunk_size` 13, `overlap` 2, `chunker` 1).

The breakdown by verdict is the part worth quoting, because it says *which*
fix belongs to *which* failure:

| Verdict | n | Outcome |
|---|---|---|
| `span_fragmented` | 2 | **100% fixed by `overlap`** |
| `below_cutoff` | 29 | `k` fixes 11, `chunk_size` 4, 14 unfixable |
| `absent_from_candidates` | 17 | **14 of 17 unfixable**; chunking fixed 3 |

So the honest claim is not "chunking matters" but: **chunking fixes
fragmentation almost perfectly and does nearly nothing for vocabulary
mismatch.** Raising `k` rescues only ~38% of ranking failures. And 58% of
failures aren't a tuning problem at all — they need a better retriever,
which is exactly what `headroom` exists to tell you before you spend a day
on parameters.

One more observation that validates the minimality ordering: with `k`
available it fixes 11 `below_cutoff` queries and `chunk_size` only 4;
remove `k` from the space and `chunk_size` fixes 11 of the same queries.
Many failures have several valid fixes, and the ordering correctly prefers
the one that needs no re-embedding.

### Lexical vs dense: the caveat, tested

The obvious objection to the table above is that the `hash` embedder is
lexical, so the unfixable bucket is really just vocabulary mismatch a dense
model would solve. So I ran both on an identical slice — 300 paragraphs, 120
questions, same baseline:

```bash
python data/load_squad.py /tmp/dev.json --max-queries 120 --max-docs 300 \
  --out-prefix squad_small
python scripts/finding.py --corpus data/squad_small_corpus.jsonl \
  --queries data/squad_small_queries.jsonl --limit 120
python scripts/finding.py --corpus data/squad_small_corpus.jsonl \
  --queries data/squad_small_queries.jsonl --limit 120 \
  --embedder st:all-MiniLM-L6-v2
```

| | lexical (`hash`) | dense (`all-MiniLM-L6-v2`) |
|---|---|---|
| baseline recall@5 | 48.3% (58/120) | **75.0%** (90/120) |
| failures | 62 | 30 |
| fixable by one change | 31 (50.0%) | **30 (100%)** |
| unfixable | 31 | **0** |

And the composition of the failures:

| Verdict | lexical | dense |
|---|---|---|
| `absent_from_candidates` | 16 | **3** |
| `below_cutoff` | 43 | 24 |
| `span_fragmented` | **3** | **3** |

Three things fall out of this, and the third is the one I did not expect:

1. **The objection was right.** `absent_from_candidates` collapses from 16 to
   3 with a dense model. Most of what looked like an untunable retrieval
   limit was vocabulary mismatch.
2. **With a competent retriever, tuning actually works.** Every one of the 30
   dense failures is fixable by a single change — against half for lexical.
   The unfixable bucket is a property of a weak embedder, not of retrieval.
3. **`span_fragmented` is identical in both: 3 and 3.** Chunking happens
   *before* embedding, so a chunker that cuts through an answer cuts through
   it no matter how you vectorise. Fragmentation is a structural failure,
   invariant to the retriever — which is exactly why it deserves its own
   verdict rather than being folded into a recall number.

So the defensible claim is: **chunking failures and semantic failures are
independent, and only the latter respond to a better embedder.** Quote the
dense numbers, not the lexical ones, as the realistic picture.

## Correctness notes

- Chunk offsets satisfy `chunk.text == document[start:end]`. Localization
  asks "which chunk contains the gold span", and an off-by-a-few offset turns
  a correct diagnosis into a plausible wrong one.
- A chunk from the right document that *omits* the answer is not a hit.
  Conflating the two is how RAG evaluation flatters itself.
- Gold-span matching is whitespace- and case-insensitive, because a span
  typed into a JSON file rarely matches the document's whitespace exactly.
- The index is exact brute force, not approximate. A debugger that blamed
  your chunking when its own ANN recall was the culprit would be worse than
  useless. `BruteForceIndex.search` is the only interface the pipeline
  depends on, so a hand-written ANN can be dropped in behind it.

## Limitations

- The default `hash` embedder is **lexical** and materially changes the
  conclusions — see the lexical/dense comparison above, where it doubles the
  failure count and invents an unfixable bucket that does not exist under a
  dense model. It is the default so the tool runs instantly with no download;
  use `st:<model>` for anything you intend to quote.
- With 2048 hash features, collisions can create spurious similarity.
- The search space is four axes by design (`chunker`, `chunk_size`,
  `overlap`, `k`). A space you can enumerate beats one you must sample.
- Sample data is **engineered**, not scraped: each query is built to trigger
  a specific verdict so the stages have something real to diagnose.

## Tests

```bash
python -m pytest -q      # 76 tests, ~5s, no network
```

The invariants worth knowing are pinned: chunk offsets equal their source
slice across every chunker and size, overlap equal to chunk size terminates
instead of hanging, cache hit/miss accounting including dedup within a single
call and persistence across instances, each of the six verdict boundaries in
isolation, that the space really is ordered so the first hit is minimal, that
the degenerate guard refuses and counts, and that a chunk-shrinking fix which
breaks another query is reported as a regression.

## What's left

- **More of the dataset.** The runs above use 80–120 questions against
  300–2,067 paragraphs. Nothing blocks a larger slice except patience: the
  sweep is linear in configs × queries and the cache makes repeat runs free.
- **A reranker axis.** The space is four knobs; a cross-encoder rerank
  on/off would likely absorb a chunk of the `below_cutoff` bucket and is the
  most obviously missing dimension.
- **Hybrid lexical+dense weighting** as a fifth axis, which the two runs
  above suggest would be the most informative one to add.
- **A hand-written ANN index** behind `BruteForceIndex.search`, if you want
  to go deeper on algorithms than this project currently does.
- Minor: `--max-docs` in the SQuAD loader keeps the first N paragraphs rather
  than sampling, so a small slice is topically clustered.

## Layout

```
whiff/config.py          RetrievalConfig, the config space, minimality ordering
whiff/chunking.py        three chunkers, offsets guaranteed correct
whiff/embed.py           embedders + content-addressed cache
whiff/index.py           exact brute-force index
whiff/pipeline.py        corpus, queries, IndexStore, Retriever
whiff/triage.py          stage 1: which kind of failure
whiff/localize.py        stage 2: where the answer lives, dilution check
whiff/counterfactual.py  stage 3 and 4: minimal fix, regressions, headroom
whiff/cli.py             triage | explain | headroom | bench
data/make_sample.py      regenerates the engineered sample corpus
data/load_squad.py       SQuAD -> whiff JSONL
scripts/finding.py       the single-change fix study
tests/                   76 tests
```

## License

MIT.
