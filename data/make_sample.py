"""Generate the sample corpus and queries.

The sample data is engineered, not scraped: each query is built to trigger a
specific failure mode at the default config (fixed/150+0, hash, k=5,
candidate_k=20) so that `whiff triage` demonstrates every verdict and
`whiff explain` has something real to diagnose.

Run:  python data/make_sample.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

FILLER = (
    "The operations handbook records process changes for each quarter and is "
    "reviewed by the regional leads before publication. Historical revisions "
    "remain available in the archive for audit purposes. Teams should consult "
    "their lead before deviating from documented procedure. "
)


def pad(text: str, target: int) -> str:
    """Pad with filler to at least `target` characters."""
    out = text
    while len(out) < target:
        out += FILLER
    return out


docs: dict[str, str] = {}

# --- q1: a plain hit. The answer sits early, well inside the first chunk,
# so no boundary cuts it at any chunk size. --------------------------------
docs["handbook-returns"] = (
    "Returns policy. The returns window for damaged freight is fourteen days "
    "from delivery. A damaged pallet must also be photographed on arrival "
    "and reported within 24 hours. " + FILLER
)

# --- q2: fragmentation. The answer is a 220-char sentence placed to straddle
# a 150-char boundary but sit inside a single 300-char chunk. ---------------
_prefix = pad("Cold chain procedures. ", 310)[:310]
_span_q2 = (
    "Refrigerated containers must be pre-cooled to minus eighteen degrees "
    "celsius for a minimum of six hours before loading, and the temperature "
    "log must be signed by both the loading supervisor and the duty manager."
)
docs["handbook-coldchain"] = _prefix + _span_q2 + " " + FILLER

# --- q3: below cutoff. The gold sentence is diluted by surrounding filler,
# while short distractor notices are dense with the query's exact terms, so
# the gold chunk lands just outside k=5 but inside k=10. -------------------
docs["handbook-forklift"] = (
    pad("Depot induction notes, section four. ", 200)[:200]
    + "Operators renew forklift certification every three years. "
    + pad(" The record is held by the depot safety officer. ", 400)
)
for depot in ["ashford", "bristol", "cardiff", "dundee", "exeter",
              "falkirk", "glasgow"]:
    docs[f"depot-{depot}-safety"] = (
        f"{depot.title()} depot notice. Forklift certification renewal. "
        f"How often is forklift certification renewed? Forklift "
        f"certification renewal queries go to the safety officer."
    )

# --- q4: absent from candidates. The answer uses vocabulary that does not
# overlap the question, which a lexical embedder cannot bridge. -------------
# Every content word differs from the question's: no lexical bridge exists,
# so this one is genuinely unfixable by chunking or k. It is here to prove
# the tool can say "stop tuning, this is a retriever limit".
docs["handbook-remuneration"] = pad(
    "Remuneration schedule. Emolument for duties discharged beyond the "
    "rostered span is compensated at one and one half of the standard "
    "hourly tariff, rising to double on statutory holidays. ", 900)

# --- q5: dilution. The answer is one sentence in a long document. ----------
docs["handbook-operations-full"] = (
    pad("Operations handbook, consolidated edition. ", 1500)
    + "The night shift differential is eleven percent of base pay. "
    + pad(" Further sections cover routing, fuel cards and depot access. ", 1600)
)

# --- distractors, to give the candidate pool something to fill up with -----
for topic in ["routing", "fuelcards", "uniforms", "lockers", "parking",
              "canteen", "lostproperty"]:
    docs[f"notice-{topic}"] = pad(
        f"{topic.title()} notice. This notice describes the {topic} process "
        f"and who to contact about {topic} issues at each depot. ", 420)

# --- q7: packing. A long gold chunk that is retrieved but will not fit a
# small context budget. ----------------------------------------------------
docs["handbook-keys"] = pad(
    "Key custody. Depot gate keys are signed out from the duty office and "
    "must be returned before the end of shift. ", 600)


queries = [
    {"id": "q1", "text": "what is the returns window for damaged freight",
     "gold_doc": "handbook-returns",
     "gold_span": "fourteen days from delivery"},

    {"id": "q2", "text": "how long must refrigerated containers be pre-cooled "
                         "before loading",
     "gold_doc": "handbook-coldchain",
     "gold_span": _span_q2},

    {"id": "q3", "text": "how often is forklift certification renewed",
     "gold_doc": "handbook-forklift",
     "gold_span": "renew forklift certification every three years"},

    {"id": "q4", "text": "what is the overtime pay rate",
     "gold_doc": "handbook-remuneration",
     "gold_span": "one and one half of the standard hourly tariff"},

    {"id": "q5", "text": "what is the night shift differential",
     "gold_doc": "handbook-operations-full",
     "gold_span": "eleven percent of base pay"},

    {"id": "q6", "text": "what is the mileage reimbursement rate",
     "gold_doc": "policy-travel-2024",
     "gold_span": "45p per mile"},

    {"id": "q7", "text": "where are depot gate keys signed out",
     "gold_doc": "handbook-keys",
     "gold_span": "signed out from the duty office"},
]


def main() -> None:
    corpus_path = HERE / "sample_corpus.jsonl"
    queries_path = HERE / "sample_queries.jsonl"

    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc_id in sorted(docs):
            fh.write(json.dumps({"id": doc_id, "text": docs[doc_id]}) + "\n")

    with queries_path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")

    total = sum(len(t) for t in docs.values())
    print(f"wrote {len(docs)} docs ({total} chars) -> {corpus_path.name}")
    print(f"wrote {len(queries)} queries -> {queries_path.name}")
    print(f"at chunk_size=150 that is about {total // 150} chunks")


if __name__ == "__main__":
    main()
