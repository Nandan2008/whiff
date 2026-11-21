"""A local web interface for whiff.

Standard library only. The tool's whole pitch is that it runs offline with no
API key and almost no dependencies, so adding a web framework to show its
output would be a bad trade.

The JSON payload builders are plain functions with no HTTP in them, so they
can be tested directly without binding a port.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import RetrievalConfig
from .counterfactual import search_fix
from .embed import EmbeddingCache
from .localize import fragmentation_report, localize
from .pipeline import Corpus, IndexStore, Query, load_queries
from .triage import summarize, triage, triage_all

STATIC = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------
# config from query string
# --------------------------------------------------------------------------

def config_from_params(params: dict[str, list[str]]) -> RetrievalConfig:
    def one(name, cast, default):
        raw = params.get(name, [None])[0]
        if raw is None or raw == "":
            return default
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return default

    budget = one("context_budget", int, None)
    hybrid = params.get("hybrid_with", [""])[0] or None

    return RetrievalConfig(
        chunker=one("chunker", str, "fixed"),
        chunk_size=one("chunk_size", int, 150),
        overlap=one("overlap", int, 0),
        embedder=one("embedder", str, "hash"),
        k=one("k", int, 5),
        candidate_k=one("candidate_k", int, 20),
        context_budget=budget if budget and budget > 0 else None,
        hybrid_with=hybrid,
        hybrid_weight=one("hybrid_weight", float, 0.5),
    )


def config_dict(cfg: RetrievalConfig) -> dict:
    return {
        "chunker": cfg.chunker, "chunk_size": cfg.chunk_size,
        "overlap": cfg.overlap, "embedder": cfg.embedder, "k": cfg.k,
        "candidate_k": cfg.candidate_k, "context_budget": cfg.context_budget,
        "hybrid_with": cfg.hybrid_with, "hybrid_weight": cfg.hybrid_weight,
        "describe": cfg.describe(),
    }


# --------------------------------------------------------------------------
# payload builders - no HTTP in here, so they are directly testable
# --------------------------------------------------------------------------

def overview_payload(store: IndexStore, cfg: RetrievalConfig,
                     queries: list[Query]) -> dict:
    retr = store.get(cfg)
    results = triage_all(retr, queries)
    rows = []
    for t in results:
        rows.append({
            "id": t.query.id,
            "text": t.query.text,
            "gold_doc": t.query.gold_doc,
            "verdict": t.verdict.value,
            "blame": t.verdict.blame,
            "ok": t.ok,
            "rank": t.best_rank,
        })
    s = summarize(results)
    return {
        "config": config_dict(cfg),
        "corpus": {"docs": len(retr.corpus.docs), "chunks": len(retr.chunks)},
        "summary": s,
        "queries": rows,
    }


def explain_payload(store: IndexStore, cfg: RetrievalConfig,
                    queries: list[Query], query_id: str,
                    max_changed: int = 2) -> dict:
    by_id = {q.id: q for q in queries}
    if query_id not in by_id:
        return {"error": f"unknown query id: {query_id}"}
    q = by_id[query_id]
    retr = store.get(cfg)

    t = triage(retr, q)
    loc = localize(retr, q)
    gold_ids = {c.id for c in retr.gold_chunks(q)}

    results = []
    for r in retr.candidates(q.text)[: max(cfg.k, 8)]:
        results.append({
            "rank": r.rank,
            "score": round(r.score, 4),
            "chunk_id": r.chunk.id,
            "doc_id": r.chunk.doc_id,
            "is_gold": r.chunk.id in gold_ids,
            "in_gold_doc": r.chunk.doc_id == q.gold_doc,
            "text": " ".join(r.chunk.text.split())[:240],
        })

    fix_payload = None
    if not t.ok:
        res = search_fix(store, cfg, q, queries=queries,
                         max_changed_fields=max_changed)
        if res.fix is not None:
            f = res.fix
            fix_payload = {
                "found": True,
                "changes": [{"field": k, "from": str(a), "to": str(b)}
                            for k, (a, b) in sorted(f.changed.items())],
                "describe": f.describe(),
                "config": f.config.describe(),
                "fields_changed": f.distance[0],
                "requires_reembedding": f.requires_reembedding,
                "regressions": f.regressions,
                "improvements": f.improvements,
                "net": f.net,
                "safe": f.safe,
                "configs_evaluated": res.configs_evaluated,
                "skipped_degenerate": res.skipped_degenerate,
            }
        else:
            fix_payload = {
                "found": False,
                "configs_evaluated": res.configs_evaluated,
                "skipped_degenerate": res.skipped_degenerate,
                "max_changed": max_changed,
            }

    return {
        "config": config_dict(cfg),
        "query": {"id": q.id, "text": q.text, "gold_doc": q.gold_doc,
                  "gold_span": q.gold_span},
        "triage": {
            "verdict": t.verdict.value,
            "blame": t.verdict.blame,
            "ok": t.ok,
            "best_rank": t.best_rank,
            "cutoff_k": t.cutoff_k,
            "candidate_k": t.candidate_k,
            "gold_chunk_count": t.gold_chunk_count,
            "notes": list(t.notes),
        },
        "localize": {
            "summary": loc.summary(),
            "chunk_rank": loc.chunk_rank,
            "doc_rank": loc.doc_rank,
            "n_docs": loc.n_docs,
            "doc_chunk_count": loc.doc_chunk_count,
            "gold_chunk_position": loc.gold_chunk_position,
            "dilution": loc.dilution,
            "notes": list(loc.notes),
            "fragmentation": fragmentation_report(retr, q),
        },
        "results": results,
        "fix": fix_payload,
    }


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------

class _State:
    """Holds the corpus, queries and index store behind one lock.

    IndexStore memoizes built indexes and is not thread-safe, and a browser
    will happily fire several requests at once, so every request takes the
    lock. The work is milliseconds; contention is not the problem here.
    """

    def __init__(self, corpus_path: Path, queries_path: Path):
        self.corpus = Corpus.from_jsonl(corpus_path)
        self.queries = load_queries(queries_path)
        self.cache = EmbeddingCache(None)   # memory only; nothing to persist
        self.store = IndexStore(self.corpus, self.cache)
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "whiff"

    def __init__(self, *a, state: _State, **kw):
        self.state = state
        super().__init__(*a, **kw)

    # quiet: one line per request is noise when the UI polls on every keystroke
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _static(self, name: str) -> None:
        # Serve only the known files; never join user input onto a path.
        types = {"index.html": "text/html; charset=utf-8",
                 "app.css": "text/css; charset=utf-8",
                 "app.js": "application/javascript; charset=utf-8"}
        if name not in types:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        path = STATIC / name
        if not path.exists():
            self._send(404, b"missing static file", "text/plain; charset=utf-8")
            return
        self._send(200, path.read_bytes(), types[name])

    def do_GET(self) -> None:
        url = urlparse(self.path)
        params = parse_qs(url.query)
        route = url.path

        if route in ("/", "/index.html"):
            return self._static("index.html")
        if route in ("/app.css", "/app.js"):
            return self._static(route.lstrip("/"))

        if route.startswith("/api/"):
            cfg = config_from_params(params)
            try:
                with self.state.lock:
                    if route == "/api/overview":
                        return self._json(overview_payload(
                            self.state.store, cfg, self.state.queries))
                    if route == "/api/explain":
                        qid = params.get("id", [""])[0]
                        maxc = int(params.get("max_changed", ["2"])[0])
                        return self._json(explain_payload(
                            self.state.store, cfg, self.state.queries, qid,
                            max_changed=maxc))
            except Exception as exc:  # a broken config must not kill the server
                return self._json(
                    {"error": f"{type(exc).__name__}: {exc}"}, code=400)
            return self._json({"error": "unknown endpoint"}, code=404)

        self._send(404, b"not found", "text/plain; charset=utf-8")


def serve(corpus_path: Path | str, queries_path: Path | str,
          host: str = "127.0.0.1", port: int = 8008,
          open_browser: bool = True) -> None:
    state = _State(Path(corpus_path), Path(queries_path))
    handler = partial(Handler, state=state)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"

    print(f"whiff is running at {url}")
    print(f"  corpus:  {len(state.corpus)} documents")
    print(f"  queries: {len(state.queries)}")
    print("press Ctrl-C to stop")

    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
