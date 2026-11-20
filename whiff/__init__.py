"""whiff - a retrieval debugger.

Tells you why your retriever missed a document, and the smallest change that
would have found it.
"""

from .config import RetrievalConfig
from .pipeline import Corpus, IndexStore, Query, build, load_queries
from .embed import EmbeddingCache
from .triage import Verdict, triage, triage_all, summarize
from .localize import localize, fragmentation_report
from .counterfactual import find_minimal_fix, headroom, search_fix

__version__ = "0.1.0"

__all__ = [
    "RetrievalConfig", "Corpus", "Query", "IndexStore", "EmbeddingCache",
    "build", "load_queries", "Verdict", "triage", "triage_all", "summarize",
    "localize", "fragmentation_report", "find_minimal_fix", "search_fix",
    "headroom",
]
