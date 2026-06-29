"""Stage 1: skill retrieval.

Public API::

    from sragents.retrieve import get, register, list_retrievers
    from sragents.retrieve.schema import RetrievalResults

Built-in retrievers (``bm25``, ``tfidf``, ``bge``, ``contriever``) are
registered on import. Hybrid fusion is in :mod:`sragents.retrieve.hybrid`,
LLM rerank in :mod:`sragents.retrieve.llm_rerank`.
"""

# Trigger registration of built-in retrievers.
from sragents.retrieve import bm25, tfidf, dense  # noqa: F401
# linearrag is optional: it pulls spacy→thinc, which hits the numpy-2 ABI break
# in the `sra` env. It is unused by the kmeans/CE-Raw pipelines, so make it
# best-effort (keep this guard or the kmeans scripts re-break on import).
try:
    from sragents.retrieve import linearrag  # noqa: F401
except Exception:
    pass
from sragents.retrieve.base import (
    Retriever,
    get,
    list_retrievers,
    register,
)
from sragents.retrieve.metrics import compute_retrieval_metrics

__all__ = [
    "Retriever",
    "register",
    "get",
    "list_retrievers",
    "compute_retrieval_metrics",
]
