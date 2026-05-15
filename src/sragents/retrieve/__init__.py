"""Stage 1: skill retrieval.

Public API::

    from sragents.retrieve import get, register, list_retrievers
    from sragents.retrieve.schema import RetrievalResults

Built-in retrievers (``bm25``, ``tfidf``, ``bge``, ``contriever``) are
registered on import. Hybrid fusion is in :mod:`sragents.retrieve.hybrid`,
LLM rerank in :mod:`sragents.retrieve.llm_rerank`.
"""

# Trigger registration of built-in retrievers.
# linearrag is loaded lazily to avoid the spacy/numpy ABI conflict at import
# time — it is only needed when the `retrieve` subcommand actually runs.
from sragents.retrieve import bm25, tfidf, dense  # noqa: F401

def _load_linearrag() -> None:
    """Import linearrag (and spacy) only when a retriever is actually used."""
    from sragents.retrieve import linearrag  # noqa: F401, PLC0415
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
