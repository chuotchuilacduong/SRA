"""Query embedding computation for M4-v2.

The repo never cached query embeddings (queries were re-encoded at retrieval
time). M4-v2 needs the raw L2-normalized query vector ``e_q`` for soft cluster
affinity, PRF, and adaptive-alpha, so we encode once with the SAME encoder +
prefix the corpus/RRF pipeline used (``BAAI/bge-base-en-v1.5``) and cache.

This is the only step that may need a GPU; everything downstream is matmuls.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import M4V2Config
from . import io

logger = logging.getLogger(__name__)

_MODEL = None  # lazy singleton


def _pick_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model(cfg: M4V2Config):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        device = _pick_device()
        logger.info("loading %s on %s", cfg.bge_model, device)
        _MODEL = SentenceTransformer(cfg.bge_model, device=device)
    return _MODEL


def encode_queries(cfg: M4V2Config, queries: list[str]) -> np.ndarray:
    """Encode ``queries`` -> (n, d) float32, L2-normalized, with the BGE prefix."""
    model = _get_model(cfg)
    emb = model.encode(
        [cfg.bge_prefix + q for q in queries],
        batch_size=cfg.bge_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_dataset_query_embeddings(cfg: M4V2Config, ds: str,
                                   force: bool = False) -> int:
    """Encode + cache query embeddings for one dataset. Returns #queries."""
    if not force and cfg.query_emb_path(ds).exists() and cfg.query_emb_ids_path(ds).exists():
        ids = io.read_json(cfg.query_emb_ids_path(ds))
        logger.info("[%s] query embeddings cached (%d)", ds, len(ids))
        return len(ids)
    instances = io.load_instances(cfg, ds)
    ids = [r["instance_id"] for r in instances]
    queries = [r["query"] for r in instances]
    emb = encode_queries(cfg, queries)
    io.save_query_embeddings(cfg, ds, ids, emb)
    logger.info("[%s] encoded %d queries -> %s", ds, len(ids), cfg.query_emb_path(ds))
    return len(ids)
