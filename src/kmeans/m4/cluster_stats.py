"""Cluster reliability statistics (spec Step 4) — runs once, offline.

rel_raw(k) = coh(k) · idf_cluster(k), where
  coh(k)         = mean_{s in C_k} e_s · mu_k        (compactness; centroids unit-norm)
  idf_cluster(k) = log( N / (|C_k| + 1) )            (specificity; small cluster -> high)
rel(k) = clamp( minmax_k(rel_raw), rel_min, rel_max ).

Note on the spec: §2 writes rel(k)=coh·log(N/|C_k|+1) (the *raw* form) but Step 4
defines rel(k) as the min-max-normalized + clamped value. Step 4 is authoritative
and is what we use (the affinity sum multiplies by the normalized, clamped rel).
"""

from __future__ import annotations

import logging

import numpy as np

from ..common.config import M4V2Config
from ..common.io import Artifacts
from ..common import io

logger = logging.getLogger(__name__)


def compute_cluster_stats(cfg: M4V2Config, art: Artifacts) -> dict:
    """Compute the per-cluster reliability and return the stats dict.

    Also returns the dense ``rel`` vector under key ``_rel_vec`` (length K)
    for in-memory use; that key is stripped before serialization.
    """
    N, K = art.N, art.K
    emb = art.corpus_emb
    cl = art.skill_cluster

    # per-row affinity to its OWN centroid (cosine, both unit-norm)
    own_aff = np.einsum("ij,ij->i", emb, art.centroids[cl]).astype(np.float64)

    size = np.bincount(cl, minlength=K).astype(np.int64)
    coh_sum = np.bincount(cl, weights=own_aff, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(size > 0, coh_sum / np.maximum(size, 1), 0.0)

    idf = np.log(N / (size + 1.0))
    rel_raw = coh * idf

    # min-max over all clusters, then clamp to [rel_min, rel_max]
    lo, hi = float(rel_raw.min()), float(rel_raw.max())
    if hi - lo < cfg.eps:
        rel = np.full(K, 0.5, dtype=np.float64)
    else:
        rel = (rel_raw - lo) / (hi - lo + cfg.eps)
    rel = np.clip(rel, cfg.rel_min, cfg.rel_max)

    clusters = {}
    for k in range(K):
        clusters[str(k)] = {
            "size": int(size[k]),
            "cohesion": round(float(coh[k]), 6),
            "cluster_idf": round(float(idf[k]), 6),
            "rel_raw": round(float(rel_raw[k]), 6),
            "rel": round(float(rel[k]), 6),
        }
    stats = {
        "K": K, "N": N,
        "rel_min": cfg.rel_min, "rel_max": cfg.rel_max,
        "clusters": clusters,
        "_rel_vec": rel.astype(np.float32),
    }
    logger.info("cluster stats: coh[min=%.3f max=%.3f] size[min=%d max=%d] "
                "rel[min=%.3f max=%.3f]", float(coh.min()), float(coh.max()),
                int(size.min()), int(size.max()), float(rel.min()), float(rel.max()))
    return stats


def build_and_save(cfg: M4V2Config, art: Artifacts, force: bool = False) -> np.ndarray:
    """Compute (or load) cluster stats, persist JSON, and return the rel vector."""
    path = cfg.paths["cluster_stats"]
    if not force and path.exists():
        data = io.read_json(path)
        K = data["K"]
        rel = np.array([data["clusters"][str(k)]["rel"] for k in range(K)],
                       dtype=np.float32)
        return rel
    stats = compute_cluster_stats(cfg, art)
    rel = stats.pop("_rel_vec")
    io.write_json(path, stats)
    logger.info("wrote %s", path)
    return rel


def load_rel_vector(cfg: M4V2Config) -> np.ndarray:
    data = io.read_json(cfg.paths["cluster_stats"])
    K = data["K"]
    return np.array([data["clusters"][str(k)]["rel"] for k in range(K)],
                    dtype=np.float32)
