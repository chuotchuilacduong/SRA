"""Safe pseudo-relevance feedback query refinement (spec Step 5).

Safe set A(q) = candidates trusted by BOTH retrievers
    {s : rank_bm25 ≤ R_b ∧ rank_bge ≤ R_d}.
If |A(q)| < min_safe, fall back to the top-``fallback_top_rrf`` by RRF.

PRF weights w_i = softmax(γ · RRF_norm(s_i)) over A(q); refined vector
    e'_q = normalize( (1-ρ) e_q + ρ Σ w_i e_{s_i} ).
NaN / zero-norm result -> keep e_q (recorded as a fallback).
"""

from __future__ import annotations

import numpy as np

from .config import M4V2Config
from .io import Artifacts
from .soft_cluster import _softmax


def refine_query(cfg: M4V2Config, e_q: np.ndarray, candidates: list[dict],
                 rrf_norm: np.ndarray, art: Artifacts) -> tuple[np.ndarray, dict]:
    """Return (refined_e_q, info). ``candidates`` is the pool slice (each with
    ``skill_id``, ``bm25_rank``, ``bge_rank``); ``rrf_norm`` aligns to it."""
    info = {"used_prf": True, "safe_set_size": 0, "fallback": None}

    def _rank(c, key):
        r = c.get(key)
        return r if r is not None else np.inf

    safe_idx = [i for i, c in enumerate(candidates)
                if _rank(c, "bm25_rank") <= cfg.prf_bm25_rank_cutoff
                and _rank(c, "bge_rank") <= cfg.prf_bge_rank_cutoff]

    if len(safe_idx) < cfg.prf_min_safe_candidates:
        # fallback: top-N by RRF (candidates are already RRF-sorted in the pool)
        safe_idx = list(range(min(cfg.prf_fallback_top_rrf, len(candidates))))
        info["fallback"] = "few_safe_candidates"

    info["safe_set_size"] = len(safe_idx)
    if not safe_idx:
        info["used_prf"] = False
        info["fallback"] = "empty_pool"
        return e_q, info

    sel_norm = rrf_norm[safe_idx]
    w = _softmax(cfg.prf_gamma * sel_norm)                 # (|A|,)
    rows = np.array([art.sid_to_idx[candidates[i]["skill_id"]] for i in safe_idx])
    centroid = (w[:, None] * art.corpus_emb[rows]).sum(axis=0)   # Σ w_i e_{s_i}

    refined = (1.0 - cfg.prf_rho) * e_q + cfg.prf_rho * centroid
    nrm = float(np.linalg.norm(refined))
    if not np.isfinite(refined).all() or nrm < cfg.eps:
        info["used_prf"] = False
        info["fallback"] = "nan_or_zero_norm"
        return e_q, info
    return (refined / nrm).astype(np.float32), info
