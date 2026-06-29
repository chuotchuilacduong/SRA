"""Adaptive alpha (spec Step 9).

conf_rrf(q)     = RRF_norm(s_1) - RRF_norm(s_10)         (RRF top-gap confidence)
H(q)            = -Σ_k p(k|q) log(p(k|q)+ε)
conf_cluster(q) = 1 - H(q)/log K                          (query cluster peakedness)
α(q)            = clip( α0 + λ·[conf_rrf - conf_cluster], α_min, α_max )

Decision (documented in IMPLEMENTATION_NOTES §9): the entropy uses the FULL-K
query softmax distribution, not the sparse top-L one used for the affinity. With
only L=10 nonzero entries the normalization by log K would cap H_norm at
log L / log K ≈ 0.40 and make conf_cluster meaningless; the full-K distribution
keeps H_norm ∈ [0,1] as the formula intends.
"""

from __future__ import annotations

import numpy as np

from .config import M4V2Config


def rrf_confidence(cfg: M4V2Config, rrf_score: np.ndarray, rrf_norm: np.ndarray) -> float:
    n = len(rrf_norm)
    if n == 0:
        return 0.0
    # stable sort so s_1 / s_10 are deterministic when rrf_score ties (the gap
    # value is unaffected — tied candidates share rrf_norm — but the choice of
    # index is now platform-independent).
    order = np.argsort(-np.asarray(rrf_score), kind="stable")
    top = order[min(cfg.conf_rrf_top - 1, n - 1)]
    bottom = order[min(cfg.conf_rrf_bottom - 1, n - 1)]
    return float(rrf_norm[top] - rrf_norm[bottom])


def cluster_confidence(cfg: M4V2Config, p_full: np.ndarray, K: int) -> tuple[float, float]:
    """Return (conf_cluster, entropy). ``p_full`` is the full-K query softmax."""
    H = float(-np.sum(p_full * np.log(p_full + cfg.eps)))
    H_norm = H / np.log(K) if K > 1 else 0.0
    return 1.0 - H_norm, H


def compute_alpha(cfg: M4V2Config, rrf_score: np.ndarray, rrf_norm: np.ndarray,
                  p_full: np.ndarray, K: int) -> tuple[float, dict]:
    conf_rrf = rrf_confidence(cfg, rrf_score, rrf_norm)
    conf_cluster, entropy = cluster_confidence(cfg, p_full, K)
    raw = cfg.alpha0 + cfg.alpha_lambda * (conf_rrf - conf_cluster)
    alpha = float(np.clip(raw, cfg.alpha_min, cfg.alpha_max))
    return alpha, {
        "conf_rrf": round(conf_rrf, 4),
        "conf_cluster": round(conf_cluster, 4),
        "entropy": round(entropy, 4),
        "alpha_raw": round(float(raw), 4),
    }
