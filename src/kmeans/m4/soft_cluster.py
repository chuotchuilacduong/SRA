"""Soft / hard cluster affinity (spec Steps 6 & 7).

* hard:            aff(s) = e'_q · mu_{c(s)}                      (current-M4 signal)
* hard_calibrated: aff(s) = e'_q · mu_{c(s)} · rel(c(s))
* soft:            Aff(q,s) = Σ_{k∈topL_q∩topL_s} p(k|q) p(k|s)
* soft_calibrated: Aff(q,s) = Σ_{k} p(k|q) p(k|s) rel(k)

p(k|·) is a temperature-τ softmax over the **top-L** nearest clusters only
(sparse); clusters outside the top-L get probability 0, so a query/skill pair
with no shared top-L cluster scores 0 (spec §6 "important implementation
detail"). Skill-side top-L distributions are precomputed once for the whole
corpus (a single (N,K) matmul); query-side is computed per query.
"""

from __future__ import annotations

import numpy as np

from ..common.config import M4V2Config
from ..common.io import Artifacts


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _topl(sims: np.ndarray, L: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Top-L cluster ids + softmax(tau·sim) within those L, for a (.., K) array."""
    K = sims.shape[-1]
    L = min(L, K)
    idx = np.argpartition(-sims, kth=L - 1, axis=-1)[..., :L]
    if sims.ndim == 1:
        top_sims = sims[idx]
        order = np.argsort(-top_sims)
        idx = idx[order]
        probs = _softmax(tau * sims[idx])
    else:
        rows = np.arange(sims.shape[0])[:, None]
        top_sims = sims[rows, idx]
        order = np.argsort(-top_sims, axis=1)
        idx = idx[rows, order]
        probs = _softmax(tau * sims[rows, idx], axis=1)
    return idx, probs.astype(np.float32)


class SoftClusterIndex:
    """Holds centroids, reliability, and precomputed skill-side top-L."""

    def __init__(self, cfg: M4V2Config, art: Artifacts, rel: np.ndarray):
        self.cfg = cfg
        self.art = art
        self.centroids = art.centroids                 # (K, d)
        self.K = art.K
        self.L = cfg.soft_cluster_top_l
        self.tau = cfg.softmax_tau
        self.rel = rel.astype(np.float32)              # (K,)
        self._skill_topl_idx: np.ndarray | None = None  # (N, L) int
        self._skill_topl_p: np.ndarray | None = None    # (N, L) float

    def _ensure_skill_topl(self) -> None:
        if self._skill_topl_idx is not None:
            return
        sims = self.art.corpus_emb @ self.centroids.T   # (N, K)
        self._skill_topl_idx, self._skill_topl_p = _topl(sims, self.L, self.tau)

    # --- query side ----------------------------------------------------------
    def query_sims(self, e_q: np.ndarray) -> np.ndarray:
        return e_q @ self.centroids.T                   # (K,)

    def query_full_softmax(self, q_sims: np.ndarray) -> np.ndarray:
        """Dense softmax over all K clusters (used for adaptive-α entropy)."""
        return _softmax(self.tau * q_sims).astype(np.float32)

    def query_topl(self, q_sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _topl(q_sims, self.L, self.tau)          # (L,), (L,)

    def query_prob_dense(self, q_topl_idx: np.ndarray, q_topl_p: np.ndarray) -> np.ndarray:
        dense = np.zeros(self.K, dtype=np.float32)
        dense[q_topl_idx] = q_topl_p
        return dense

    # --- affinities over a pool of candidate rows ----------------------------
    def hard_affinity(self, cand_rows: np.ndarray, q_sims: np.ndarray,
                      calibrated: bool) -> np.ndarray:
        clusters = self.art.skill_cluster[cand_rows]
        aff = q_sims[clusters]
        if calibrated:
            aff = aff * self.rel[clusters]
        return aff.astype(np.float32)

    def soft_affinity(self, cand_rows: np.ndarray, q_prob_dense: np.ndarray,
                      calibrated: bool) -> np.ndarray:
        """Σ_k p(k|q) p(k|s) [rel(k)] via the skill's top-L only.

        ``q_weighted[k] = p(k|q) * (rel(k) if calibrated else 1)``; the sum then
        runs over the skill's top-L cluster ids (clusters outside the query's
        top-L are 0 in ``q_weighted``, giving the intersection automatically).
        """
        self._ensure_skill_topl()
        q_weighted = q_prob_dense * self.rel if calibrated else q_prob_dense
        s_idx = self._skill_topl_idx[cand_rows]         # (n, L)
        s_p = self._skill_topl_p[cand_rows]             # (n, L)
        return np.sum(s_p * q_weighted[s_idx], axis=1).astype(np.float32)

    def skill_topl_for(self, cand_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_skill_topl()
        return self._skill_topl_idx[cand_rows], self._skill_topl_p[cand_rows]
