"""Family QSC — Query-Specific Clustering (plan §4).

For each query we run a *local* KMeans on the candidate embeddings of its RRF
top-M pool, then rerank with three scoring variants:

  QSC-1 (affinity blend):  alpha*baseNorm + (1-alpha)*localAffNorm
  QSC-2 (prior blend):     alpha*baseNorm + beta*localAffNorm + gamma*localPriorNorm
  QSC-3 (tie-break only):  baseNorm + lambda*I[baseMax-baseNorm<=delta]*localAffNorm

The local clustering for a query is base-independent (it only depends on the
candidate embeddings), so it is computed once and reused across Q0-Q6 and as the
source of the QSC features for LTR.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ..m4.rrf_pool import minmax
from ..m4.scorer import _stable_order


def local_k(n: int, default_k: int) -> int:
    """H for a pool of size n: min(default_k, max(2, floor(sqrt(n)))) (plan §4.3)."""
    return int(min(default_k, max(2, int(np.floor(np.sqrt(max(n, 1)))))))


class LocalClustering:
    """Local KMeans result for one query's candidate pool."""

    def __init__(self, e_s: np.ndarray, H: int, seed: int, n_init: int, max_iter: int):
        n = e_s.shape[0]
        self.H = min(H, n)
        if self.H <= 1:
            self.labels = np.zeros(n, dtype=np.int64)
        else:
            km = KMeans(n_clusters=self.H, random_state=seed, n_init=n_init,
                        max_iter=max_iter)
            self.labels = km.fit_predict(e_s).astype(np.int64)
        # normalized local centroids nu_h = normalize(mean_{s in G_h} e_s)
        d = e_s.shape[1]
        nu = np.zeros((self.H, d), dtype=np.float32)
        for h in range(self.H):
            mask = self.labels == h
            if mask.any():
                nu[h] = e_s[mask].mean(axis=0)
        norms = np.linalg.norm(nu, axis=1, keepdims=True)
        self.nu = nu / np.maximum(norms, 1e-12)
        self.size = np.bincount(self.labels, minlength=self.H).astype(np.float32)
        self.e_s = e_s

    def local_aff(self, qvec: np.ndarray) -> np.ndarray:
        """localAff(s) = qvec . nu_{z(s)} for each candidate."""
        cluster_aff = self.nu @ qvec                      # (H,)
        return cluster_aff[self.labels].astype(np.float32)

    def cluster_aff(self, qvec: np.ndarray) -> np.ndarray:
        return (self.nu @ qvec).astype(np.float32)

    def local_prior(self, base_norm: np.ndarray) -> np.ndarray:
        """localPrior(s) = mean base_norm over candidates in z(s)."""
        prior_h = np.zeros(self.H, dtype=np.float32)
        for h in range(self.H):
            mask = self.labels == h
            if mask.any():
                prior_h[h] = float(base_norm[mask].mean())
        return prior_h[self.labels]

    def centrality(self, e_s: np.ndarray) -> np.ndarray:
        """centrality(s) = e_s . nu_{z(s)}."""
        return np.einsum("ij,ij->i", e_s, self.nu[self.labels]).astype(np.float32)

    def cluster_size_per_cand(self) -> np.ndarray:
        return self.size[self.labels]

    def cluster_rank_by(self, cluster_value: np.ndarray) -> np.ndarray:
        """Per-candidate rank (1=best) of its cluster by the given cluster-level value."""
        order = np.argsort(-cluster_value, kind="stable")
        rank_of_cluster = np.empty(self.H, dtype=np.float32)
        rank_of_cluster[order] = np.arange(1, self.H + 1)
        return rank_of_cluster[self.labels]


def build_local_clustering(e_s: np.ndarray, qsc_cfg: dict) -> LocalClustering:
    H = local_k(e_s.shape[0], int(qsc_cfg["default_local_k"]))
    return LocalClustering(e_s, H, int(qsc_cfg["kmeans_seed"]),
                           int(qsc_cfg["kmeans_n_init"]), int(qsc_cfg["kmeans_max_iter"]))


# --- QSC scoring -------------------------------------------------------------

def score_qsc(base_norm: np.ndarray, local_aff: np.ndarray, local_prior: np.ndarray,
              scoring: str, qsc_cfg: dict) -> np.ndarray:
    aff_norm = minmax(local_aff)
    if scoring == "qsc1":
        a = float(qsc_cfg["qsc1_alpha"])
        return a * base_norm + (1 - a) * aff_norm
    if scoring == "qsc2":
        a, b, g = (float(qsc_cfg["qsc2_alpha"]), float(qsc_cfg["qsc2_beta"]),
                   float(qsc_cfg["qsc2_gamma"]))
        return a * base_norm + b * aff_norm + g * minmax(local_prior)
    if scoring == "qsc3":
        delta, lam = float(qsc_cfg["qsc3_delta"]), float(qsc_cfg["qsc3_lambda"])
        base_max = float(base_norm.max()) if base_norm.size else 0.0
        I = (base_max - base_norm <= delta).astype(np.float32)
        return base_norm + lam * I * aff_norm
    raise ValueError(f"unknown QSC scoring {scoring!r}")


def score_one(cfg, qsc_cfg: dict, e: dict, local: LocalClustering, variant: dict,
              query_text: dict[str, str]) -> dict:
    """Rerank one query for a single Q-variant. ``variant`` = {id,name,base,scoring}."""
    base_key = variant["base"]
    use_prf = (base_key == "a7")
    qid = e["instance_id"]
    base_norm = e["base_norm"][base_key]
    qvec = e["e_q_prf"] if use_prf else e["e_q"]
    local_aff = local.local_aff(qvec)
    local_prior = local.local_prior(base_norm)
    final = score_qsc(base_norm, local_aff, local_prior, variant["scoring"], qsc_cfg)

    aff_norm = minmax(local_aff)
    rrf_rank = e["arrays"]["rrf_rank"]
    skill_ids = e["skill_ids"]
    order = _stable_order(final, base_norm, rrf_rank.astype(np.int64), skill_ids)
    order = order[:cfg.output_top_k]
    gold = set(e["gold_skill_ids"] or [])
    ranked = []
    for new_rank, i in enumerate(order, start=1):
        ranked.append({
            "skill_id": skill_ids[i], "rank": new_rank,
            "final_score": round(float(final[i]), 6),
            "base_norm": round(float(base_norm[i]), 6),
            "local_aff": round(float(local_aff[i]), 6),
            "local_aff_norm": round(float(aff_norm[i]), 6),
            "local_cluster_id": int(local.labels[i]),
            "local_cluster_size": int(local.size[local.labels[i]]),
            "is_gold": skill_ids[i] in gold,
        })
    return {
        "instance_id": qid, "query_id": qid, "query": query_text.get(qid, ""),
        "variant": variant["id"], "base": base_key, "scoring": variant["scoring"],
        "gold_skill_ids": list(e["gold_skill_ids"] or []),
        "local_H": local.H, "retrieved": ranked,
    }


# --- QSC features for LTR (canonical: query vec = e_q, prior base = rrf) ------

def qsc_features(e: dict, local: LocalClustering) -> dict[str, np.ndarray]:
    qvec = e["e_q"]
    base_norm = e["base_norm"]["rrf"]
    local_aff = local.local_aff(qvec)
    local_prior = local.local_prior(base_norm)
    cluster_aff = local.cluster_aff(qvec)
    prior_h = np.array([base_norm[local.labels == h].mean() if (local.labels == h).any()
                        else 0.0 for h in range(local.H)], dtype=np.float32)
    return {
        "qsc_local_affinity": local_aff,
        "qsc_local_aff_norm": minmax(local_aff),
        "qsc_local_prior": local_prior,
        "qsc_local_prior_norm": minmax(local_prior),
        "qsc_local_centrality": local.centrality(e["e_s"]),
        "qsc_local_cluster_id": local.labels.astype(np.float32),
        "qsc_local_cluster_size": local.cluster_size_per_cand(),
        "qsc_local_cluster_rank_by_prior": local.cluster_rank_by(prior_h),
        "qsc_local_cluster_rank_by_affinity": local.cluster_rank_by(cluster_aff),
    }
