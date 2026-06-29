"""Per-query M4-v2 scoring (spec Step 10) for any ablation variant.

Pipeline per query: RRF pool slice -> minmax RRF (Step 3) -> optional PRF
(Step 5) -> affinity (Steps 6/7, hard|soft|calibrated|none) -> minmax affinity
(Step 8) -> alpha (Step 9, fixed|adaptive|none) -> blend -> stable sort -> top-K.

The output ``retrieved`` list is metrics-compatible (``sragents.retrieve.metrics``
reads ``retrieved[*].skill_id``); it carries the spec's §5.1 per-candidate fields
plus a compact ``cluster_debug``. The query-level ``top_query_clusters`` and
alpha / PRF debug live on the record (they're query-level, not per-candidate, so
storing them once keeps output files ~1000x smaller than the spec's literal layout).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..common.config import M4V2Config, Variant
from ..common.io import Artifacts
from .rrf_pool import minmax
from .soft_cluster import SoftClusterIndex
from .prf import refine_query
from . import adaptive_alpha as aa

logger = logging.getLogger(__name__)


def _stable_order(final: np.ndarray, rrf_norm: np.ndarray, rrf_rank: np.ndarray,
                  skill_ids: list[str]) -> list[int]:
    """Spec tie-break: final desc, rrf_norm desc, rrf_rank asc, skill_id asc."""
    idx = list(range(len(final)))
    idx.sort(key=lambda i: (-float(final[i]), -float(rrf_norm[i]),
                            int(rrf_rank[i]), skill_ids[i]))
    return idx


def score_query(cfg: M4V2Config, variant: Variant, art: Artifacts,
                soft: SoftClusterIndex, rel: np.ndarray,
                e_q: np.ndarray, candidates: list[dict],
                gold: set[str]) -> tuple[list[dict], dict]:
    """Score one query's candidate pool. Returns (ranked_top_k, query_debug)."""
    n = len(candidates)
    skill_ids = [c["skill_id"] for c in candidates]
    rrf_score = np.array([float(c.get("rrf_score", c.get("score", 0.0)))
                          for c in candidates], dtype=np.float32)
    rrf_rank = np.array([int(c.get("rrf_rank", c.get("rank", i + 1)))
                         for i, c in enumerate(candidates)], dtype=np.int64)
    rrf_norm = minmax(rrf_score)                                   # Step 3
    cand_rows = np.array([art.sid_to_idx[s] for s in skill_ids], dtype=np.int64)

    qdebug: dict = {"prf": None, "alpha_debug": None, "top_query_clusters": None}

    # --- Step 5: PRF -------------------------------------------------------
    if variant.prf:
        e_q_eff, prf_info = refine_query(cfg, e_q, candidates, rrf_norm, art)
        qdebug["prf"] = prf_info
    else:
        e_q_eff = e_q

    # --- Steps 6-8: affinity ----------------------------------------------
    aff = np.zeros(n, dtype=np.float32)
    aff_norm = np.zeros(n, dtype=np.float32)
    hard_cluster = art.skill_cluster[cand_rows]
    q_sims = None
    p_full = None
    q_topl_idx = q_topl_p = None
    if variant.uses_affinity:
        q_sims = soft.query_sims(e_q_eff)
        if variant.uses_soft:
            q_topl_idx, q_topl_p = soft.query_topl(q_sims)
            q_prob_dense = soft.query_prob_dense(q_topl_idx, q_topl_p)
            aff = soft.soft_affinity(cand_rows, q_prob_dense, variant.uses_calibration)
            qdebug["top_query_clusters"] = [[int(k), round(float(p), 4)]
                                            for k, p in zip(q_topl_idx, q_topl_p)]
        else:
            aff = soft.hard_affinity(cand_rows, q_sims, variant.uses_calibration)
        aff_norm = minmax(aff)                                     # Step 8

    # --- Step 9: alpha -----------------------------------------------------
    if variant.alpha == "adaptive":
        if q_sims is None:
            q_sims = soft.query_sims(e_q_eff)
        p_full = soft.query_full_softmax(q_sims)
        alpha, adbg = aa.compute_alpha(cfg, rrf_score, rrf_norm, p_full, art.K)
        qdebug["alpha_debug"] = adbg
    elif variant.alpha == "fixed":
        alpha = cfg.fixed_alpha
    else:  # none — pure RRF ordering (A0)
        alpha = 1.0

    # --- Step 10: blend + sort --------------------------------------------
    if variant.uses_affinity:
        final = alpha * rrf_norm + (1.0 - alpha) * aff_norm
    else:
        final = rrf_norm  # A0: RRF order
    order = _stable_order(final, rrf_norm, rrf_rank, skill_ids)[:cfg.output_top_k]

    ranked = []
    for new_rank, i in enumerate(order, start=1):
        sid = skill_ids[i]
        ranked.append({
            "skill_id": sid,
            "rank": new_rank,
            "final_score": round(float(final[i]), 6),
            "rrf_score": round(float(rrf_score[i]), 8),
            "rrf_norm": round(float(rrf_norm[i]), 6),
            "aff_score": round(float(aff[i]), 6),
            "aff_norm": round(float(aff_norm[i]), 6),
            "is_gold": sid in gold,
            "cluster_debug": {
                "hard_cluster": int(hard_cluster[i]),
                "cluster_reliability": round(float(rel[hard_cluster[i]]), 4),
            },
        })
    return ranked, {"alpha": round(float(alpha), 4), **qdebug}


def score_dataset_variant(cfg: M4V2Config, variant: Variant, ds: str,
                          art: Artifacts, soft: SoftClusterIndex, rel: np.ndarray,
                          pool: list[dict], qemb: dict[str, np.ndarray],
                          query_text: dict[str, str]) -> tuple[list[dict], dict]:
    """Score every query of a dataset for one variant.

    Returns (records, summary). ``records`` are metrics-compatible
    (``instance_id``/``gold_skill_ids``/``retrieved``) and carry the spec §5.1
    fields; ``summary`` aggregates alpha / PRF / NaN stats for the run log (§16).
    """
    records = []
    alphas, entropies = [], []
    prf_fallbacks = 0
    nan_aff = 0
    t0 = time.time()
    for item in pool:
        qid = item["instance_id"]
        gold = set(item["gold_skill_ids"] or [])
        cands = item["candidates"]
        if not cands:
            records.append({"instance_id": qid, "query_id": qid,
                            "query": query_text.get(qid, ""),
                            "variant": variant.id, "gold_skill_ids": list(gold),
                            "alpha": None, "retrieved": []})
            continue
        e_q = qemb[qid]
        ranked, qd = score_query(cfg, variant, art, soft, rel, e_q, cands, gold)
        if qd.get("prf") and qd["prf"].get("fallback"):
            prf_fallbacks += 1
        if any(not np.isfinite(c["aff_score"]) for c in ranked):
            nan_aff += 1
        alphas.append(qd["alpha"])
        if qd.get("alpha_debug"):
            entropies.append(qd["alpha_debug"]["entropy"])
        records.append({
            "instance_id": qid,
            "query_id": qid,
            "query": query_text.get(qid, ""),
            "variant": variant.id,
            "gold_skill_ids": list(item["gold_skill_ids"] or []),
            # A0 has no alpha (pure RRF order); report None per spec §7.1.
            "alpha": (None if variant.alpha == "none" else qd["alpha"]),
            "alpha_debug": qd.get("alpha_debug"),
            "prf": qd.get("prf"),
            "top_query_clusters": qd.get("top_query_clusters"),
            "pool_size": len(cands),
            "retrieved": ranked,
        })
    a = np.array(alphas, dtype=np.float64) if alphas else np.array([0.0])
    summary = {
        "variant": variant.id, "variant_name": variant.name, "dataset": ds,
        "pool_size": variant.pool, "num_queries": len(records),
        "fallback_prf_count": prf_fallbacks, "nan_affinity_count": nan_aff,
        "alpha_mean": round(float(a.mean()), 4),
        "alpha_p10": round(float(np.percentile(a, 10)), 4),
        "alpha_p90": round(float(np.percentile(a, 90)), 4),
        "entropy_mean": round(float(np.mean(entropies)), 4) if entropies else None,
        "wall_sec": round(time.time() - t0, 3),
    }
    return records, summary
