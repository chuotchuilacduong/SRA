"""Per-query candidate table over the RRF top-M pool (shared by QSC and LTR).

Computes, for every candidate in a query's RRF top-M pool, all the raw building
blocks the extension needs — retrieval signals, the M4 hard-affinity score, the
A7 PRF-refined score, base norms, and query-level confidence — by reusing the
M4-v2 components (no re-derivation). QSC adds local clustering on top of this;
LTR turns it into a feature matrix.
"""

from __future__ import annotations

import numpy as np

from ..common.config import M4V2Config
from ..common.io import Artifacts, load_instances
from ..m4.rrf_pool import minmax, pool_for_variant
from ..m4.soft_cluster import SoftClusterIndex
from ..m4.prf import refine_query
from ..common import io


def load_cluster_arrays(cfg: M4V2Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (size[K], cohesion[K], rel[K]) from the cached cluster stats."""
    data = io.read_json(cfg.paths["cluster_stats"])
    K = data["K"]
    size = np.array([data["clusters"][str(k)]["size"] for k in range(K)], dtype=np.float32)
    coh = np.array([data["clusters"][str(k)]["cohesion"] for k in range(K)], dtype=np.float32)
    rel = np.array([data["clusters"][str(k)]["rel"] for k in range(K)], dtype=np.float32)
    return size, coh, rel


def _tok_len(q: str) -> int:
    return len(q.split())


def iter_base_table(cfg: M4V2Config, ds: str, art: Artifacts, soft: SoftClusterIndex,
                    rel: np.ndarray, qemb: dict[str, np.ndarray], pool_size: int):
    """Yield one entry per query (streaming, so candidate embeddings aren't all
    held at once). Candidate arrays are aligned to the RRF top-M order."""
    from dataclasses import replace
    # RRF pool (= A0 candidate set) sliced to pool_size, from the fused pool.
    a0_variant = replace(cfg.variant("A0"), pool=pool_size)
    pool = pool_for_variant(cfg, a0_variant, ds, art, qemb)

    instances = {r["instance_id"]: r for r in load_instances(cfg, ds)}
    csize, ccoh, crel = load_cluster_arrays(cfg)

    for item in pool:
        qid = item["instance_id"]
        cands = item["candidates"]
        n = len(cands)
        if n == 0:
            continue
        skill_ids = [c["skill_id"] for c in cands]
        rows = np.array([art.sid_to_idx[s] for s in skill_ids], dtype=np.int64)
        e_q = qemb[qid]
        e_s = art.corpus_emb[rows]                       # (n, d)

        rrf_score = np.array([float(c["rrf_score"]) for c in cands], dtype=np.float32)
        rrf_rank = np.array([int(c.get("rrf_rank", i + 1)) for i, c in enumerate(cands)],
                            dtype=np.float64)
        # ranks/scores with missing handling
        MISS = 1_000_000.0
        bm25_rank = np.array([float(c["bm25_rank"]) if c.get("bm25_rank") is not None else MISS
                              for c in cands], dtype=np.float64)
        bge_rank = np.array([float(c["bge_rank"]) if c.get("bge_rank") is not None else MISS
                             for c in cands], dtype=np.float64)
        bm25_score = np.array([float(c["bm25_score"]) if c.get("bm25_score") is not None else 0.0
                               for c in cands], dtype=np.float32)
        bge_score = np.array([float(c["bge_score"]) if c.get("bge_score") is not None else 0.0
                              for c in cands], dtype=np.float32)
        missing_bm25 = (bm25_rank >= MISS).astype(np.float32)
        missing_bge = (bge_rank >= MISS).astype(np.float32)

        rrf_norm = minmax(rrf_score)
        bm25_norm = minmax(bm25_score)
        bge_norm = minmax(bge_score)
        direct_cos = (e_s @ e_q).astype(np.float32)

        # --- M4 hard affinity (= A1/A2 signal over this pool) ---------------
        clusters = art.skill_cluster[rows]
        q_sims = soft.query_sims(e_q)
        m4_aff = q_sims[clusters].astype(np.float32)
        m4_aff_norm = minmax(m4_aff)
        m4_score = cfg.fixed_alpha * rrf_norm + (1 - cfg.fixed_alpha) * m4_aff_norm
        m4_rank = _rank_of(m4_score)

        # --- A7 PRF-refined hard affinity -----------------------------------
        e_q_prf, prf_info = refine_query(cfg, e_q, cands, rrf_norm, art)
        q_sims_prf = soft.query_sims(e_q_prf)
        a7_aff = q_sims_prf[clusters].astype(np.float32)
        a7_aff_norm = minmax(a7_aff)
        a7_score = cfg.fixed_alpha * rrf_norm + (1 - cfg.fixed_alpha) * a7_aff_norm
        a7_rank = _rank_of(a7_score)
        query_shift = float(1.0 - float(e_q @ e_q_prf))
        cos_refined = (e_s @ e_q_prf).astype(np.float32)
        safe_mean_rrf = _safe_mean_rrf(cfg, cands, rrf_norm)

        # --- query-level confidence -----------------------------------------
        order = np.argsort(-rrf_score, kind="stable")
        rn_sorted = rrf_norm[order]
        margin_1_10 = float(rn_sorted[0] - rn_sorted[min(9, n - 1)])
        margin_1_5 = float(rn_sorted[0] - rn_sorted[min(4, n - 1)])
        p_full = soft.query_full_softmax(q_sims)
        entropy = float(-np.sum(p_full * np.log(p_full + cfg.eps)))
        overlaps = _bm25_bge_overlap(bm25_rank, bge_rank, (10, 20, 50))

        yield {
            "instance_id": qid,
            "gold_skill_ids": item["gold_skill_ids"],
            "skill_ids": skill_ids,
            "rows": rows,
            "e_q": e_q, "e_q_prf": e_q_prf, "e_s": e_s,
            "clusters": clusters,
            "arrays": {
                "rrf_score": rrf_score, "rrf_norm": rrf_norm, "rrf_rank": rrf_rank,
                "bm25_score": bm25_score, "bm25_norm": bm25_norm, "bm25_rank": bm25_rank,
                "bge_score": bge_score, "bge_norm": bge_norm, "bge_rank": bge_rank,
                "direct_cosine_q_s": direct_cos,
                "inv_rrf_rank": 1.0 / (1.0 + rrf_rank),
                "inv_bm25_rank": np.where(missing_bm25 > 0, 0.0, 1.0 / (1.0 + bm25_rank)),
                "inv_bge_rank": np.where(missing_bge > 0, 0.0, 1.0 / (1.0 + bge_rank)),
                "missing_bm25_flag": missing_bm25, "missing_bge_flag": missing_bge,
                "m4_score": m4_score, "m4_rank": m4_rank,
                "m4_affinity": m4_aff, "m4_aff_norm": m4_aff_norm,
                "global_cluster_id": clusters.astype(np.float32),
                "global_cluster_size": csize[clusters],
                "global_cluster_cohesion": ccoh[clusters],
                "global_cluster_reliability": crel[clusters],
                "prf_score": a7_score, "prf_rank": a7_rank,
                "cosine_refined_query_skill": cos_refined,
            },
            "base_norm": {
                "rrf": rrf_norm,
                "m4": minmax(m4_score),
                "a7": minmax(a7_score),
            },
            "query_feats": {
                "prf_query_shift_norm": query_shift,
                "prf_safe_set_size": float(prf_info.get("safe_set_size", 0)),
                "prf_safe_set_mean_rrf": safe_mean_rrf,
                "rrf_margin_1_10": margin_1_10,
                "rrf_margin_1_5": margin_1_5,
                "bm25_bge_top10_overlap": overlaps[10],
                "bm25_bge_top20_overlap": overlaps[20],
                "bm25_bge_top50_overlap": overlaps[50],
                "query_cluster_entropy_global": entropy,
                "query_length_tokens": float(_tok_len(instances[qid]["query"])),
            },
            "prf_info": prf_info,
        }


def _rank_of(score: np.ndarray) -> np.ndarray:
    """1-indexed rank (descending score), stable."""
    order = np.argsort(-score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return ranks


def _safe_mean_rrf(cfg: M4V2Config, cands: list[dict], rrf_norm: np.ndarray) -> float:
    idx = [i for i, c in enumerate(cands)
           if (c.get("bm25_rank") is not None and c["bm25_rank"] <= cfg.prf_bm25_rank_cutoff)
           and (c.get("bge_rank") is not None and c["bge_rank"] <= cfg.prf_bge_rank_cutoff)]
    if len(idx) < cfg.prf_min_safe_candidates:
        idx = list(range(min(cfg.prf_fallback_top_rrf, len(cands))))
    return float(np.mean(rrf_norm[idx])) if idx else 0.0


def _bm25_bge_overlap(bm25_rank: np.ndarray, bge_rank: np.ndarray,
                      ks: tuple[int, ...]) -> dict[int, float]:
    out = {}
    for K in ks:
        bm = set(np.where(bm25_rank <= K)[0].tolist())
        bg = set(np.where(bge_rank <= K)[0].tolist())
        out[K] = float(len(bm & bg) / K) if K > 0 else 0.0
    return out
