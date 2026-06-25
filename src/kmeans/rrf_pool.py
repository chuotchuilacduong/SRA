"""RRF top-M candidate pool construction (spec Steps 2 & 3).

Two pool sources:

* **Cached extended pool** (``results/pool/extended-{ds}.json``): the published
  top-100 RRF pool with full rank lineage. Used for the A1 "current M4"
  replicate so its numbers reproduce the report exactly.

* **Recomputed fused pool** (cached under ``results/m4_v2/cache/fused_pool``):
  for M>100 we extend the BGE side to depth ``M_max`` using the corpus-embedding
  matmul (``q·corpus_emb`` — the *same* embeddings the original BGE retrieval
  used, so ranks ≤100 reproduce exactly) and keep the cached BM25 top-50, then
  RRF-fuse. This is the faithful M-generalization of how ``extended-{ds}.json``
  was built (BM25 top-50 ∪ BGE top-K, fused with k=60). Every variant slices its
  pool from this one fused list, so A2−A1 is a pure pool-size effect.

RRF(q,s) = 1/(k_rrf + rank_bm25) + 1/(k_rrf + rank_bge); a missing ranker
contributes 0 (= 1/(k+∞)).
"""

from __future__ import annotations

import logging

import numpy as np

from .config import M4V2Config
from .io import Artifacts
from . import io

logger = logging.getLogger(__name__)


def minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-pool min-max to [0,1]. All-equal pool -> 0.5 (spec edge case).

    Matches ``experiments/build_hybrid_pool.py`` exactly so A1 reproduces the
    published Method-4 numbers.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


# --- BM25 ranks from cache ----------------------------------------------------

def _load_bm25_ranks(cfg: M4V2Config, ds: str) -> dict[str, dict[str, tuple[int, float]]]:
    """``instance_id -> {skill_id: (rank_1indexed, bm25_score)}`` from the cached
    top-50 BM25 retrieval (rank implicit by list position)."""
    path = cfg.paths.get("corpus_emb").parent.parent / "retrieval_bm25" / f"{ds}-bm25.json"
    data = io.read_json(path)
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for rec in data["results"]:
        ranks = {}
        for pos, c in enumerate(rec["retrieved"], start=1):
            ranks[c["skill_id"]] = (pos, float(c.get("score", 0.0)))
        out[rec["instance_id"]] = ranks
    return out


# --- BGE top-M via corpus_emb matmul -----------------------------------------

def _bge_topm(q_emb: np.ndarray, corpus_emb: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``m`` corpus rows per query. Returns (idx[n,m], score[n,m]) sorted desc."""
    sims = q_emb @ corpus_emb.T                      # (n, N) cosine (both unit-norm)
    m = min(m, corpus_emb.shape[0])
    part = np.argpartition(-sims, kth=m - 1, axis=1)[:, :m]
    rows = np.arange(sims.shape[0])[:, None]
    part_scores = sims[rows, part]
    order = np.argsort(-part_scores, axis=1)
    idx = part[rows, order]
    score = part_scores[rows, order]
    return idx, score


# --- fused pool build ---------------------------------------------------------

def build_fused_pool(cfg: M4V2Config, ds: str, art: Artifacts,
                     qemb: dict[str, np.ndarray]) -> list[dict]:
    """Recompute the top-``pool_build_max`` RRF pool with full lineage."""
    instances = io.load_instances(cfg, ds)
    qids = [r["instance_id"] for r in instances]
    gold_by_id = {r["instance_id"]: r["gold_skill_ids"] for r in instances}

    q_mat = np.stack([qemb[qid] for qid in qids]).astype(np.float32)
    bm25 = _load_bm25_ranks(cfg, ds)
    M = cfg.pool_build_max
    bge_idx, bge_score = _bge_topm(q_mat, art.corpus_emb, M)
    k = cfg.rrf_k

    records: list[dict] = []
    for qi, qid in enumerate(qids):
        bge_rank = {int(art_idx): (r + 1, float(bge_score[qi, r]))
                    for r, art_idx in enumerate(bge_idx[qi])}
        bm = bm25.get(qid, {})
        bm_by_idx = {art.sid_to_idx[sid]: (rk, sc)
                     for sid, (rk, sc) in bm.items() if sid in art.sid_to_idx}

        cand_idx = set(bge_rank) | set(bm_by_idx)
        rows = []
        for idx in cand_idx:
            br, bsc = bge_rank.get(idx, (None, None))
            mr, msc = bm_by_idx.get(idx, (None, None))
            rrf = (1.0 / (k + mr) if mr is not None else 0.0) + \
                  (1.0 / (k + br) if br is not None else 0.0)
            rows.append({
                "skill_id": art.corpus_ids[idx],
                "rrf_score": rrf,
                "bm25_rank": mr, "bm25_score": msc,
                "bge_rank": br, "bge_score": bsc,
            })
        # sort by RRF desc; tie-break by bge_rank then skill_id (deterministic)
        rows.sort(key=lambda r: (-r["rrf_score"],
                                 r["bge_rank"] if r["bge_rank"] is not None else 1 << 30,
                                 r["skill_id"]))
        rows = rows[:M]
        for rk, r in enumerate(rows, start=1):
            r["rrf_rank"] = rk
            r["rank"] = rk
            r["score"] = r["rrf_score"]
        records.append({
            "instance_id": qid,
            "gold_skill_ids": gold_by_id[qid],
            "retrieved": rows,
        })
    return records


def load_or_build_fused_pool(cfg: M4V2Config, ds: str, art: Artifacts,
                             qemb: dict[str, np.ndarray],
                             force: bool = False) -> list[dict]:
    path = cfg.fused_pool_path(ds)
    if not force and path.exists():
        return io.read_json(path)["results"]
    logger.info("[%s] building fused RRF pool (M_max=%d)...", ds, cfg.pool_build_max)
    records = build_fused_pool(cfg, ds, art, qemb)
    io.write_json(path, {
        "metadata": {"dataset": ds, "retriever": "m4v2_fused_rrf",
                     "rrf_k": cfg.rrf_k, "pool_build_max": cfg.pool_build_max,
                     "n_queries": len(records),
                     "note": "BGE depth extended via corpus_emb matmul; BM25 from cached top-50"},
        "results": records,
    })
    return records


# --- per-variant pool selection ----------------------------------------------

def pool_for_variant(cfg: M4V2Config, variant, ds: str, art: Artifacts,
                     qemb: dict[str, np.ndarray]) -> list[dict]:
    """Return per-query candidate lists sliced to ``variant.pool``.

    A1 may use the cached extended pool (exact report match); all others use the
    recomputed fused pool. Output: list of ``{instance_id, gold_skill_ids,
    candidates: [{skill_id, rrf_score, bm25_rank, bge_rank, rrf_rank}]}``.
    """
    use_cached = (variant.id == "A1" and cfg.a1_use_cached_extended_pool)
    if use_cached:
        raw = io.read_json(cfg.extended_pool_path(ds))["results"]
    else:
        raw = load_or_build_fused_pool(cfg, ds, art, qemb)

    out = []
    for rec in raw:
        cands = rec["retrieved"][:variant.pool]
        out.append({
            "instance_id": rec["instance_id"],
            "gold_skill_ids": rec.get("gold_skill_ids") or [],
            "candidates": cands,
            "pool_source": "cached_extended" if use_cached else "fused",
        })
    return out
