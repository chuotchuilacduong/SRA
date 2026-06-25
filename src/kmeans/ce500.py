"""Cross-Encoder Method 7 at pool depth 500 (CE@500).

Method 7 = α-CE β=0.7: rerank a candidate pool with the trained CE (ce-joint-v3,
MiniLM-L6) then fuse ``0.7·minmax(CE) + 0.3·minmax(Stage1)``. The published
Method 7 reranks the RRF top-100 (Stage1 = M4 hybrid score). CE@500 reruns the
SAME pipeline on the RRF top-500 (Stage1 = M4-over-500). No retraining — only CE
inference on the extra candidates.

CPU-only (MPS rejected for CrossEncoder), parallelized across processes.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from pathlib import Path

from sragents.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CE_MODEL = "results/models/ce-joint-v3"
PACKER_KW = {"mode": "field_tagged", "max_content_chars": 1800}
MAX_LENGTH = 256
BATCH_SIZE = 64
BETA = 0.7


def build_reranker(beta: float = BETA):
    from sragents.corpus import load_corpus_dict
    from sragents.retrieve.skill_packer import SkillPacker
    from sragents.retrieve.beta_ce_rerank import BetaCEReranker
    packer = SkillPacker(**PACKER_KW)
    return BetaCEReranker(model_name=str(PROJECT_ROOT / CE_MODEL), device="cpu",
                          max_length=MAX_LENGTH, packer=packer,
                          corpus=load_corpus_dict(), beta=beta)


def rerank_chunk(args: dict) -> list[dict]:
    """Worker: rerank a list of queries with a fresh CE instance.

    args: {queries: [{instance_id, dataset, question, gold, candidates:[{skill_id, score}]}],
           beta, top_k}
    """
    ce = build_reranker(args.get("beta", BETA))
    out = []
    for q in args["queries"]:
        ranked = ce.rerank(q["question"], q["candidates"], top_k=args["top_k"],
                           batch_size=BATCH_SIZE)
        out.append({"instance_id": q["instance_id"], "dataset": q["dataset"],
                    "gold_skill_ids": q["gold"],
                    "retrieved": [{"skill_id": c["skill_id"]} for c in ranked]})
    return out


# --- candidate-list builders -------------------------------------------------

def queries_from_feature_tables(cfg, ext, datasets, pool_size=500):
    """RRF top-``pool_size`` per query with Stage1 = M4 score (from feature tables)."""
    from kmeans import ltr_features, ltr, io
    im4 = ltr_features.ALL_FEATURES.index("m4_score")
    qtext = {}
    for ds in datasets:
        for r in io.load_instances(cfg, ds):
            qtext[r["instance_id"]] = r["query"]
    out = []
    for ds in datasets:
        t = ltr_features.load_features(PROJECT_ROOT / ext["reporting"]["cache_dir"] / f"{ds}.npz")
        for qid, Xq, yq, sids, gold in ltr.iter_queries(t):
            n = min(pool_size, len(sids))
            cands = [{"skill_id": sids[j], "score": float(Xq[j, im4])} for j in range(n)]
            out.append({"instance_id": qid, "dataset": ds, "question": qtext[qid],
                        "gold": list(gold), "candidates": cands})
    return out


def queries_from_hybrid_km(cfg, datasets, top_k=100):
    """Reproduce the published Method-7 input: hybrid_km_alpha70 top-100 (Stage1 = its score)."""
    from kmeans import io
    out = []
    for ds in datasets:
        pool = json.loads((PROJECT_ROOT / f"results/pool/hybrid_km_alpha70-{ds}.json").read_text())
        qtext = {r["instance_id"]: r["query"] for r in io.load_instances(cfg, ds)}
        for rec in pool["results"]:
            cands = [{"skill_id": c["skill_id"], "score": float(c.get("score", 0.0))}
                     for c in rec["retrieved"][:top_k]]
            out.append({"instance_id": rec["instance_id"], "dataset": ds,
                        "question": qtext[rec["instance_id"]],
                        "gold": rec["gold_skill_ids"] or [], "candidates": cands})
    return out


# --- parallel driver ---------------------------------------------------------

def run_parallel(queries: list[dict], workers: int, beta: float, top_k: int) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    chunks = [queries[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    results = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
        futs = [ex.submit(rerank_chunk, {"queries": ch, "beta": beta, "top_k": top_k})
                for ch in chunks]
        for i, fut in enumerate(as_completed(futs)):
            results.extend(fut.result())
            logger.info("  chunk %d/%d done (%d queries total)", i + 1, len(chunks), len(results))
    return results
