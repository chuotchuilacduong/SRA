"""Ablation runner — drives variants × datasets (spec §6 pseudocode, §7 order).

Per-dataset work (query embeddings, fused pool, soft-cluster precompute) is done
ONCE and shared across all variants. Per variant × dataset we write a
metrics-compatible JSONL of reranked top-K, then evaluate and persist metrics.
Run-level stats (alpha distribution, PRF fallbacks, NaN counts) go to
``logs/m4_v2/run_summary.json`` (§16).
"""

from __future__ import annotations

import logging
import subprocess
import time

import numpy as np

from .config import M4V2Config
from .io import Artifacts
from .soft_cluster import SoftClusterIndex
from . import io, embeddings, rrf_pool, cluster_stats, scorer, evaluate

logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _prepare_dataset(cfg: M4V2Config, ds: str, art: Artifacts, need_fused: bool):
    embeddings.build_dataset_query_embeddings(cfg, ds)
    qemb = io.load_query_embeddings(cfg, ds)
    instances = io.load_instances(cfg, ds)
    qtext = {r["instance_id"]: r["query"] for r in instances}
    if need_fused:
        rrf_pool.load_or_build_fused_pool(cfg, ds, art, qemb)
    return qemb, qtext


def run(cfg: M4V2Config, datasets: list[str] | None = None,
        variant_ids: list[str] | None = None, force_embeddings: bool = False,
        force_stats: bool = False) -> dict:
    datasets = datasets or cfg.datasets
    variants = ([cfg.variant(v) for v in variant_ids] if variant_ids
                else list(cfg.variants))
    t_start = time.time()

    art = load_artifacts_cached(cfg)
    rel = cluster_stats.build_and_save(cfg, art, force=force_stats)
    soft = SoftClusterIndex(cfg, art, rel)

    # Any non-A1 variant (or A1 when not using the cached pool) needs the fused pool.
    need_fused = any(not (v.id == "A1" and cfg.a1_use_cached_extended_pool)
                     for v in variants)
    data = {ds: _prepare_dataset(cfg, ds, art, need_fused) for ds in datasets}

    all_summaries: list[dict] = []
    all_eval: dict[str, dict] = {}

    for v in variants:
        logger.info("=== variant %s (%s) ===", v.id, v.name)
        records_by_ds: dict[str, list[dict]] = {}
        for ds in datasets:
            qemb, qtext = data[ds]
            pool = rrf_pool.pool_for_variant(cfg, v, ds, art, qemb)
            records, summary = scorer.score_dataset_variant(
                cfg, v, ds, art, soft, rel, pool, qemb, qtext)
            out_path = cfg.paths["out_dir"] / v.slug / f"{ds}.jsonl"
            io.write_jsonl(out_path, records)
            metrics = evaluate.eval_records(cfg, records)
            summary["metrics"] = {k: round(metrics[k], 6) for k in metrics}
            all_summaries.append(summary)
            records_by_ds[ds] = records
            logger.info("[%s/%s] R@1=%.2f R@10=%.2f R@100=%.2f nDCG@10=%.2f (a=%.3f)",
                        v.id, ds, metrics.get("Recall@1", 0) * 100,
                        metrics.get("Recall@10", 0) * 100,
                        metrics.get("Recall@100", 0) * 100,
                        metrics.get("nDCG@10", 0) * 100, summary["alpha_mean"])
        ev = evaluate.eval_variant(cfg, records_by_ds)
        all_eval[v.id] = {"id": v.id, "name": v.name, "pool": v.pool,
                          "prf": v.prf, "affinity": v.affinity, "alpha": v.alpha,
                          "eval": ev}
        io.write_json(cfg.paths["ablation_dir"] / f"{v.id}.json", all_eval[v.id])

    # consolidated eval (for the report writer)
    io.write_json(cfg.paths["ablation_dir"] / "_all_eval.json",
                  {"datasets": datasets, "variants": all_eval})

    run_summary = {
        "git_commit": _git_commit(),
        "datasets": datasets,
        "variants": [v.id for v in variants],
        "config": cfg.raw["m4_v2"],
        "total_wall_sec": round(time.time() - t_start, 2),
        "per_run": all_summaries,
    }
    io.write_json(cfg.paths["logs_dir"] / "run_summary.json", run_summary)
    logger.info("done in %.1fs", time.time() - t_start)
    return {"eval": all_eval, "summary": run_summary, "datasets": datasets}


# small module-level cache so re-runs in one process don't reload the big arrays
_ART_CACHE: Artifacts | None = None


def load_artifacts_cached(cfg: M4V2Config) -> Artifacts:
    global _ART_CACHE
    if _ART_CACHE is None:
        _ART_CACHE = io.load_artifacts(cfg)
    return _ART_CACHE
