"""Orchestrator for the QSC + LTR extension (plan §15 execution order).

Per dataset (streaming, one query at a time to bound memory):
  build base table -> local KMeans -> score Q0-Q6 -> accumulate LTR features.
Then: train/predict LTR L0-L7 on a query-level stratified split, and evaluate
everything. QSC is evaluated on the FULL query set (no training); LTR on the
held-out TEST split (with baselines recomputed on the same test split for a fair
delta). Writes a consolidated eval JSON for the report writer.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from sragents.config import PROJECT_ROOT

from ..common.config import M4V2Config
from ..m4.soft_cluster import SoftClusterIndex
from ..m4.ablation_runner import load_artifacts_cached
from .base_table import iter_base_table
from .qsc import build_local_clustering, score_one
from .ltr_features import FeatureAccumulator, save_features
from ..common import io, embeddings, evaluate
from ..m4 import cluster_stats
from . import ltr

logger = logging.getLogger(__name__)


def load_ext(path: str | None = None) -> tuple[dict, M4V2Config]:
    p = Path(path) if path else (PROJECT_ROOT / "src/kmeans/configs/qsc_ltr_extension.yaml")
    raw = yaml.safe_load(Path(p).read_text())
    ext = raw["qsc_ltr_extension"]
    cfg = M4V2Config.load(PROJECT_ROOT / ext["base_config"])
    return ext, cfg


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _group_by_dataset(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get("dataset", "unknown"), []).append(r)
    return out


def _filter_by_ids(records: list[dict], ids: set) -> list[dict]:
    return [r for r in records if r["instance_id"] in ids]


def _eval_m4_baseline_on_ids(cfg: M4V2Config, vid: str, datasets: list[str],
                             ids: set | None) -> dict:
    """Evaluate a stored M4-v2 variant (A0/A1/A2/A7...) on a query-id subset."""
    slug = cfg.variant(vid).slug
    by_ds = {}
    for ds in datasets:
        path = cfg.paths["out_dir"] / slug / f"{ds}.jsonl"
        if not path.exists():
            continue
        recs = _read_jsonl(path)
        if ids is not None:
            recs = _filter_by_ids(recs, ids)
        if recs:
            by_ds[ds] = recs
    return evaluate.eval_variant(cfg, by_ds)


def _read_jsonl(path: Path) -> list[dict]:
    import json
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def run(ext_path: str | None = None, datasets: list[str] | None = None) -> dict:
    ext, cfg = load_ext(ext_path)
    datasets = datasets or ext["datasets"]
    qsc_cfg = ext["qsc"]
    pool_size = int(qsc_cfg["pool_size"])
    q_variants = qsc_cfg["variants"]
    out_dir = _abs(ext["reporting"]["out_dir"])
    cache_dir = _abs(ext["reporting"]["cache_dir"])

    art = load_artifacts_cached(cfg)
    rel = cluster_stats.build_and_save(cfg, art)
    soft = SoftClusterIndex(cfg, art, rel)  # only query-side methods used here

    # --- QSC + LTR feature build (streaming per dataset) --------------------
    qsc_records: dict[str, dict[str, list[dict]]] = {v["id"]: {} for v in q_variants}
    tables: dict[str, dict] = {}
    for ds in datasets:
        embeddings.build_dataset_query_embeddings(cfg, ds)
        qemb = io.load_query_embeddings(cfg, ds)
        qtext = {r["instance_id"]: r["query"] for r in io.load_instances(cfg, ds)}
        acc = FeatureAccumulator(ds)
        per_ds = {v["id"]: [] for v in q_variants}
        nq = 0
        for e in iter_base_table(cfg, ds, art, soft, rel, qemb, pool_size):
            local = build_local_clustering(e["e_s"], qsc_cfg)
            for v in q_variants:
                per_ds[v["id"]].append(score_one(cfg, qsc_cfg, e, local, v, qtext))
            acc.add(e, local)
            nq += 1
        table = acc.table()
        save_features(cache_dir / f"{ds}.npz", table)
        tables[ds] = table
        for v in q_variants:
            qsc_records[v["id"]][ds] = per_ds[v["id"]]
            io.write_jsonl(out_dir / "qsc" / v["id"] / f"{ds}.jsonl", per_ds[v["id"]])
        logger.info("[%s] %d queries: QSC scored (Q0-Q6) + features cached", ds, nq)

    # --- QSC evaluation (FULL set) ------------------------------------------
    qsc_eval = {}
    for v in q_variants:
        ev = evaluate.eval_variant(cfg, qsc_records[v["id"]])
        qsc_eval[v["id"]] = {"id": v["id"], "name": v["name"], "base": v["base"],
                             "scoring": v["scoring"], "eval": ev}
        m = ev["macro"]
        logger.info("[QSC %s %s] R@1=%.2f R@10=%.2f R@100=%.2f nDCG@10=%.2f",
                    v["id"], v["name"], m.get("Recall@1", 0) * 100,
                    m.get("Recall@10", 0) * 100, m.get("Recall@100", 0) * 100,
                    m.get("nDCG@10", 0) * 100)

    # --- LTR train/predict on query-level split -----------------------------
    split = ltr.make_split(tables, ext["ltr"]["split"])
    logger.info("LTR split: train=%d dev=%d test=%d queries",
                len(split["train"]), len(split["dev"]), len(split["test"]))
    ltr_eval = {}
    for v in ext["ltr"]["variants"]:
        recs, meta = ltr.run_variant(cfg, tables, split, v,
                                     ext["ltr"]["lightgbm"], ext["ltr"]["linear_pairwise"])
        io.write_jsonl(out_dir / "ltr" / v["id"] / "test.jsonl", recs)
        ev = evaluate.eval_variant(cfg, _group_by_dataset(recs))
        ltr_eval[v["id"]] = {"id": v["id"], "name": v["name"], "groups": v["groups"],
                             "model": v["model"], "eval": ev, "meta": meta}
        m = ev["macro"]
        logger.info("[LTR %s %s] (TEST) R@1=%.2f R@10=%.2f R@100=%.2f nDCG@10=%.2f",
                    v["id"], v["name"], m.get("Recall@1", 0) * 100,
                    m.get("Recall@10", 0) * 100, m.get("Recall@100", 0) * 100,
                    m.get("nDCG@10", 0) * 100)

    # --- baselines: full-set (from M4-v2) + test-split ----------------------
    baselines_full = {}
    m4_all = cfg.paths["ablation_dir"] / "_all_eval.json"
    if m4_all.exists():
        data = io.read_json(m4_all)["variants"]
        for vid in ["A0", "A1", "A2", "A7", "A8"]:
            if vid in data:
                baselines_full[vid] = {"id": vid, "name": data[vid]["name"],
                                       "eval": data[vid]["eval"]}

    test_ids = split["test"]
    baselines_test = {}
    for vid in ["A0", "A1", "A2", "A7", "A8"]:
        baselines_test[vid] = {"id": vid,
                               "eval": _eval_m4_baseline_on_ids(cfg, vid, datasets, test_ids)}
    for qid in ["Q2", "Q6"]:
        by_ds = {ds: _filter_by_ids(qsc_records[qid][ds], test_ids) for ds in datasets}
        by_ds = {k: v for k, v in by_ds.items() if v}
        baselines_test[qid] = {"id": qid, "eval": evaluate.eval_variant(cfg, by_ds)}

    consolidated = {
        "datasets": datasets,
        "qsc": qsc_eval,
        "ltr": ltr_eval,
        "baselines_full": baselines_full,
        "baselines_test": baselines_test,
        "split_sizes": {k: len(v) for k, v in split.items()},
        "test_ids_per_dataset": {ds: sum(1 for q in tables[ds]["instance_ids"]
                                         if q in test_ids) for ds in datasets},
    }
    io.write_json(out_dir / "consolidated_eval.json", consolidated)
    logger.info("wrote %s", out_dir / "consolidated_eval.json")
    return consolidated
