"""Evaluation for M4-v2 — thin wrapper over ``sragents.retrieve.metrics``.

We reuse the EXACT metric functions that produced the published numbers, so
A1 is directly comparable. Metrics are always computed with ``top_k=eval_top_k``
(=100) so Recall@{1,5,10,50,100} and nDCG@{1,5,10} are all produced.

Macro = equal-weight mean over datasets (spec §9.5). Micro = pooled over queries.
"""

from __future__ import annotations

import logging

import numpy as np

from sragents.retrieve.metrics import compute_retrieval_metrics

from .config import M4V2Config

logger = logging.getLogger(__name__)

REPORT_METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100",
                  "nDCG@1", "nDCG@5", "nDCG@10"]


def eval_records(cfg: M4V2Config, records: list[dict]) -> dict[str, float]:
    """Metrics for one dataset's records (records have gold_skill_ids+retrieved)."""
    return compute_retrieval_metrics(records, top_k=cfg.eval_top_k)


def eval_variant(cfg: M4V2Config, records_by_ds: dict[str, list[dict]]) -> dict:
    """Per-dataset + macro + micro metrics for a single variant."""
    by_ds = {ds: eval_records(cfg, recs) for ds, recs in records_by_ds.items()}
    pooled = [r for recs in records_by_ds.values() for r in recs]
    micro = compute_retrieval_metrics(pooled, top_k=cfg.eval_top_k)

    macro: dict[str, float] = {}
    if by_ds:
        keys = set().union(*[set(m) for m in by_ds.values()])
        for k in keys:
            macro[k] = float(np.mean([by_ds[ds].get(k, 0.0) for ds in by_ds]))
    return {"by_dataset": by_ds, "macro": macro, "micro": micro}


def variant_metrics_rows(variant_id: str, variant_name: str, ev: dict,
                         datasets: list[str],
                         metrics: list[str] = REPORT_METRICS) -> list[dict]:
    """Flatten one variant's eval into CSV rows: one row per (metric)."""
    rows = []
    for metric in metrics:
        row = {"variant": variant_id, "variant_name": variant_name, "metric": metric}
        for ds in datasets:
            row[ds] = round(ev["by_dataset"].get(ds, {}).get(metric, float("nan")) * 100, 2)
        row["macro_avg"] = round(ev["macro"].get(metric, float("nan")) * 100, 2)
        row["micro_avg"] = round(ev["micro"].get(metric, float("nan")) * 100, 2)
        rows.append(row)
    return rows
