"""Retrieval evaluation metrics: Recall, nDCG, Hit, MRR, P @ K.

All metrics are *multi-label aware*: for a query with ``|gold| = g`` golds,
``Recall@K = |gold ∩ retrieved[:K]| / g`` and the ideal DCG sums over
``min(g, K)`` perfectly-placed relevances. For single-gold queries these
reduce to the standard formulas.
"""

from __future__ import annotations

import numpy as np

# Pre-compute log2(rank+1) discounts up to a large rank so callers don't pay
# the log cost in a hot loop.
_MAX_K = 200
_LOG2_DISCOUNT = 1.0 / np.log2(np.arange(2, _MAX_K + 2))


_DEFAULT_KS = {
    "recall": (1, 5, 10, 50, 100),
    "ndcg": (1, 5, 10, 50, 100),
    "hit": (1, 10),
    "mrr": (10, 100),
    "p": (1, 5, 10),
}


def _filter_ks(ks: tuple[int, ...], cap: int) -> list[int]:
    """Keep only K values reachable given a retrieval cap (``len(retrieved)``)
    is not used as the cap, otherwise metrics would silently mute when a
    retriever returns fewer than K. Callers pass ``top_k`` (the configured
    cutoff) so the caps are config-driven, not data-driven.
    """
    return [k for k in ks if k <= cap]


def compute_retrieval_metrics(
    results: list[dict],
    top_k: int = 10,
    ks: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, float]:
    """Compute macro-averaged retrieval metrics over a list of records.

    Each ``results`` entry has ``gold_skill_ids`` (list) and ``retrieved``
    (list of ``{skill_id, score, ...}`` dicts).

    Args:
        results: Retrieval records.
        top_k: Maximum K to consider. Metric cutoffs > top_k are dropped.
        ks: Optional override per metric family. Keys: ``recall``, ``ndcg``,
            ``hit``, ``mrr``, ``p``.

    Returns:
        ``{"<Metric>@K": float, ...}``. Queries with empty gold are skipped
        (consistent with the existing pipeline contract — but callers can
        keep them as ``candidate_recall_miss=true`` records for audit).
    """
    ks_cfg = {**_DEFAULT_KS, **(ks or {})}
    recall_ks = _filter_ks(ks_cfg["recall"], top_k)
    ndcg_ks = _filter_ks(ks_cfg["ndcg"], top_k)
    hit_ks = _filter_ks(ks_cfg["hit"], top_k)
    mrr_ks = _filter_ks(ks_cfg["mrr"], top_k)
    p_ks = _filter_ks(ks_cfg["p"], top_k)

    recall_acc = {k: [] for k in recall_ks}
    ndcg_acc = {k: [] for k in ndcg_ks}
    hit_acc = {k: [] for k in hit_ks}
    mrr_acc = {k: [] for k in mrr_ks}
    p_acc = {k: [] for k in p_ks}

    largest_k = max(
        [top_k]
        + recall_ks
        + ndcg_ks
        + hit_ks
        + mrr_ks
        + p_ks
        or [1]
    )

    for r in results:
        gold = set(r.get("gold_skill_ids") or [])
        if not gold:
            continue
        retrieved = [entry["skill_id"] for entry in r.get("retrieved", [])]
        n = min(largest_k, len(retrieved))
        rels = np.array([1.0 if sid in gold else 0.0 for sid in retrieved[:n]])

        for k in recall_ks:
            cut = min(k, len(rels))
            recall_acc[k].append(float(rels[:cut].sum()) / len(gold))

        for k in ndcg_ks:
            cut = min(k, len(rels))
            dcg = float(rels[:cut] @ _LOG2_DISCOUNT[:cut])
            ideal_n = min(len(gold), k)
            idcg = float(_LOG2_DISCOUNT[:ideal_n].sum())
            ndcg_acc[k].append(dcg / idcg if idcg > 0 else 0.0)

        for k in hit_ks:
            cut = min(k, len(rels))
            hit_acc[k].append(1.0 if rels[:cut].sum() > 0 else 0.0)

        for k in mrr_ks:
            cut = min(k, len(rels))
            mrr = 0.0
            for i in range(cut):
                if rels[i] > 0:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_acc[k].append(mrr)

        for k in p_ks:
            cut = min(k, len(rels))
            if cut == 0:
                p_acc[k].append(0.0)
            else:
                p_acc[k].append(float(rels[:cut].sum()) / k)

    out: dict[str, float] = {}
    for k in recall_ks:
        out[f"Recall@{k}"] = _mean(recall_acc[k])
    for k in ndcg_ks:
        out[f"nDCG@{k}"] = _mean(ndcg_acc[k])
    for k in hit_ks:
        out[f"Hit@{k}"] = _mean(hit_acc[k])
    for k in mrr_ks:
        out[f"MRR@{k}"] = _mean(mrr_acc[k])
    for k in p_ks:
        out[f"P@{k}"] = _mean(p_acc[k])
    return out


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def compute_metrics_by_dataset(
    results: list[dict],
    instances_by_id: dict[str, dict],
    dataset_field: str = "dataset",
    top_k: int = 100,
    ks: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Macro/micro metric split + per-dataset breakdown.

    The "macro" line is the equal-weight average over datasets; the "micro"
    is the average over all queries pooled. They differ when datasets have
    different sizes.

    Args:
        results: Records, each with ``instance_id``, ``gold_skill_ids``,
                 ``retrieved``.
        instances_by_id: Lookup ``instance_id -> instance dict`` (used to
                         locate the dataset name).
        dataset_field: Key on the instance dict that names the dataset.

    Returns:
        (overall_metrics, by_dataset_metrics).
        ``overall_metrics`` has keys ``Macro/<Metric>@K`` and ``Micro/<Metric>@K``.
        ``by_dataset_metrics`` maps dataset -> {Metric@K: float}.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        inst = instances_by_id.get(r["instance_id"])
        if inst is None:
            continue
        ds = inst.get(dataset_field) or "unknown"
        groups.setdefault(ds, []).append(r)

    by_dataset: dict[str, dict[str, float]] = {}
    for ds, recs in groups.items():
        by_dataset[ds] = compute_retrieval_metrics(recs, top_k=top_k, ks=ks)

    micro = compute_retrieval_metrics(results, top_k=top_k, ks=ks)

    macro: dict[str, float] = {}
    if by_dataset:
        all_keys = set()
        for m in by_dataset.values():
            all_keys.update(m)
        for key in all_keys:
            vals = [m.get(key, 0.0) for m in by_dataset.values()]
            macro[key] = float(np.mean(vals))

    overall = {f"Micro/{k}": v for k, v in micro.items()}
    overall.update({f"Macro/{k}": v for k, v in macro.items()})
    return overall, by_dataset
