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


# ---------------------------------------------------------------------------
# Utility-aware metrics (UCB-ProbeHYRR validation)
# ---------------------------------------------------------------------------
#
# These adjudicate the UCB-ProbeHYRR vs CE-HYRR comparison. Standard retrieval
# metrics (above) can give a FALSE NEGATIVE here: a utility-aware reranker may
# demote a true-gold skill the generator cannot actually use (v_gold=0) and
# promote a helpful non-gold skill, which *lowers* Recall/nDCG while *raising*
# task utility. So the comparison must be decided on the metrics below, not on
# Recall@k / nDCG@k.
#
# Data contract:
#   results: same shape as compute_retrieval_metrics — a *reranked* ranking,
#       each record {instance_id, gold_skill_ids, retrieved: [{skill_id, ...}]}.
#   utility_labels: ground-truth counterfactual labels from the held-out probe
#       test set (Gate 3). Maps
#           instance_id -> {
#               "v_no":  int (0/1),          # V(q, G(q))
#               "v_gold": int|None (0/1),    # V(q, G(q+gold)), None if no gold
#               "skills": { skill_id: {"v": int, "u": int, "label": str} },
#           }
#       Only skills that were actually probed appear in "skills"; everything
#       else is "unknown". Each metric therefore reports a companion
#       "<Metric>/coverage" = fraction of the queries (or ranked cells) on which
#       the metric had ground truth, so a reader knows its support.


def _util_of(ul: dict, sid: str):
    """Return utility int for (query-label-block ``ul``, skill ``sid``), or
    None if that skill was never probed (unknown ground truth)."""
    rec = ul.get("skills", {}).get(sid)
    return None if rec is None else rec.get("u")


def _v_of(ul: dict, sid: str):
    rec = ul.get("skills", {}).get(sid)
    return None if rec is None else rec.get("v")


def compute_utility_metrics(
    results: list[dict],
    utility_labels: dict[str, dict],
    ks: tuple[int, ...] = (10, 50),
) -> dict[str, float]:
    """Macro-averaged utility-aware metrics over reranked records.

    Args:
        results: Reranked rankings (see module-level contract).
        utility_labels: instance_id -> counterfactual label block.
        ks: cutoffs for CS-Gold@K (and the false-friend window is fixed at 10).

    Returns:
        Flat dict. Headline metrics + a ``/coverage`` sibling each:
          - ``Utility@1``      : mean 1[u(top1_ranked) > 0]; higher better.
          - ``FalseFriend@10`` : mean fraction of probed top-10 with u <= 0;
                                 lower better (semantically plausible but useless).
          - ``HarmfulExposure``: P(top1 breaks a skill-free-correct query | v_no=1);
                                 lower better.
          - ``CS-Gold@K``      : P(top1 has u>0 | some gold in ranked top-K);
                                 higher better — "conditional success given gold@K".
    """
    util1, util1_cov = [], 0
    ff_frac, ff_cov = [], 0
    harm, harm_cov = [], 0
    csg = {k: [] for k in ks}
    csg_cov = {k: 0 for k in ks}
    n = 0

    for r in results:
        ul = utility_labels.get(r["instance_id"])
        if ul is None:
            continue
        n += 1
        ranked = [e["skill_id"] for e in r.get("retrieved", [])]
        gold = set(r.get("gold_skill_ids") or [])

        # --- Utility@1 -----------------------------------------------------
        if ranked:
            u_top1 = _util_of(ul, ranked[0])
            if u_top1 is not None:
                util1_cov += 1
                util1.append(1.0 if u_top1 > 0 else 0.0)

        # --- FalseFriend@10 ------------------------------------------------
        top10 = ranked[:10]
        probed = [s for s in top10 if _util_of(ul, s) is not None]
        if probed:
            ff_cov += 1
            bad = sum(1 for s in probed if _util_of(ul, s) <= 0)
            ff_frac.append(bad / len(probed))

        # --- HarmfulExposure (only queries the model gets right skill-free)-
        if ul.get("v_no") == 1 and ranked:
            v_top1 = _v_of(ul, ranked[0])
            if v_top1 is not None:
                harm_cov += 1
                harm.append(1.0 if v_top1 == 0 else 0.0)

        # --- CS-Gold@K -----------------------------------------------------
        for k in ks:
            if gold & set(ranked[:k]):
                u_top1 = _util_of(ul, ranked[0]) if ranked else None
                if u_top1 is not None:
                    csg_cov[k] += 1
                    csg[k].append(1.0 if u_top1 > 0 else 0.0)

    out: dict[str, float] = {}
    denom = max(1, n)
    out["Utility@1"] = _mean(util1)
    out["Utility@1/coverage"] = util1_cov / denom
    out["FalseFriend@10"] = _mean(ff_frac)
    out["FalseFriend@10/coverage"] = ff_cov / denom
    out["HarmfulExposure"] = _mean(harm)
    out["HarmfulExposure/coverage"] = harm_cov / denom
    for k in ks:
        out[f"CS-Gold@{k}"] = _mean(csg[k])
        out[f"CS-Gold@{k}/coverage"] = csg_cov[k] / denom
    out["n_evaluable"] = float(n)
    return out


def compute_utility_metrics_by_dataset(
    results: list[dict],
    utility_labels: dict[str, dict],
    instances_by_id: dict[str, dict],
    dataset_field: str = "dataset",
    ks: tuple[int, ...] = (10, 50),
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Per-dataset + macro/micro split for the utility-aware metrics.

    Mirrors :func:`compute_metrics_by_dataset` so callers can print utility
    metrics alongside retrieval metrics with identical plumbing. Go/no-go
    decisions should use the macro line restricted to strong-verifier
    datasets (toolqa/logicbench/medcalcbench/bigcodebench).
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        inst = instances_by_id.get(r["instance_id"])
        if inst is None:
            continue
        ds = inst.get(dataset_field) or "unknown"
        groups.setdefault(ds, []).append(r)

    by_dataset = {
        ds: compute_utility_metrics(recs, utility_labels, ks=ks)
        for ds, recs in groups.items()
    }
    micro = compute_utility_metrics(results, utility_labels, ks=ks)

    macro: dict[str, float] = {}
    if by_dataset:
        all_keys = {k for m in by_dataset.values() for k in m}
        for key in all_keys:
            macro[key] = float(np.mean([m.get(key, 0.0) for m in by_dataset.values()]))

    overall = {f"Micro/{k}": v for k, v in micro.items()}
    overall.update({f"Macro/{k}": v for k, v in macro.items()})
    return overall, by_dataset
