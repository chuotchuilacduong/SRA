"""Retrieval metrics: Recall@K, nDCG@K, MRR.

All metrics consume ``ranked_ids`` (highest-first) and ``gold_ids`` (set).
"""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = sum(1 for sid in ranked_ids[:k] if sid in gold)
    return hits / len(gold)


def ndcg_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for i, sid in enumerate(ranked_ids[:k]):
        if sid in gold:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: list[str], gold_ids: Iterable[str], k: int | None = None) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    limit = k if k is not None else len(ranked_ids)
    for i, sid in enumerate(ranked_ids[:limit]):
        if sid in gold:
            return 1.0 / (i + 1)
    return 0.0


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}
