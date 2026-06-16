"""Stage 0b — build results/m4/m4_top50_{split}.jsonl from the RRF+KMeans pools.

"M4" == the existing top-100 pool ``results/pool/hybrid_km_alpha30-{ds}.json``.
This is a pure converter (no retrieval): per query it slices the top-50, renames
``bge_rank -> dense_rank``, keeps ``rank/score/bm25_rank/cluster_id/is_gold``,
derives ``gold_rank_m4``/``gold_in_top50``, and reshapes the per-query JSON object
into one JSONL line, partitioned by the data/splits/{split}.jsonl membership.
"""

from __future__ import annotations

import argparse

from sragents.config import PROJECT_ROOT
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, load_json, write_jsonl


def _pool_lookup(dataset: str) -> dict[str, dict]:
    """instance_id -> pool record {instance_id, gold_skill_ids, retrieved[]}."""
    rel = C.M4_POOL_TMPL.format(ds=dataset)
    data = load_json(PROJECT_ROOT / rel)
    return {r["instance_id"]: r for r in data["results"]}


def _candidate(c: dict) -> dict:
    """Project a pool candidate to the m4_top50 schema (rename + sentinel)."""
    return {
        "skill_id": c["skill_id"],
        "rank_m4": c.get("rank"),
        "m4_score": c.get("score"),
        "bm25_rank": c.get("bm25_rank"),         # may be None (absent from BM25 list)
        "dense_rank": c.get("bge_rank"),         # rename bge_rank -> dense_rank; may be None
        "cluster_id": c.get("cluster_id"),
        "cluster_affinity": c.get("cluster_affinity"),
        "is_gold": bool(c.get("is_gold", False)),
    }


def _gold_rank(top50: list[dict], gold: set[str]) -> int | None:
    ranks = [c["rank_m4"] for c in top50 if c["skill_id"] in gold]
    return min(ranks) if ranks else None


def build_split(split: str) -> int:
    """Emit results/m4/m4_top50_{split}.jsonl. Returns rows written."""
    pools: dict[str, dict[str, dict]] = {}
    rows = []
    for q in iter_jsonl(C.split_jsonl_path(split)):
        ds = q["dataset"]
        if ds not in pools:
            pools[ds] = _pool_lookup(ds)
        rec = pools[ds].get(q["qid"])
        if rec is None:
            raise KeyError(f"{q['qid']} not found in M4 pool for {ds}")

        gold = set(q["gold_skill_ids"])
        top50 = [_candidate(c) for c in rec["retrieved"][: C.TOP50]]
        grank = _gold_rank(top50, gold)
        rows.append(
            {
                "qid": q["qid"],
                "dataset": ds,
                "split": split,
                "query": q["question"],
                "gold_skill_ids": list(gold),
                "gold_rank_m4": grank,
                "gold_in_top50": grank is not None,
                "top50": top50,
            }
        )
    return write_jsonl(C.m4_cache_path(split), rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build M4 top-50 cache from RRF+KMeans pools")
    ap.add_argument("--splits", nargs="*", default=C.SPLITS)
    args = ap.parse_args()

    for split in args.splits:
        n = build_split(split)
        print(f"m4_top50 -> {C.m4_cache_path(split)}: {n} queries")


if __name__ == "__main__":
    main()
