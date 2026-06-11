"""No-LLM candidate/query loading + probe-set assembly.

Importable and fully testable without torch/transformers/GPU — this is the
surface exercised by `run_probe.py --dry-run`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.skill_probe_hidden.config import (
    POOL_TMPL,
    RERANK_TMPL,
    ROOT,
    TOP_K,
)


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_instances(dataset: str) -> dict[str, dict]:
    """instance_id -> instance dict (question, dataset, skill_annotations, eval_data)."""
    path = ROOT / "data" / "bench" / "instances" / f"{dataset}.json"
    return {inst["instance_id"]: inst for inst in _load_json(path)}


def load_test_ids(dataset: str) -> list[str]:
    """Canonical TEST set = order of instance_ids in the CE-HYRR rerank file.

    (The split file's `test` list is a superset; the rerank file is the
    authoritative evaluated set, so we anchor on it.)
    """
    rr = _load_json(ROOT / RERANK_TMPL.format(ds=dataset))
    return [r["instance_id"] for r in rr["results"]]


def pick_queries(dataset: str, n: int, seed: int) -> list[str]:
    """Reproducible sample of `n` test ids (sorted for stable output ordering)."""
    ids = load_test_ids(dataset)
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(n, len(ids))))


def load_method_candidates(dataset: str, method: str) -> dict[str, list[dict]]:
    """instance_id -> ranked retrieved[] for one method.

    M4 = hybrid_km pool (top-100); CE-HYRR = ce-v3 rerank (top-20).
    """
    tmpl = POOL_TMPL if method == "M4" else RERANK_TMPL
    data = _load_json(ROOT / tmpl.format(ds=dataset))
    return {r["instance_id"]: r["retrieved"] for r in data["results"]}


def gold_skill_ids(instance: dict) -> list[str]:
    """All gold skills for a query (may be multiple, e.g. champ)."""
    return list(instance.get("skill_annotations") or [])


def build_probe_set(
    instance: dict,
    retrieved: list[dict],
    top_k: int = TOP_K,
) -> tuple[list[tuple[str, list[str]]], list[dict]]:
    """Return (probe_jobs, candidate_meta) for one (query, method).

    probe_jobs: ordered list of (probe_tag, skill_ids) to run:
        ("no_skill", []), ("gold", [<gold...>]), then one per top-k candidate
        ("cand", [skill_id]).  `gold` is skipped if the query has no gold.
    candidate_meta: top-k candidate records (rank, skill_id, is_gold, score) so
        the JSON can carry retrieval provenance even before utilities are known.
    """
    gold = gold_skill_ids(instance)
    top = retrieved[:top_k]
    candidate_meta = [
        {
            "rank": c.get("rank", i + 1),
            "skill_id": c["skill_id"],
            "is_gold": bool(c.get("is_gold", c["skill_id"] in set(gold))),
            "score": c.get("score"),
        }
        for i, c in enumerate(top)
    ]

    jobs: list[tuple[str, list[str]]] = [("no_skill", [])]
    if gold:
        jobs.append(("gold", list(gold)))
    for c in candidate_meta:
        jobs.append(("cand", [c["skill_id"]]))
    return jobs, candidate_meta


def skill_tag(skill_ids: list[str]) -> str:
    """Stable filename tag for a probe's skill set (drives .npy de-dup)."""
    if not skill_ids:
        return "no_skill"
    if len(skill_ids) == 1:
        return skill_ids[0]
    return "gold__" + "_".join(sorted(skill_ids))
