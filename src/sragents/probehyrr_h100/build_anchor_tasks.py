"""Stage 1 — build the anchor probe task manifest.

For every query emit up to three anchor probes (spec Stage 1, Option A):
    1. no_skill                          (always)
    2. gold_skill  (all gold contents)   (every valid_4 query has gold)
    3. m4_top1     (rank-1 M4 candidate) (always)

Probes that resolve to the same (qid, sorted skill_ids) are deduped by cache_key
so a shared no_skill / gold / m4_top1 (e.g. when m4_top1 *is* the gold skill) is
generated once. Output: results/ce_probehyrr_h100/tasks/anchor_probe_tasks_{split}.jsonl
"""

from __future__ import annotations

import argparse

from sragents.corpus import load_corpus_dict
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import cache_key, iter_jsonl, prompt_hash, write_jsonl
from sragents.probehyrr_h100.prompt_templates import build_probe_prompt, skill_contents_for


def _task_id(probe_kind: str, split: str, qid: str, skill_ids: list[str]) -> str:
    tag = "+".join(skill_ids) if skill_ids else "none"
    return f"anchor::{split}::{qid}::{probe_kind}::{tag}"


def _make_task(q: dict, probe_kind: str, skill_ids: list[str], corpus: dict, meta: dict) -> dict | None:
    """Build one anchor task row, or None if a with-skill probe has no resolvable content."""
    if probe_kind == "no_skill":
        contents = []
    else:
        contents = skill_contents_for(skill_ids, corpus)
        if not contents:
            return None  # skill(s) have empty content -> nothing to probe
    system, user = build_probe_prompt(q, contents)
    ph = prompt_hash(system, user)
    gen_cfg = C.generation_config(q["dataset"])
    return {
        "task_id": _task_id(probe_kind, q["split"], q["qid"], skill_ids),
        "split": q["split"],
        "qid": q["qid"],
        "dataset": q["dataset"],
        "probe_kind": probe_kind,
        "skill_ids": skill_ids,
        "system": system,
        "user": user,
        "prompt_hash": ph,
        "cache_key": cache_key(
            qid=q["qid"], skill_ids=skill_ids, prompt_hash_val=ph,
            max_new_tokens=gen_cfg["max_new_tokens"],
        ),
        "generation_config": gen_cfg,
        "metadata": meta,
    }


def build_split(split: str) -> dict[str, int]:
    corpus = load_corpus_dict()
    m4 = {r["qid"]: r for r in iter_jsonl(C.m4_cache_path(split))}

    rows: list[dict] = []
    seen: set[str] = set()
    counts = {"no_skill": 0, "gold_skill": 0, "m4_top1": 0, "deduped": 0, "skipped": 0}

    for q in iter_jsonl(C.split_jsonl_path(split)):
        rec = m4[q["qid"]]
        top50 = rec["top50"]
        gold_ids = list(q["gold_skill_ids"])
        m4_top1 = top50[0]["skill_id"] if top50 else None

        probes: list[tuple[str, list[str], dict]] = [
            ("no_skill", [], {"rank_m4": None, "is_gold": False, "gold_rank_m4": rec["gold_rank_m4"]}),
        ]
        if gold_ids:
            probes.append(
                ("gold_skill", gold_ids,
                 {"rank_m4": None, "is_gold": True, "gold_rank_m4": rec["gold_rank_m4"]})
            )
        if m4_top1 is not None:
            probes.append(
                ("m4_top1", [m4_top1],
                 {"rank_m4": top50[0]["rank_m4"], "is_gold": top50[0]["is_gold"],
                  "gold_rank_m4": rec["gold_rank_m4"]})
            )

        for probe_kind, skill_ids, meta in probes:
            task = _make_task(q, probe_kind, skill_ids, corpus, meta)
            if task is None:
                counts["skipped"] += 1
                continue
            if task["cache_key"] in seen:
                counts["deduped"] += 1
                continue
            seen.add(task["cache_key"])
            rows.append(task)
            counts[probe_kind] += 1

    write_jsonl(C.anchor_tasks_path(split), rows)
    counts["total"] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build anchor probe task manifest")
    ap.add_argument("--splits", nargs="*", default=C.SPLITS)
    args = ap.parse_args()

    for split in args.splits:
        counts = build_split(split)
        print(f"anchor tasks -> {C.anchor_tasks_path(split)}")
        print(f"  {split}: {counts}")


if __name__ == "__main__":
    main()
