"""Stage 5 — build one round of micro-batch UCB candidate probe tasks (spec §5).

For each query that still has probe budget left, choose an arm by UCB (warm-up first),
choose the best candidate inside that arm, and emit a generation task in the exact
schema ``run_batched_generation.py`` consumes (so Stage 6 + caching behave like anchors).

    python -m sragents.probehyrr_h100.build_ucb_tasks \
        --round 1 --budget-per-query 3 \
        --m4 results/m4/m4_top50_train.jsonl \
        --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
        --regimes results/ce_probehyrr_h100/query_regime_labels_train.jsonl \
        --previous-logs results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
        --ucb-state results/ce_probehyrr_h100/state/ucb_state_train.json \
        --out results/ce_probehyrr_h100/tasks/ucb_probe_tasks_round_1_train.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100 import microbatch_ucb as ucb
from sragents.probehyrr_h100.io_utils import cache_key, iter_jsonl, prompt_hash, write_jsonl
from sragents.probehyrr_h100.prompt_templates import build_probe_prompt, skill_contents_for


def _probes_by_qid(previous_logs: str) -> dict[str, set[str]]:
    """qid -> set of skill_ids already probed by UCB in earlier rounds."""
    out: dict[str, set[str]] = defaultdict(set)
    if previous_logs and Path(previous_logs).exists():
        for r in iter_jsonl(previous_logs):
            for sid in r.get("skill_ids", []):
                out[r["qid"]].add(sid)
    return out


def _limit(qids: list[str], regimes: dict, n: int | None) -> list[str]:
    """Stratified cap: first ~n/|datasets| qids per dataset (deterministic)."""
    if not n:
        return qids
    by_ds: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        by_ds[regimes[q]["dataset"]].append(q)
    datasets = sorted(by_ds)
    per, rem = divmod(n, len(datasets))
    keep: list[str] = []
    for i, ds in enumerate(datasets):
        k = per + (1 if i < rem else 0)
        keep.extend(sorted(by_ds[ds])[:k])
    return sorted(keep)


def build_round(round_: int, budget: int, m4_path, regimes_path, previous_logs,
                ucb_state_path, out_path, limit_queries=None, force=False) -> dict:
    corpus = load_corpus_dict()
    m4 = {r["qid"]: r for r in iter_jsonl(m4_path)}
    regimes = {r["qid"]: r for r in iter_jsonl(regimes_path)}
    probed = _probes_by_qid(previous_logs)
    n_probes = {q: len(probed.get(q, set())) for q in regimes}
    state = ucb.load_state(ucb_state_path, split=next(iter(regimes.values()))["split"] if regimes else "train")

    # instances (question text) for prompt building
    split = next(iter(regimes.values()))["split"] if regimes else None
    instances = {r["qid"]: r for r in iter_jsonl(C.split_jsonl_path(split))} if split else {}

    out_path = Path(out_path)
    done_qids: set[str] = set()
    if out_path.exists() and not force:
        done_qids = {r["qid"] for r in iter_jsonl(out_path)}

    qids = _limit(sorted(regimes), regimes, limit_queries)

    working_n: dict[str, int] = defaultdict(int)
    provisional = 0
    rows: list[dict] = []
    counts = {"scheduled": 0, "no_budget": 0, "no_action": 0, "resumed": 0, "arms": defaultdict(int)}

    for qid in qids:
        if qid in done_qids:
            counts["resumed"] += 1
            continue
        if n_probes.get(qid, 0) >= budget:
            counts["no_budget"] += 1
            continue
        rec = m4.get(qid)
        inst = instances.get(qid)
        if rec is None or inst is None:
            counts["no_action"] += 1
            continue

        actions = ucb.candidate_actions(rec, regimes[qid], probed.get(qid, set()),
                                        corpus, seed=f"{qid}::round_{round_}")
        if not actions:
            counts["no_action"] += 1
            continue

        arm = ucb.select_arm(list(actions), state, working_n, t=state["global_t"] + provisional)
        cand = actions[arm]
        working_n[arm] += 1
        provisional += 1
        counts["arms"][arm] += 1

        sid = cand["skill_id"]
        contents = skill_contents_for([sid], corpus)
        system, user = build_probe_prompt(inst, contents)
        ph = prompt_hash(system, user)
        gen_cfg = C.generation_config(inst["dataset"])
        rows.append({
            "task_id": f"ucb::round_{round_}::{split}::{qid}::{arm}::{sid}",
            "split": split,
            "qid": qid,
            "dataset": inst["dataset"],
            "probe_kind": "ucb_candidate",
            "ucb_arm": arm,
            "skill_ids": [sid],
            "system": system,
            "user": user,
            "prompt_hash": ph,
            "cache_key": cache_key(qid=qid, skill_ids=[sid], prompt_hash_val=ph,
                                   max_new_tokens=gen_cfg["max_new_tokens"]),
            "generation_config": gen_cfg,
            "metadata": {
                "round": round_,
                "rank_m4": cand.get("rank_m4"),
                "m4_score": cand.get("m4_score"),
                "bm25_rank": cand.get("bm25_rank"),
                "dense_rank": cand.get("dense_rank"),
                "cluster_id": cand.get("cluster_id"),
                "is_gold": False,
                "gold_rank_m4": rec.get("gold_rank_m4"),
                "ucb_arm": arm,
                "priority_score": cand.get("m4_score"),
            },
        })
        counts["scheduled"] += 1

    write_jsonl(out_path, rows, append=not force and out_path.exists())
    counts["arms"] = dict(counts["arms"])
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 — build one UCB round of probe tasks")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--budget-per-query", type=int, default=C.BUDGET_PER_QUERY)
    ap.add_argument("--m4", required=True)
    ap.add_argument("--anchors", required=False, help="(unused here; kept for CLI symmetry)")
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--previous-logs", default=None)
    ap.add_argument("--ucb-state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-queries", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    counts = build_round(args.round, args.budget_per_query, args.m4, args.regimes,
                         args.previous_logs, args.ucb_state, args.out,
                         limit_queries=args.limit_queries, force=args.force)
    print(f"[ucb-tasks] round={args.round} -> {args.out}")
    print(f"  {counts}")


if __name__ == "__main__":
    main()
