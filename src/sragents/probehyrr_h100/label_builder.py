"""Stage 7b — label UCB candidate probes and update arm statistics (spec §5.7, §7).

Joins each verified UCB candidate with its query's anchor scores (v_no/v_gold),
assigns a rich label_category via the decision tree below, maps it to a reward, appends
to the cumulative probe log, and updates the (global + per-dataset) UCB arm statistics.

Label decision tree (designed; the spec lists labels + rewards but no explicit tree —
derived from UCB_ProbeHYRR Architecture-v2 §11.2 ``derive_label`` + reward semantics §5.7):

    parse_failed                                 -> unverifiable
    (truncated-but-parsed is kept: use verifier_score, like the anchors)
    utility = v_cand - v_no
    utility > 0                                  -> helpful
    v_no==1 and v_cand==0                        -> harmful
    v_no==1 and v_cand==1                        -> no_load_preferred
    v_no==0 and v_gold==1 and v_cand==0          -> strict_false_friend (rank<=10) | bad_candidate_when_skill_needed
    v_no==0 and v_gold in (0,None) and v_cand==0 -> weak_false_friend (rank<=10) | safe_but_unneeded
    otherwise                                    -> neutral

    python -m sragents.probehyrr_h100.label_builder \
        --verified results/ce_probehyrr_h100/verified/ucb_verified_round_1_train.jsonl \
        --anchors  results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
        --m4       results/m4/m4_top50_train.jsonl \
        --append-log results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
        --update-ucb-state results/ce_probehyrr_h100/state/ucb_state_train.json \
        --round 1
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100 import microbatch_ucb as ucb
from sragents.probehyrr_h100.io_utils import iter_jsonl, write_jsonl


def anchor_scores(anchors_path) -> dict[str, dict]:
    """qid -> {v_no, v_gold} from anchor verifications."""
    out: dict[str, dict] = {}
    for r in iter_jsonl(anchors_path):
        d = out.setdefault(r["qid"], {})
        if r["probe_kind"] == "no_skill":
            d["v_no"] = r["verifier_score"]
        elif r["probe_kind"] == "gold_skill":
            d["v_gold"] = r["verifier_score"]
    return out


def classify(v_cand: int, v_no, v_gold, rank_m4, parse_failed: bool, truncated: bool) -> str:
    # parse_failed = no answer could be extracted -> the verdict is untrustworthy.
    # NOTE: ``truncated`` (finish_reason == "length") is NOT treated as ambiguous:
    # ~59% of the accepted anchor generations are truncated yet still parse to a
    # valid answer, so we use verifier_score directly, consistent with the anchors.
    if parse_failed:
        return "unverifiable"
    v_no = 0 if v_no is None else v_no
    rank = rank_m4 if rank_m4 is not None else (C.TOP50 + 1)
    if v_cand - v_no > 0:
        return "helpful"
    if v_no == 1 and v_cand == 0:
        return "harmful"
    if v_no == 1 and v_cand == 1:
        return "no_load_preferred"
    if v_no == 0 and v_gold == 1 and v_cand == 0:
        return "strict_false_friend" if rank <= C.RANK_FRIEND_CUTOFF else "bad_candidate_when_skill_needed"
    if v_no == 0 and (v_gold in (0, None)) and v_cand == 0:
        return "weak_false_friend" if rank <= C.RANK_FRIEND_CUTOFF else "safe_but_unneeded"
    return "neutral"


def build(round_, verified_path, anchors_path, append_log, ucb_state_path) -> dict:
    anc = anchor_scores(anchors_path)
    verified = list(iter_jsonl(verified_path))
    split = verified[0].get("split") if verified else None

    state = ucb.load_state(ucb_state_path, split=split or "train")

    done: set[str] = set()
    if Path(append_log).exists():
        done = {r["task_id"] for r in iter_jsonl(append_log)}

    rows: list[dict] = []
    label_counts: Counter = Counter()
    for v in verified:
        if v["task_id"] in done:
            continue
        meta = v.get("metadata", {})
        arm = meta.get("ucb_arm")
        rank = meta.get("rank_m4")
        a = anc.get(v["qid"], {})
        label = classify(v["verifier_score"], a.get("v_no"), a.get("v_gold"),
                         rank, v.get("parse_failed", False), bool(v.get("truncated")))
        reward = ucb.reward_for_label(label)
        label_counts[label] += 1

        rows.append({
            "round": round_,
            "task_id": v["task_id"],
            "qid": v["qid"],
            "dataset": v["dataset"],
            "split": v.get("split"),
            "probe_kind": v.get("probe_kind", "ucb_candidate"),
            "skill_ids": v.get("skill_ids", []),
            "ucb_arm": arm,
            "parsed_answer": v.get("parsed_answer"),
            "verifier_score": v["verifier_score"],
            "parse_failed": v.get("parse_failed", False),
            "verifier_type": v.get("verifier_type"),
            "truncated": bool(v.get("truncated")),
            "num_output_tokens": v.get("num_output_tokens"),
            "label_category": label,
            "reward": reward,
            "metadata": meta,
        })
        if arm is not None:
            ucb.update_stats(state, arm, v["dataset"], reward)

    write_jsonl(append_log, rows, append=True)
    state["round"] = round_
    ucb.save_state(ucb_state_path, state)
    return {"labeled": len(rows), "skipped_done": len(verified) - len(rows),
            "labels": dict(label_counts)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7b — label UCB probes + update arm stats")
    ap.add_argument("--verified", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--m4", required=False, help="(rank read from record metadata; kept for symmetry)")
    ap.add_argument("--append-log", required=True)
    ap.add_argument("--update-ucb-state", required=True)
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    stats = build(args.round, args.verified, args.anchors, args.append_log, args.update_ucb_state)
    print(f"[label] round={args.round} -> {args.append_log}")
    print(f"  {stats}")


if __name__ == "__main__":
    main()
