"""Stage 9 — group utility pairs into per-query training groups (spec §9). FINAL artifact.

Each query becomes one listwise group: its candidate skills (anchors + UCB probes) with
binary/category labels, plus a ``group_type`` (G1–G5) derived from the query regime. This
is the file a listwise/utility cross-encoder trainer consumes.

    python -m sragents.probehyrr_h100.build_training_groups \
        --pairs results/ce_probehyrr_h100/utility_pairs_train.jsonl \
        --regimes results/ce_probehyrr_h100/query_regime_labels_train.jsonl \
        --out results/ce_probehyrr_h100/utility_train_groups_train.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from sragents.probehyrr_h100.io_utils import iter_jsonl, write_jsonl

_FLAGS = ["no_load_opportunity", "harmful_m4_top1", "need_external",
          "m4_oracle_gap", "gold_absent_top50"]


def classify_group(regime: dict, has_pos: bool, has_neg: bool) -> str:
    # m4_oracle_gap (gold helps, m4_top1 doesn't) is the sharpest ProbeHYRR signal
    # and is subsumed by need_external (v_no=0) / no_load_opportunity (v_no=1), so it
    # MUST be checked first — otherwise G1/G2 always preempt it and G3 stays empty
    # (spec §9 expects all five group types to be reachable).
    if regime.get("m4_oracle_gap") and has_pos:
        return "G3"  # m4_oracle_gap_group
    if regime.get("need_external") and has_pos:
        return "G1"  # need_external_with_positive
    if regime.get("no_load_opportunity") and has_neg:
        return "G2"  # no_load_risk_group
    if regime.get("gold_absent_top50"):
        return "G4"  # gold_absent_group
    return "G5"      # classification_only_group


def build_split(pairs_path, regimes_path, out_path) -> dict:
    regimes = {r["qid"]: r for r in iter_jsonl(regimes_path)}

    pairs_by_qid: dict[str, list[dict]] = defaultdict(list)
    for p in iter_jsonl(pairs_path):
        pairs_by_qid[p["qid"]].append({
            "skill_id": p["skill_id"],
            "skill": p["skill"],
            "label_category": p["label_category"],
            "label_binary": p["label_binary"],
            "verifier_score": p.get("verifier_score"),
            "rank_m4": p.get("rank_m4"),
            "probe_kind": p.get("probe_kind"),
            "ucb_arm": p.get("ucb_arm"),
        })

    rows: list[dict] = []
    gtypes: Counter = Counter()
    for qid, reg in regimes.items():
        pairs = sorted(pairs_by_qid.get(qid, []),
                       key=lambda x: (x["rank_m4"] if x["rank_m4"] is not None else 10**9))
        has_pos = any(p["label_binary"] == 1 for p in pairs)
        has_neg = any(p["label_binary"] == 0 for p in pairs)
        gtype = classify_group(reg, has_pos, has_neg)
        gtypes[gtype] += 1
        rows.append({
            "instance_id": qid,
            "qid": qid,
            "dataset": reg["dataset"],
            "split": reg.get("split"),
            "group_type": gtype,
            "regime_flags": {f: bool(reg.get(f)) for f in _FLAGS},
            "pairs": pairs,
            "num_positive": sum(p["label_binary"] == 1 for p in pairs),
            "num_negative": sum(p["label_binary"] == 0 for p in pairs),
        })

    write_jsonl(out_path, rows)
    return {"groups": len(rows), "group_types": dict(gtypes),
            "queries_with_pairs": sum(1 for r in rows if r["pairs"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 9 — build grouped training data")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = build_split(args.pairs, args.regimes, args.out)
    print(f"[groups] -> {args.out}")
    print(f"  {stats}")


if __name__ == "__main__":
    main()
