"""Quality report for a built split (spec §17 gates) — run after the full build.

Summarizes the UCB probe-log label distribution, arm statistics, and final group-type
mix, then checks the §17 acceptance gates. Pilot is skipped in this build, so this is the
post-hoc sanity check on the real data.

    python -m sragents.probehyrr_h100.report_quality --split train
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, load_json


def report(split: str) -> dict:
    logs_path = C.ucb_probe_logs_path(split)
    state_path = C.ucb_state_path(split)
    groups_path = C.utility_train_groups_path(split)

    labels: Counter = Counter()
    n_logs = pf = tr = 0
    for r in iter_jsonl(logs_path) if Path(logs_path).exists() else []:
        labels[r["label_category"]] += 1
        n_logs += 1
        pf += int(bool(r.get("parse_failed")))
        tr += int(bool(r.get("truncated")))

    arm_stats = load_json(state_path)["arm_stats"] if Path(state_path).exists() else {}

    gtypes: Counter = Counter()
    n_groups = pos = 0
    for g in iter_jsonl(groups_path) if Path(groups_path).exists() else []:
        gtypes[g["group_type"]] += 1
        n_groups += 1
        pos += int(g["num_positive"] > 0)

    amb = labels.get("ambiguous", 0) + labels.get("unverifiable", 0)
    gates = {
        "parse_failure_lt_5pct": (pf / n_logs < 0.05) if n_logs else None,
        # Truncation gate: a high cut-off rate silently corrupts labels (the verifier
        # grabs a wrong intermediate token instead of the cut-off final answer). This was
        # the v0 defect that all the other gates missed — see config.MAX_NEW_TOKENS note.
        "truncated_lt_%dpct" % int(C.MAX_TRUNCATED_RATE * 100):
            (tr / n_logs < C.MAX_TRUNCATED_RATE) if n_logs else None,
        "ambiguous_lt_10pct": (amb / n_logs < 0.10) if n_logs else None,
        "harmful_present": labels.get("harmful", 0) > 0,
        "strict_false_friend_present": labels.get("strict_false_friend", 0) > 0,
        "positive_present": (labels.get("helpful", 0) + labels.get("gold_verified_helpful", 0)) > 0,
    }

    summary = {
        "split": split,
        "ucb_probes": n_logs,
        "parse_failure_rate": (pf / n_logs) if n_logs else None,
        "truncated_rate": (tr / n_logs) if n_logs else None,
        "label_distribution": dict(labels),
        "arm_stats": {a: {"n": s["n"], "mean_reward": round(s["mean_reward"], 4)}
                      for a, s in arm_stats.items()},
        "group_types": dict(gtypes),
        "groups_total": n_groups,
        "groups_with_positive": pos,
        "gates": gates,
        "all_gates_pass": all(v for v in gates.values() if v is not None),
    }
    out = C.RUN_ROOT / f"quality_report_{split}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[report] split={split} -> {out}")
    print(json.dumps({k: summary[k] for k in
                      ("ucb_probes", "label_distribution", "group_types", "gates", "all_gates_pass")},
                     ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality report for a built split")
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    report(args.split)


if __name__ == "__main__":
    main()
