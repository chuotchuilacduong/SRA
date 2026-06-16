"""Stage 4 — derive per-query reward regimes (spec §4).

Merge ``anchor_verified_{split}.jsonl`` (no_skill / gold_skill / m4_top1 probes)
with ``m4_top50_{split}.jsonl`` to label, for each query, the five regime flags
used by the UCB scheduler (Stage 5) and the group builder (Stage 9).

    python -m sragents.probehyrr_h100.query_regimes \
        --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
        --m4      results/m4/m4_top50_train.jsonl \
        --out     results/ce_probehyrr_h100/query_regime_labels_train.jsonl
"""

from __future__ import annotations

import argparse

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, write_jsonl


def _score_maps(anchor_rows: list[dict]) -> tuple[dict, dict]:
    """For one qid's anchor rows: (by_probe_kind, by_skillset) -> verifier_score."""
    by_kind: dict[str, int] = {}
    by_skillset: dict[tuple, int] = {}
    for r in anchor_rows:
        by_kind[r["probe_kind"]] = r["verifier_score"]
        by_skillset[tuple(sorted(r.get("skill_ids", [])))] = r["verifier_score"]
    return by_kind, by_skillset


def derive_split(anchors_path, m4_path) -> list[dict]:
    m4 = {r["qid"]: r for r in iter_jsonl(m4_path)}

    by_qid: dict[str, list[dict]] = {}
    for r in iter_jsonl(anchors_path):
        by_qid.setdefault(r["qid"], []).append(r)

    out: list[dict] = []
    for qid, rows in by_qid.items():
        rec = m4.get(qid)
        if rec is None:
            continue  # query has no M4 cache entry; cannot place in any regime
        by_kind, by_skillset = _score_maps(rows)

        v_no = by_kind.get("no_skill")
        v_gold = by_kind.get("gold_skill")  # None if query has no gold skills
        # m4_top1 may have been deduped into gold_skill (when gold == m4 top1);
        # recover its score by the single-skill set, falling back to probe_kind.
        top50 = rec.get("top50", [])
        m4_top1_id = top50[0]["skill_id"] if top50 else None
        v_m4 = by_skillset.get((m4_top1_id,)) if m4_top1_id is not None else None
        if v_m4 is None:
            v_m4 = by_kind.get("m4_top1")

        gold_rank_m4 = rec.get("gold_rank_m4")
        gold_in_top50 = bool(rec.get("gold_in_top50", False))

        out.append({
            "qid": qid,
            "dataset": rows[0]["dataset"],
            "split": rows[0].get("split"),
            "v_no": v_no,
            "v_gold": v_gold,
            "v_m4": v_m4,
            "gold_rank_m4": gold_rank_m4,
            "gold_in_top50": gold_in_top50,
            # regime flags (§4) — exact formulas
            "no_load_opportunity": (v_no == 1),
            "harmful_m4_top1": (v_no == 1 and v_m4 == 0),
            "need_external": (v_no == 0 and v_gold == 1),
            "m4_oracle_gap": (v_gold == 1 and v_m4 == 0),
            "gold_absent_top50": (gold_rank_m4 is None or gold_rank_m4 > C.TOP50),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 — derive query regimes")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--m4", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = derive_split(args.anchors, args.m4)
    write_jsonl(args.out, rows)

    flags = ["no_load_opportunity", "harmful_m4_top1", "need_external",
             "m4_oracle_gap", "gold_absent_top50"]
    counts = {f: sum(int(r[f]) for r in rows) for f in flags}
    print(f"[regimes] {len(rows)} queries -> {args.out}")
    print(f"  flag counts: {counts}")


if __name__ == "__main__":
    main()
