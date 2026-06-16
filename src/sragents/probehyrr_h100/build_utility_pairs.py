"""Stage 8 — assemble (query, skill) utility pairs (spec §8).

Joins anchor verifications + UCB candidate probe logs with M4 rank metadata and the
skill dictionary, assigns each pair a ``label_category`` and a collapsed
``label_binary`` (1 = should rank high), dropping noisy/uninformative labels.

    python -m sragents.probehyrr_h100.build_utility_pairs \
        --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
        --probe-logs results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
        --m4 results/m4/m4_top50_train.jsonl \
        --skills data/corpus/skills.jsonl \
        --out results/ce_probehyrr_h100/utility_pairs_train.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, write_jsonl


def m4_top1_label(v_no, v_gold, v_m4) -> str:
    """Label for the m4_top1 anchor (spec §8: by v_no, v_gold, v_m4; rank == 1)."""
    if v_m4 == 1:
        return "helpful" if v_no == 0 else "no_load_preferred"
    if v_no == 1:
        return "harmful"
    if v_gold == 1:
        return "strict_false_friend"
    return "safe_but_unneeded"


def _skill_block(corpus: dict, sid: str) -> dict | None:
    s = corpus.get(sid)
    if not (s and s.get("content")):
        return None
    return {"skill_id": sid, "name": s.get("name"), "title": s.get("title"),
            "description": s.get("description"), "content": s["content"]}


def build_split(anchors_path, probe_logs_path, m4_path, out_path) -> dict:
    corpus = load_corpus_dict()
    m4 = {r["qid"]: r for r in iter_jsonl(m4_path)}
    rank_by = {qid: {c["skill_id"]: c["rank_m4"] for c in rec.get("top50", [])}
               for qid, rec in m4.items()}

    # anchors grouped per qid
    anc: dict[str, dict] = {}
    for r in iter_jsonl(anchors_path):
        d = anc.setdefault(r["qid"], {"dataset": r["dataset"], "split": r.get("split")})
        d[r["probe_kind"]] = r

    def rank_of(qid, sid, default=None):
        return rank_by.get(qid, {}).get(sid, default)

    rows: list[dict] = []
    seen: set[tuple] = set()
    counts: Counter = Counter()

    def emit(qid, dataset, split, sid, label, vscore, probe_kind, rank, ucb_arm=None, vtype=None):
        if label in C.DROP_LABELS:
            counts["dropped_label"] += 1
            return
        if (qid, sid) in seen:
            counts["dup_skipped"] += 1
            return
        block = _skill_block(corpus, sid)
        if block is None:
            counts["no_content"] += 1
            return
        seen.add((qid, sid))
        rows.append({
            "instance_id": qid,
            "qid": qid,
            "dataset": dataset,
            "split": split,
            "question": (m4.get(qid) or {}).get("query"),
            "skill_id": sid,
            "skill": block,
            "label_category": label,
            "label_binary": 1 if label in C.POSITIVE_LABELS else 0,
            "verifier_score": vscore,
            "probe_kind": probe_kind,
            "ucb_arm": ucb_arm,
            "rank_m4": rank,
            "verifier_type": vtype,
        })
        counts[f"label::{label}"] += 1
        counts["pairs"] += 1

    # --- anchor pairs ---
    for qid, d in anc.items():
        dataset, split = d["dataset"], d["split"]
        v_no = d.get("no_skill", {}).get("verifier_score")
        gold = d.get("gold_skill")
        v_gold = gold.get("verifier_score") if gold else None
        m1 = d.get("m4_top1")
        v_m4 = m1.get("verifier_score") if m1 else None

        if gold and not gold.get("parse_failed"):
            glabel = "gold_verified_helpful" if v_gold == 1 else "gold_verified_harmful"
            for sid in gold.get("skill_ids", []):
                emit(qid, dataset, split, sid, glabel, v_gold, "gold_skill",
                     rank_of(qid, sid, gold.get("metadata", {}).get("gold_rank_m4")),
                     vtype=gold.get("verifier_type"))
        if m1 and not m1.get("parse_failed"):
            for sid in m1.get("skill_ids", []):
                emit(qid, dataset, split, sid, m4_top1_label(v_no, v_gold, v_m4), v_m4,
                     "m4_top1", rank_of(qid, sid, 1), vtype=m1.get("verifier_type"))

    # --- UCB candidate pairs ---
    if probe_logs_path and Path(probe_logs_path).exists():
        for r in iter_jsonl(probe_logs_path):
            if r.get("parse_failed"):
                counts["candidate_parse_failed"] += 1
            sid = (r.get("skill_ids") or [None])[0]
            if sid is None:
                continue
            emit(r["qid"], r["dataset"], r.get("split"), sid, r["label_category"],
                 r.get("verifier_score"), "ucb_candidate",
                 r.get("metadata", {}).get("rank_m4"), ucb_arm=r.get("ucb_arm"),
                 vtype=r.get("verifier_type"))

    write_jsonl(out_path, rows)
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8 — build utility pairs")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--probe-logs", default=None)
    ap.add_argument("--m4", required=True)
    ap.add_argument("--skills", required=False, help="(corpus loaded via sragents.corpus)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    counts = build_split(args.anchors, args.probe_logs, args.m4, args.out)
    print(f"[pairs] -> {args.out}")
    print(f"  {counts}")


if __name__ == "__main__":
    main()
