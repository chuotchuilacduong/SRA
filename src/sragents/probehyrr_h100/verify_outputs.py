"""Stage 3/7 — CPU-parallel deterministic verification of generations.

Wraps the repo's per-dataset ``sragents.evaluate.evaluate(raw_output, instance)``
(pure-CPU, no GPU/network) in a ProcessPoolExecutor. For each generation it emits
a verification record with ``verifier_score`` (0/1), ``parsed_answer``,
``parse_failed`` and a ``verifier_type`` label, plus probe provenance so Stage 4
(query regimes) and the label builder can pivot by (qid, probe_kind).

The original bench instance (``data/bench/instances/{ds}.json``) is passed to the
verifier verbatim so behavior is identical to the prior art (logicbench MCQA needs
``instance['question']``; medcalc needs ``eval_data`` tolerances).

    python -m sragents.probehyrr_h100.verify_outputs \
        --generations results/ce_probehyrr_h100/generations/anchor_outputs_train.jsonl \
        --out         results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
        --workers 8 --summary
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sragents.config import INSTANCES_DIR
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, load_json, write_jsonl


def _verifier_type(dataset: str, result: dict) -> str:
    """Descriptive verifier family + subtype (for quality reporting / down-weighting)."""
    if dataset == "medcalcbench":
        return f"medcalc_{result.get('output_type', 'decimal')}"
    if dataset == "logicbench":
        return "logic_mcqa" if result.get("task_type") == "MCQA" else "logic_bqa"
    if dataset == "theoremqa":
        return f"theoremqa_{result.get('answer_type', 'float')}"
    if dataset == "champ":
        return f"champ_{result.get('match_type', 'none')}"
    return dataset


def _verify_one(args: tuple[dict, dict]) -> dict:
    """Top-level (picklable) worker: verify one generation row.

    ``gen`` is the generation record; ``instance`` is the original bench instance.
    """
    from sragents.evaluate import evaluate  # imported in worker to keep parent light

    gen, instance = args
    dataset = gen["dataset"]
    raw = gen.get("raw_output", "") or ""
    result = evaluate(raw, instance)
    parsed = result.get("extracted_answer", "")
    parse_failed = not (isinstance(parsed, str) and parsed.strip()) or not raw.strip()
    return {
        "task_id": gen["task_id"],
        "qid": gen["qid"],
        "dataset": dataset,
        "split": gen.get("split"),
        "probe_kind": gen["probe_kind"],
        "skill_ids": gen.get("skill_ids", []),
        "parsed_answer": parsed,
        "verifier_score": int(bool(result.get("correct"))),
        "parse_failed": bool(parse_failed),
        "verifier_type": _verifier_type(dataset, result),
        "truncated": gen.get("finish_reason") == "length",
        "num_output_tokens": gen.get("num_output_tokens"),
        "metadata": gen.get("metadata", {}),
    }


def _load_instances(datasets: set[str]) -> dict[str, dict]:
    """instance_id -> bench instance, for the datasets present in the generations."""
    by_id: dict[str, dict] = {}
    for ds in datasets:
        for inst in load_json(INSTANCES_DIR / f"{ds}.json"):
            by_id[inst["instance_id"]] = inst
    return by_id


def run(generations: str, out: str, workers: int, force: bool) -> list[dict]:
    out_path = Path(out)
    gens = list(iter_jsonl(generations))

    done = set()
    if out_path.exists() and not force:
        done = {r["task_id"] for r in iter_jsonl(out_path)}
    todo = [g for g in gens if g["task_id"] not in done]
    print(f"[verify] generations={len(gens)} done={len(done)} todo={len(todo)}")
    if not todo:
        return list(iter_jsonl(out_path)) if out_path.exists() else []

    instances = _load_instances({g["dataset"] for g in todo})
    jobs = [(g, instances[g["qid"]]) for g in todo]

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for rec in ex.map(_verify_one, jobs, chunksize=32):
            records.append(rec)

    write_jsonl(out_path, records, append=not force)
    print(f"[verify] wrote {len(records)} records -> {out_path}")
    return list(iter_jsonl(out_path))


def summarize(records: list[dict]) -> None:
    """Print the §17 pilot gate metrics."""
    n = len(records)
    if not n:
        print("[summary] no records")
        return
    pf = sum(r["parse_failed"] for r in records)
    tr = sum(bool(r["truncated"]) for r in records)
    by_kind: dict[str, list[int]] = defaultdict(list)
    by_ds: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_kind[r["probe_kind"]].append(r["verifier_score"])
        by_ds[r["dataset"]].append(r["verifier_score"])

    print("\n==== pilot gate summary ====")
    print(f"records: {n}")
    print(f"parse_failure: {pf}/{n} = {pf/n:.3%}   (gate <5%)")
    print(f"truncated(finish=length): {tr}/{n} = {tr/n:.3%}")
    print("verifier_score mean by probe_kind:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:12s} n={len(v):5d}  acc={sum(v)/len(v):.3f}")
    print("verifier_score mean by dataset:")
    for k, v in sorted(by_ds.items()):
        print(f"  {k:14s} n={len(v):5d}  acc={sum(v)/len(v):.3f}")
    # quick oracle-gap signal: anchors only
    print("============================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU-parallel deterministic verifier")
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print pilot gate metrics")
    ap.add_argument("--report-only", action="store_true",
                    help="don't verify; just summarize an existing --out file")
    args = ap.parse_args()

    if args.report_only:
        summarize(list(iter_jsonl(args.out)))
        return

    records = run(args.generations, args.out, args.workers, args.force)
    if args.summary:
        summarize(records)


if __name__ == "__main__":
    main()
