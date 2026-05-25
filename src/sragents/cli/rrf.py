"""``sragents rrf`` — Reciprocal Rank Fusion of two retrieval result files.

RRF score: score(d) = Σ_i  1 / (k + rank_i(d))   [k=60 by default]

Combines any two retrieval files (e.g. BM25 + Dense) into a single
re-ranked list without requiring score normalization.
"""

import json
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.retrieve import compute_retrieval_metrics
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults, load


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rrf",
        help="Reciprocal Rank Fusion of two retrieval result files (stage 1)",
        description=(
            "Merges two retrieval JSONs using RRF: "
            "score(d) = 1/(k+rank_a) + 1/(k+rank_b).  "
            "Produces a new retrieval JSON in the same schema."
        ),
    )
    p.add_argument("--a", type=Path, required=True,
                   help="First retrieval JSON (e.g. BM25)")
    p.add_argument("--b", type=Path, required=True,
                   help="Second retrieval JSON (e.g. Dense/BGE)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output retrieval JSON (fused)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Output list length per query (default: 50)")
    p.add_argument("--k", type=int, default=60,
                   help="RRF k constant (default: 60)")
    p.set_defaults(func=run)


def _rrf_merge(
    list_a: list[dict],
    list_b: list[dict],
    k: int,
    top_k: int,
) -> list[dict]:
    """Fuse two ranked lists with RRF, return top_k entries."""
    scores: dict[str, float] = {}
    for rank, entry in enumerate(list_a, 1):
        sid = entry["skill_id"]
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    for rank, entry in enumerate(list_b, 1):
        sid = entry["skill_id"]
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [{"skill_id": sid, "score": score} for sid, score in ranked]


def run(args) -> None:
    require_exists(args.a, "a")
    require_exists(args.b, "b")

    data_a = load(args.a)
    data_b = load(args.b)

    map_b = {r["instance_id"]: r["retrieved"] for r in data_b["results"]}

    records_out = []
    skipped = 0
    for ra in data_a["results"]:
        inst_id = ra["instance_id"]
        rb_list = map_b.get(inst_id)
        if rb_list is None:
            skipped += 1
            records_out.append({
                "instance_id": inst_id,
                "gold_skill_ids": ra["gold_skill_ids"],
                "retrieved": ra["retrieved"][: args.top_k],
            })
            continue

        fused = _rrf_merge(ra["retrieved"], rb_list, k=args.k, top_k=args.top_k)
        records_out.append({
            "instance_id": inst_id,
            "gold_skill_ids": ra["gold_skill_ids"],
            "retrieved": fused,
        })

    if skipped:
        print(f"  [warn] {skipped} instances missing from --b; kept --a results")

    metrics = compute_retrieval_metrics(records_out, top_k=args.top_k)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    name_a = data_a["metadata"].get("retriever", args.a.stem)
    name_b = data_b["metadata"].get("retriever", args.b.stem)

    result = RetrievalResults(
        retriever=f"rrf_{name_a}_{name_b}",
        top_k=args.top_k,
        corpus_size=data_a["metadata"].get("corpus_size", 0),
        records=[
            RetrievalRecord(
                instance_id=r["instance_id"],
                gold_skill_ids=r["gold_skill_ids"],
                retrieved=r["retrieved"],
            )
            for r in records_out
        ],
        metrics=metrics,
        dataset=data_a["metadata"].get("dataset"),
        extra={"sources": [str(args.a), str(args.b)], "rrf_k": args.k},
    )
    result.dump(args.output)
    print(f"Saved: {args.output}")
