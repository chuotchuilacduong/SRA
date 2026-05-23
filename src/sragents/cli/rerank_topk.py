"""``sragents rerank-topk`` — CE rerank with packing + optional MaxP.

Reads a pool JSON (paper_exact or extended), reranks each query's top-K
with a cross-encoder, and writes a RetrievalResults JSON with full
provenance (per-retriever ranks/scores, ``is_gold``, ``candidate_recall_miss``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.data_config import load_data_config
from sragents.retrieve import compute_retrieval_metrics
from sragents.retrieve.chunking import MaxPChunker
from sragents.retrieve.cross_rerank import CrossEncoderReranker
from sragents.retrieve.metrics import compute_metrics_by_dataset
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults
from sragents.retrieve.skill_packer import SkillPacker


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rerank-topk", help="Cross-encoder rerank a pool JSON (packing + MaxP)",
        description="Second-stage CE reranking with field-tagged packing and "
                    "optional MaxP chunking. Output is the same RetrievalResults "
                    "schema, with per-retriever provenance preserved.",
    )
    p.add_argument("--pool", type=Path, required=True,
                   help="Pool JSON (from `sragents build-pool` or any "
                        "RetrievalResults-shaped file).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True,
                   help="Instances JSON used to recover the question text.")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="HF id or local path of the cross-encoder.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=100,
                   help="Pool size to rerank per query.")
    p.add_argument("--packing", default="field_tagged",
                   help="title_only | title_description | "
                        "title_description_content | field_tagged | "
                        "field_tagged_maxp_chunks")
    p.add_argument("--include-tools", action="store_true")
    p.add_argument("--max-content-chars", type=int, default=1800)
    p.add_argument("--maxp", action="store_true",
                   help="Enable MaxP chunking on long skills.")
    p.add_argument("--maxp-chunk-size", type=int, default=800)
    p.add_argument("--maxp-stride", type=int, default=400)
    p.add_argument("--maxp-long-skill-chars", type=int, default=1200)
    p.add_argument("--eval-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "eval.yaml")
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.add_argument("--max-instances", type=int, default=None)
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.pool, "pool")
    require_exists(args.instances, "instances")
    data_cfg = load_data_config(args.data_config)
    eval_cfg = yaml.safe_load(args.eval_config.read_text())

    pool = json.loads(args.pool.read_text())
    pool_records = pool["results"]
    if args.max_instances:
        pool_records = pool_records[: args.max_instances]
    instances = json.loads(args.instances.read_text())
    instances_by_id = {data_cfg.instance_query_id(i): i for i in instances}
    corpus_dict = load_corpus_dict(args.corpus) if args.corpus else load_corpus_dict()

    packer = SkillPacker(
        mode=args.packing,
        include_tools=args.include_tools,
        max_content_chars=args.max_content_chars,
    )
    chunker = (
        MaxPChunker(
            chunk_size=args.maxp_chunk_size,
            stride=args.maxp_stride,
            long_skill_chars=args.maxp_long_skill_chars,
        )
        if args.maxp or args.packing == "field_tagged_maxp_chunks"
        else None
    )

    reranker = CrossEncoderReranker(
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        packer=packer,
        chunker=chunker,
        corpus=corpus_dict,
    )

    records: list[RetrievalRecord] = []
    t0 = time.time()
    for rec in pool_records:
        inst = instances_by_id.get(rec["instance_id"])
        if inst is None:
            continue
        question = data_cfg.instance_query(inst)
        cands = rec.get("retrieved", [])[: args.top_k]
        reranked = reranker.rerank(
            question, cands, top_k=args.top_k, batch_size=args.batch_size,
        )
        gold_set = set(rec.get("gold_skill_ids") or [])
        miss = bool(gold_set) and not (gold_set & {x["skill_id"] for x in reranked})
        for entry in reranked:
            entry["is_gold"] = entry["skill_id"] in gold_set
            entry["candidate_recall_miss"] = miss
        records.append(RetrievalRecord(
            instance_id=rec["instance_id"],
            gold_skill_ids=rec["gold_skill_ids"],
            retrieved=reranked,
        ))
    wall = time.time() - t0

    eval_records = [{"gold_skill_ids": r.gold_skill_ids,
                     "instance_id": r.instance_id,
                     "retrieved": r.retrieved} for r in records]
    overall, by_dataset = compute_metrics_by_dataset(
        eval_records, instances_by_id,
        dataset_field=data_cfg.f_domain, top_k=args.top_k,
        ks={k: tuple(v) for k, v in eval_cfg["cutoffs"].items()},
    )
    micro = compute_retrieval_metrics(
        eval_records, top_k=args.top_k,
        ks={k: tuple(v) for k, v in eval_cfg["cutoffs"].items()},
    )
    # We keep `micro` as the primary `metrics` block (back-compat with the
    # existing schema); macro + per-dataset live under `extra`.
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in micro.items()))

    out = RetrievalResults(
        retriever=f"ce_rerank_{Path(args.model).name}",
        top_k=args.top_k,
        corpus_size=pool.get("metadata", {}).get("corpus_size", len(corpus_dict)),
        records=records,
        metrics=micro,
        dataset=pool.get("metadata", {}).get("dataset"),
        extra={
            "source_pool": str(args.pool),
            "model": args.model,
            "device": args.device,
            "packing": args.packing,
            "maxp": chunker is not None,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "wall_seconds": wall,
            "n_queries": len(records),
            "macro": {k: v for k, v in overall.items() if k.startswith("Macro/")},
            "by_dataset": by_dataset,
        },
    )
    out.dump(args.output)
    print(f"Saved: {args.output}  wall={wall:.1f}s")

    csv_path = args.output.with_name(args.output.stem + "_by_dataset.csv")
    _write_by_dataset_csv(csv_path, by_dataset)
    print(f"Per-dataset CSV: {csv_path}")


def _write_by_dataset_csv(path: Path, by_dataset: dict[str, dict[str, float]]) -> None:
    if not by_dataset:
        path.write_text("dataset\n")
        return
    metric_keys = sorted({k for m in by_dataset.values() for k in m})
    rows = [",".join(["dataset"] + metric_keys)]
    for ds in sorted(by_dataset):
        m = by_dataset[ds]
        rows.append(",".join([ds] + [f"{m.get(k, 0):.6f}" for k in metric_keys]))
    path.write_text("\n".join(rows) + "\n")
