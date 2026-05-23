"""``sragents build-pool`` — build a candidate pool for reranking.

Two tracks:

* ``paper_exact`` — BM25 top-K. Single retriever, no fusion.
* ``extended``    — RRF of BM25 + BGE, each at top-K.

Each candidate carries provenance fields (``bm25_rank/score``, ``bge_rank/score``,
``rrf_rank/score``) so downstream reranking and hard-negative mining can
read where a candidate came from without re-running retrieval.

Queries with no gold in the pool are kept with ``candidate_recall_miss=true``
so the evaluator can audit them, *never* silently dropped.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus, skill_text
from sragents.data_config import load_data_config
from sragents.retrieve import compute_retrieval_metrics, get
from sragents.retrieve.fusion import multi_rrf_merge
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults

_TRACKS = ("paper_exact", "extended")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-pool", help="Build a CE rerank candidate pool (BM25 / BM25+BGE+RRF)",
        description="Generate a top-K candidate pool per query (paper_exact or extended track) "
                    "and write a retrieval JSON with full per-retriever provenance.",
    )
    p.add_argument("--track", choices=_TRACKS, required=True)
    p.add_argument("--dataset", required=True,
                   help="Dataset name (e.g. theoremqa). Resolved against "
                        "configs/data.yaml; pass --instances to override.")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Override corpus JSON path")
    p.add_argument("--instances", type=Path, default=None,
                   help="Override instances JSON path")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSON path (RetrievalResults schema)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Pool size per query (default from configs/retrieval.yaml)")
    p.add_argument("--retrieval-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "retrieval.yaml")
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Optional cap (for smoke tests)")
    p.add_argument("--bge-model", default=None,
                   help="Override BGE model path (default from configs/retrieval.yaml)")
    p.set_defaults(func=run)


def _build_query(instance: dict, data_cfg) -> str:
    """Same query construction as ``sragents retrieve``: prompts + question.

    We re-use :func:`sragents.cli.retrieve._build_query` semantics by
    importing on demand to avoid a hard dep when prompts are unregistered
    (some plugin datasets may not have one — fall back to question only).
    """
    try:
        from sragents.cli.retrieve import _build_query as _bq
        return _bq(instance)
    except Exception:
        return data_cfg.instance_query(instance)


def _bm25_run(corpus_ids, corpus_texts, queries, top_k, bm25_cfg):
    retriever = get("bm25", k1=bm25_cfg["k1"], b=bm25_cfg["b"])
    retriever.build_index(corpus_ids, corpus_texts)
    return retriever.retrieve(queries, top_k)


def _bge_run(corpus_ids, corpus_texts, queries, top_k, bge_cfg, model_override=None):
    retriever = get("bge", model_path=model_override or bge_cfg["model_path"],
                    batch_size=bge_cfg["batch_size"])
    retriever.build_index(corpus_ids, corpus_texts)
    return retriever.retrieve(queries, top_k)


def run(args) -> None:
    data_cfg = load_data_config(args.data_config)
    cfg = yaml.safe_load(args.retrieval_config.read_text())
    track_cfg = cfg["tracks"][args.track]
    top_k = args.top_k or track_cfg["pool_top_k"]
    branch_k = max(track_cfg["branch_top_k"], top_k)

    corpus_path = args.corpus or data_cfg.corpus_path
    require_exists(corpus_path, "corpus")
    instances_path = args.instances or (data_cfg.instances_dir / f"{args.dataset}.json")
    require_exists(instances_path, "instances")

    corpus = load_corpus(corpus_path)
    corpus_ids = [s["skill_id"] for s in corpus]
    corpus_texts = [skill_text(s) for s in corpus]
    instances = json.loads(Path(instances_path).read_text())
    if args.max_instances:
        instances = instances[: args.max_instances]

    queries: list[str] = []
    instance_ids: list[str] = []
    gold_by_id: dict[str, list[str]] = {}
    inst_by_id: dict[str, dict] = {}
    for inst in instances:
        qid = data_cfg.instance_query_id(inst)
        gold = data_cfg.instance_gold(inst)
        if not gold:
            # No gold: still surface the query in the pool so we don't
            # silently drop it; downstream metrics skip it but the eval
            # report records candidate_recall_miss.
            pass
        queries.append(_build_query(inst, data_cfg))
        instance_ids.append(qid)
        gold_by_id[qid] = gold
        inst_by_id[qid] = inst

    branches = list(track_cfg["branches"])
    print(f"build-pool track={args.track} dataset={args.dataset} "
          f"branches={branches} top_k={top_k} branch_k={branch_k}", flush=True)

    branch_runs: dict[str, list[list[tuple[str, float]]]] = {}
    timings: dict[str, float] = {}

    if "bm25" in branches:
        t0 = time.time()
        branch_runs["bm25"] = _bm25_run(corpus_ids, corpus_texts, queries, branch_k, cfg["bm25"])
        timings["bm25"] = time.time() - t0
    if "bge" in branches:
        t0 = time.time()
        branch_runs["bge"] = _bge_run(
            corpus_ids, corpus_texts, queries, branch_k, cfg["bge"],
            model_override=args.bge_model,
        )
        timings["bge"] = time.time() - t0

    rrf_k = cfg["rrf"]["k_rrf"]
    records: list[RetrievalRecord] = []
    for i, qid in enumerate(instance_ids):
        gold = gold_by_id[qid]
        gold_set = set(gold)

        # Per-branch rank/score lookups
        rank_lookup: dict[str, dict[str, tuple[int, float]]] = {b: {} for b in branches}
        for branch in branches:
            for r, (sid, score) in enumerate(branch_runs[branch][i], start=1):
                rank_lookup[branch][sid] = (r, float(score))

        if args.track == "paper_exact":
            # BM25 ranking is the pool ranking.
            ordered = [(sid, sc) for sid, sc in branch_runs["bm25"][i]][:top_k]
            ranked = [{"skill_id": sid, "score": float(sc), "rank": j + 1}
                      for j, (sid, sc) in enumerate(ordered)]
        else:
            ranked_lists = [
                [{"skill_id": sid, "score": float(score)}
                 for sid, score in branch_runs[b][i]]
                for b in branches
            ]
            fused = multi_rrf_merge(ranked_lists, top_k=top_k, k_rrf=rrf_k)
            ranked = fused  # already has rank+score

        enriched: list[dict] = []
        for entry in ranked:
            sid = entry["skill_id"]
            row = {
                "skill_id": sid,
                "score": float(entry["score"]),
                "rank": int(entry["rank"]),
                "is_gold": sid in gold_set,
            }
            for b in branches:
                if sid in rank_lookup[b]:
                    r, sc = rank_lookup[b][sid]
                    row[f"{b}_rank"] = r
                    row[f"{b}_score"] = sc
            if args.track == "extended":
                row["rrf_rank"] = entry["rank"]
                row["rrf_score"] = float(entry["score"])
            enriched.append(row)

        if gold:
            pool_skills = {e["skill_id"] for e in enriched}
            miss = not (pool_skills & gold_set)
        else:
            miss = True
        for e in enriched:
            e["candidate_recall_miss"] = miss

        records.append(RetrievalRecord(
            instance_id=qid,
            gold_skill_ids=gold,
            retrieved=enriched,
        ))

    eval_records = [{"gold_skill_ids": r.gold_skill_ids, "retrieved": r.retrieved}
                    for r in records]
    metrics = compute_retrieval_metrics(eval_records, top_k=top_k)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    extra = {
        "track": args.track,
        "branches": branches,
        "branch_top_k": branch_k,
        "rrf_k": rrf_k,
        "timings_seconds": timings,
        "n_queries_no_gold": sum(1 for v in gold_by_id.values() if not v),
        "n_recall_miss": sum(1 for r in records
                             if r.retrieved and r.retrieved[0].get("candidate_recall_miss")),
    }
    out = RetrievalResults(
        retriever=f"pool_{args.track}",
        top_k=top_k,
        corpus_size=len(corpus),
        records=records,
        metrics=metrics,
        dataset=args.dataset,
        extra=extra,
    )
    out.dump(args.output)
    print(f"Saved: {args.output}")
