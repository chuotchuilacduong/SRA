#!/usr/bin/env python3
"""RRF complementary fusion: BM25 + Dense (BGE) only — top-100.

Reuses cached Dense BGE rankings (top-100) from results/retrieval_dense/
and rebuilds BM25 to top-100 fresh (cheap). Fuses via Reciprocal Rank
Fusion (Cormack et al. 2009) and computes metrics at @1/@10/@50/@100.
"""
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragents.corpus import load_corpus_dict
from sragents.retrieve.bm25 import BM25Retriever
from sragents.retrieve.fusion import multi_rrf_merge

TOP_K = 100
POOL_K = 100
MAX_CHARS = 3000
RESULTS_DIR_DENSE = "results/retrieval_dense"
INSTANCES_DIR = "data/bench/instances"
CORPUS_PATH = "data/bench/corpus/corpus.json"
OUTPUT_DIR = "results/retrieval_rrf_bm25_dense"

METRIC_KS = [1, 10, 50, 100]


def build_corpus_text():
    corpus = json.load(open(CORPUS_PATH))
    cdict = load_corpus_dict()
    corpus_ids = []
    corpus_texts = []
    for skill in corpus:
        sid = skill["skill_id"]
        s = cdict.get(sid)
        if s is None:
            text = skill.get("content", "")
        else:
            parts = [s.get("description") or "", s.get("content") or ""]
            text = "\n".join(p for p in parts if p)
        corpus_ids.append(sid)
        corpus_texts.append(text[:MAX_CHARS])
    return corpus_ids, corpus_texts


def compute_metrics(results, instances, ks=METRIC_KS):
    gold_lookup = {
        inst.get("instance_id") or inst.get("id"):
        inst.get("skill_annotations") or inst.get("gold_skill_ids") or []
        for inst in instances
    }
    metrics = {}
    for k in ks:
        rec_sum = 0.0
        ndcg_sum = 0.0
        valid = 0
        for r in results:
            gold = gold_lookup.get(r["instance_id"], [])
            if not gold:
                continue
            retrieved_ids = [x["skill_id"] for x in r["retrieved"][:k]]
            gold_set = set(gold)
            rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
            rec_sum += rec
            dcg = sum(
                1.0 / math.log2(i + 2)
                for i, sid in enumerate(retrieved_ids)
                if sid in gold_set
            )
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
            ndcg = dcg / ideal if ideal > 0 else 0
            ndcg_sum += ndcg
            valid += 1
        if valid:
            metrics[f"Recall@{k}"] = rec_sum / valid
            metrics[f"nDCG@{k}"] = ndcg_sum / valid
    return metrics


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading corpus...", flush=True)
    t0 = time.time()
    corpus_ids, corpus_texts = build_corpus_text()
    print(f"  {len(corpus_ids)} skills in {time.time() - t0:.1f}s", flush=True)

    print("\nBuilding BM25 index...", flush=True)
    t0 = time.time()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_ids, corpus_texts)
    print(f"BM25 ready in {time.time() - t0:.1f}s", flush=True)

    summary = {}

    for ds in datasets:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {ds}", flush=True)
        print("=" * 60, flush=True)

        instances_path = f"{INSTANCES_DIR}/{ds}.json"
        dense_path = f"{RESULTS_DIR_DENSE}/{ds}-dense-bge.json"

        if not Path(dense_path).exists():
            print(f"[SKIP] No Dense cache: {dense_path}", flush=True)
            continue

        instances = json.load(open(instances_path))
        dense_data = json.load(open(dense_path))

        inst_lookup = {
            inst.get("instance_id") or inst.get("id"): inst for inst in instances
        }
        queries = []
        for r in dense_data["results"]:
            inst = inst_lookup.get(r["instance_id"])
            queries.append(
                inst.get("question") or inst.get("query") or ""
                if inst else ""
            )

        print(f"BM25 retrieve (top-{POOL_K})...", flush=True)
        t0 = time.time()
        bm25_results = bm25.retrieve(queries, top_k=POOL_K)
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        print(f"RRF fusion (BM25 + Dense), top-{TOP_K}...", flush=True)
        t0 = time.time()
        fused_results = []

        for i, dense_record in enumerate(dense_data["results"]):
            bm25_list = [
                {"skill_id": sid, "score": float(sc), "rank": r + 1}
                for r, (sid, sc) in enumerate(bm25_results[i])
            ]
            dense_list = [
                {"skill_id": item["skill_id"],
                 "score": item.get("score"),
                 "rank": item.get("rank", j + 1)}
                for j, item in enumerate(dense_record["retrieved"])
            ]
            merged = multi_rrf_merge(
                [bm25_list, dense_list],
                weights=[1.0, 1.0],
                top_k=TOP_K,
                k_rrf=60,
            )
            fused_results.append({
                "instance_id": dense_record["instance_id"],
                "gold_skill_ids": dense_record.get("gold_skill_ids", []),
                "retrieved": merged,
            })

        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        metrics = compute_metrics(fused_results, instances)

        output_path = f"{OUTPUT_DIR}/{ds}-rrf-bm25-dense.json"
        with open(output_path, "w") as f:
            json.dump({
                "metadata": {
                    "dataset": ds,
                    "retriever": "rrf_bm25_dense",
                    "top_k": TOP_K,
                    "corpus_size": len(corpus_ids),
                    "n_queries": len(fused_results),
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "extra": {
                        "fusion": "RRF (k=60)",
                        "rankers": ["bm25", "dense_bge"],
                        "weights": [1.0, 1.0],
                        "pool_per_ranker": POOL_K,
                        "bm25_params": {"k1": 1.5, "b": 0.75},
                        "dense_model": "BAAI/bge-base-en-v1.5",
                    },
                },
                "results": fused_results,
                "metrics": metrics,
            }, f, indent=2)

        summary[ds] = metrics
        print(f"\nMetrics for {ds}:", flush=True)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)
        print(f"Saved: {output_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: RRF (BM25 + Dense) — top-100", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Dataset':<14}" + "".join(
        f" {'R@' + str(k):>9}" for k in METRIC_KS
    ) + "".join(f" {'nDCG@' + str(k):>10}" for k in METRIC_KS)
    print(header)
    print("-" * len(header))
    for ds, m in summary.items():
        row = f"{ds:<14}"
        for k in METRIC_KS:
            row += f" {m.get(f'Recall@{k}', 0):>9.4f}"
        for k in METRIC_KS:
            row += f" {m.get(f'nDCG@{k}', 0):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
