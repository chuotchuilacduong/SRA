#!/usr/bin/env python3
"""Compute metrics for already-saved retrieval results."""
import json
import math
import sys
import datetime
from pathlib import Path

DATASETS_DIR = "data/bench/instances"
RESULTS_DIR = "results/retrieval_all"

def compute_metrics(dataset_name):
    instances_path = f"{DATASETS_DIR}/{dataset_name}.json"
    results_path = f"{RESULTS_DIR}/{dataset_name}-linearrag.json"

    if not Path(results_path).exists():
        print(f"[SKIP] {dataset_name}: no results file")
        return

    instances = json.load(open(instances_path))
    data = json.load(open(results_path))

    gold_lookup = {inst.get("instance_id") or inst.get("id"):
                   inst.get("skill_annotations") or inst.get("gold_skill_ids") or inst.get("gold") or []
                   for inst in instances}

    metrics = {}
    for k in [1, 5, 10, 50]:
        rec_sum = 0
        ndcg_sum = 0
        valid = 0
        for r in data["results"]:
            gold = gold_lookup.get(r["instance_id"], [])
            if not gold:
                continue
            retrieved_ids = [x["skill_id"] for x in r["retrieved"][:k]]
            gold_set = set(gold)
            rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
            rec_sum += rec
            dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(retrieved_ids) if sid in gold_set)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
            ndcg = dcg / ideal if ideal > 0 else 0
            ndcg_sum += ndcg
            valid += 1
        if valid > 0:
            metrics[f"Recall@{k}"] = rec_sum / valid
            metrics[f"nDCG@{k}"] = ndcg_sum / valid

    # Update file
    data["metrics"] = metrics
    data["metadata"] = {
        "dataset": dataset_name,
        "retriever": "linearrag",
        "top_k": 50,
        "corpus_size": 26262,
        "n_queries": len(data["results"]),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "extra": {},
    }

    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n=== {dataset_name} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics

def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"
    ]
    for ds in datasets:
        compute_metrics(ds)

if __name__ == "__main__":
    main()
