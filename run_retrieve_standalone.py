#!/usr/bin/env python3
"""Standalone retrieval - uses pre-built cache directly.
Skips redundant insert_text() calls. Builds graph + runs queries.
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from sragents.retrieve._linearrag.core import LinearRAG
from sragents.retrieve._linearrag.config import LinearRAGConfig
from sragents.corpus import load_corpus_dict
from sragents.retrieve._linearrag.utils import compute_mdhash_id


def run_dataset(dataset_name, model, working_dir="import", max_chars=3000, top_k=50):
    """Run retrieval for one dataset using cached data."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print('=' * 60, flush=True)

    # Load instances
    instances_path = f"data/bench/instances/{dataset_name}.json"
    instances = json.load(open(instances_path))
    print(f"Queries: {len(instances)}", flush=True)

    # Build passages (same as adapter does)
    corpus = json.load(open("data/bench/corpus/corpus.json"))
    cdict = load_corpus_dict()
    idx_to_skill_id = {i: skill["skill_id"] for i, skill in enumerate(corpus)}

    passages = []
    for i, skill in enumerate(corpus):
        sid = skill["skill_id"]
        s = cdict.get(sid)
        if s is None:
            text = skill.get("content", "")
        else:
            parts = [s.get("description") or "", s.get("content") or ""]
            text = "\n".join(p for p in parts if p)
        passages.append(f"{i}:{text[:max_chars]}")

    # Save skill_id_map (for adapter compat)
    map_path = f"{working_dir}/bench_full/skill_id_map.json"
    Path(map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, 'w') as f:
        json.dump({str(k): v for k, v in idx_to_skill_id.items()}, f)

    # Setup config
    cfg = LinearRAGConfig(
        dataset_name="bench_full",
        embedding_model=model,
        spacy_model="en_core_web_sm",
        working_dir=working_dir,
        retrieval_top_k=top_k,
        batch_size=8,
        max_workers=1,
        use_vectorized_retrieval=False,
        enable_passage_adjacency=False,
    )

    print("\nInitializing LinearRAG (loads embeddings from cache)...", flush=True)
    t0 = time.time()
    rag = LinearRAG(global_config=cfg)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    print("\nBuilding graph (using cache - no encoding)...", flush=True)
    t0 = time.time()
    rag.index(passages)
    print(f"  Done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Build queries
    queries = [{"question": inst.get("question") or inst.get("query") or "",
                "answer": inst.get("answer", "")} for inst in instances]

    print(f"\nRunning {len(queries)} queries...", flush=True)
    t0 = time.time()
    raw_results = rag.retrieve(queries)
    print(f"  Done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Format output
    import re
    PREFIX_RE = re.compile(r"^(\d+):")
    formatted_results = []
    for inst, raw in zip(instances, raw_results):
        ranked = []
        seen = set()
        for ptext, score in zip(raw["sorted_passage"], raw["sorted_passage_scores"]):
            m = PREFIX_RE.match(ptext.strip())
            if not m:
                continue
            idx = int(m.group(1))
            sid = idx_to_skill_id.get(idx)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            ranked.append([sid, float(score)])
            if len(ranked) >= top_k:
                break
        formatted_results.append({
            "instance_id": inst.get("instance_id") or inst.get("id"),
            "retrieved": [{"skill_id": s, "score": sc, "rank": i+1}
                         for i, (s, sc) in enumerate(ranked)],
        })

    # Save partial output FIRST (before metrics, in case of errors)
    partial_path = f"results/retrieval_all/{dataset_name}-linearrag.json"
    Path(partial_path).parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, 'w') as f:
        json.dump({"metadata": {"retriever": "linearrag", "dataset": dataset_name},
                   "results": formatted_results, "metrics": {}}, f, indent=2)
    print(f"Saved partial: {partial_path}", flush=True)

    # Compute simple metrics (inline, no import)
    import math
    metrics = {}
    gold_lookup = {inst.get("instance_id") or inst.get("id"):
                   inst.get("skill_annotations") or inst.get("gold_skill_ids") or inst.get("gold") or []
                   for inst in instances}

    for k in [1, 5, 10, 50]:
        rec_sum = 0
        ndcg_sum = 0
        valid = 0
        for r in formatted_results:
            gold = gold_lookup.get(r["instance_id"], [])
            if not gold:
                continue
            retrieved_ids = [x["skill_id"] for x in r["retrieved"][:k]]
            gold_set = set(gold)
            # Recall@k
            rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
            rec_sum += rec
            # nDCG@k
            dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(retrieved_ids) if sid in gold_set)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
            ndcg = dcg / ideal if ideal > 0 else 0
            ndcg_sum += ndcg
            valid += 1
        if valid > 0:
            metrics[f"Recall@{k}"] = rec_sum / valid
            metrics[f"nDCG@{k}"] = ndcg_sum / valid

    print(f"\nMetrics: {metrics}", flush=True)

    # Save
    output_path = f"results/retrieval_all/{dataset_name}-linearrag.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "retriever": "linearrag",
            "dataset": dataset_name,
        },
        "results": formatted_results,
        "metrics": metrics,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}", flush=True)
    return metrics


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["champ"]

    print(f"Loading embedding model...", flush=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Model loaded.\n", flush=True)

    all_metrics = {}
    for ds in datasets:
        try:
            m = run_dataset(ds, model)
            all_metrics[ds] = m
        except Exception as e:
            import traceback
            print(f"FAILED for {ds}: {e}", flush=True)
            traceback.print_exc()
            all_metrics[ds] = {"error": str(e)}

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for ds, m in all_metrics.items():
        print(f"{ds}: {m}", flush=True)


if __name__ == "__main__":
    main()
