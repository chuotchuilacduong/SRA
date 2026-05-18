#!/usr/bin/env python3
"""Hybrid retrieval: Run BM25 + merge with existing LinearRAG results.
Uses round-robin fusion to combine sparse (BM25) and dense (LinearRAG) signals.
"""
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
from sragents.retrieve.bm25 import BM25Retriever
from sragents.corpus import load_corpus_dict

TOP_K = 50
TOP_K_BM25 = 100
MAX_CHARS = 3000
RESULTS_DIR = "results/retrieval_all"
OUTPUT_DIR = "results/retrieval_hybrid"
INSTANCES_DIR = "data/bench/instances"
CORPUS_PATH = "data/bench/corpus/corpus.json"


def build_corpus_text():
    """Build the same passage texts as LinearRAG."""
    corpus = json.load(open(CORPUS_PATH))
    cdict = load_corpus_dict()
    corpus_ids = []
    corpus_texts = []
    for i, skill in enumerate(corpus):
        sid = skill['skill_id']
        s = cdict.get(sid)
        if s is None:
            text = skill.get('content', '')
        else:
            parts = [s.get('description') or '', s.get('content') or '']
            text = '\n'.join(p for p in parts if p)
        corpus_ids.append(sid)
        corpus_texts.append(text[:MAX_CHARS])
    return corpus_ids, corpus_texts


def round_robin_merge(list_a, list_b, top_k=50):
    """Interleave two ranked lists with dedup. First appearance wins."""
    seen = set()
    merged = []
    max_rank = max(len(list_a), len(list_b))
    for rank in range(max_rank):
        for lst in (list_a, list_b):
            if rank < len(lst):
                sid = lst[rank]['skill_id']
                if sid not in seen:
                    seen.add(sid)
                    merged.append(lst[rank])
                    if len(merged) == top_k:
                        return merged
    return merged


def compute_metrics(results, instances):
    gold_lookup = {inst.get('instance_id') or inst.get('id'):
                   inst.get('skill_annotations') or inst.get('gold_skill_ids') or []
                   for inst in instances}

    metrics = {}
    for k in [1, 5, 10, 50]:
        rec_sum = 0
        ndcg_sum = 0
        valid = 0
        for r in results:
            gold = gold_lookup.get(r['instance_id'], [])
            if not gold:
                continue
            retrieved_ids = [x['skill_id'] for x in r['retrieved'][:k]]
            gold_set = set(gold)
            rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
            rec_sum += rec
            dcg = sum(1.0 / math.log2(i + 2) for i, sid in enumerate(retrieved_ids) if sid in gold_set)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
            ndcg = dcg / ideal if ideal > 0 else 0
            ndcg_sum += ndcg
            valid += 1
        if valid > 0:
            metrics[f'Recall@{k}'] = rec_sum / valid
            metrics[f'nDCG@{k}'] = ndcg_sum / valid
    return metrics


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        'champ', 'theoremqa', 'logicbench', 'toolqa', 'medcalcbench', 'bigcodebench'
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build BM25 once for all datasets
    print(f"Loading corpus...", flush=True)
    t0 = time.time()
    corpus_ids, corpus_texts = build_corpus_text()
    print(f"  {len(corpus_ids)} skills in {time.time()-t0:.1f}s", flush=True)

    print(f"\nBuilding BM25 index...", flush=True)
    t0 = time.time()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_ids, corpus_texts)
    print(f"BM25 ready in {(time.time()-t0)/60:.1f} min", flush=True)

    # Process each dataset
    for ds in datasets:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {ds}", flush=True)
        print('=' * 60, flush=True)

        instances_path = f"{INSTANCES_DIR}/{ds}.json"
        linearrag_path = f"{RESULTS_DIR}/{ds}-linearrag.json"

        if not Path(linearrag_path).exists():
            print(f"[SKIP] LinearRAG result not found: {linearrag_path}", flush=True)
            continue

        instances = json.load(open(instances_path))
        linearrag_data = json.load(open(linearrag_path))

        # Build query list (preserving order from linearrag results)
        # Map instance_id -> question
        inst_lookup = {inst.get('instance_id') or inst.get('id'): inst for inst in instances}
        queries = []
        for r in linearrag_data['results']:
            inst = inst_lookup.get(r['instance_id'])
            if inst:
                queries.append(inst.get('question') or inst.get('query') or '')

        # Run BM25
        print(f"Running BM25 on {len(queries)} queries...", flush=True)
        t0 = time.time()
        bm25_results = bm25.retrieve(queries, top_k=TOP_K_BM25)
        print(f"  Done in {(time.time()-t0):.1f}s", flush=True)

        # Merge BM25 + LinearRAG via round-robin
        hybrid_results = []
        for i, lr_record in enumerate(linearrag_data['results']):
            bm25_list = [{'skill_id': sid, 'score': sc, 'rank': r+1}
                         for r, (sid, sc) in enumerate(bm25_results[i])]
            lr_list = lr_record['retrieved']

            # Add rank if missing
            for j, item in enumerate(lr_list):
                if 'rank' not in item:
                    item['rank'] = j + 1

            merged = round_robin_merge(bm25_list, lr_list, top_k=TOP_K)
            # Re-rank
            for j, item in enumerate(merged):
                item['rank'] = j + 1
            hybrid_results.append({
                'instance_id': lr_record['instance_id'],
                'retrieved': merged,
            })

        # Compute metrics
        metrics = compute_metrics(hybrid_results, instances)

        # Save
        output_path = f"{OUTPUT_DIR}/{ds}-hybrid-bm25-linearrag.json"
        output = {
            'metadata': {
                'dataset': ds,
                'retriever': 'hybrid_bm25_linearrag',
                'top_k': TOP_K,
                'corpus_size': len(corpus_ids),
                'n_queries': len(hybrid_results),
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'extra': {'sources': ['bm25', 'linearrag']},
            },
            'results': hybrid_results,
            'metrics': metrics,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nMetrics for {ds}:", flush=True)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)
        print(f"Saved: {output_path}", flush=True)


if __name__ == '__main__':
    main()
