#!/usr/bin/env python3
"""Enhanced retrieval pipeline: 3-way RRF fusion of BM25 + TF-IDF + LinearRAG.

After empirical testing, ms-marco cross-encoder hurts performance on SRA-Bench's
symbolic content (logic/math/code). Instead, this pipeline uses Reciprocal Rank
Fusion across THREE complementary retrieval signals:

  1. BM25 (sparse, lexical, exact-match emphasis)
  2. TF-IDF (sparse, normalized, smooth term weighting)
  3. LinearRAG (graph-based, entity propagation + dense embedding)

The three rankers capture different aspects: BM25 boosts rare exact terms,
TF-IDF balances term frequency, LinearRAG captures semantic+entity structure.
RRF (Cormack et al. 2009) fuses ranks without score normalization.

Pipeline:
  1. BM25 retrieve top-100
  2. TF-IDF retrieve top-100
  3. Load cached LinearRAG top-50
  4. 3-way RRF fusion (k=60) -> top-50 final
  5. Compute Recall@k, nDCG@k
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
from sragents.retrieve.tfidf import TfidfRetriever
from sragents.retrieve.fusion import multi_rrf_merge

TOP_K = 50
SPARSE_TOP_K = 100  # candidates from each sparse retriever
MAX_CHARS = 3000
RESULTS_DIR = "results/retrieval_all"
OUTPUT_DIR = "results/retrieval_enhanced"
INSTANCES_DIR = "data/bench/instances"
CORPUS_PATH = "data/bench/corpus/corpus.json"


def build_corpus_text():
    """Build the same passage texts as LinearRAG (description + content)."""
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


def compute_metrics(results, instances):
    gold_lookup = {
        inst.get("instance_id") or inst.get("id"):
        inst.get("skill_annotations") or inst.get("gold_skill_ids") or []
        for inst in instances
    }

    metrics = {}
    for k in [1, 5, 10, 50]:
        rec_sum = 0
        ndcg_sum = 0
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
        if valid > 0:
            metrics[f"Recall@{k}"] = rec_sum / valid
            metrics[f"nDCG@{k}"] = ndcg_sum / valid
    return metrics


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Build BM25 + TF-IDF indices once ---
    print(f"Loading corpus...", flush=True)
    t0 = time.time()
    corpus_ids, corpus_texts = build_corpus_text()
    print(f"  {len(corpus_ids)} skills in {time.time() - t0:.1f}s", flush=True)

    print(f"\nBuilding BM25 index...", flush=True)
    t0 = time.time()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_ids, corpus_texts)
    print(f"BM25 ready in {(time.time() - t0):.1f}s", flush=True)

    print(f"\nBuilding TF-IDF index...", flush=True)
    t0 = time.time()
    tfidf = TfidfRetriever()
    tfidf.build_index(corpus_ids, corpus_texts)
    print(f"TF-IDF ready in {(time.time() - t0):.1f}s", flush=True)

    # --- Process each dataset ---
    for ds in datasets:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {ds}", flush=True)
        print("=" * 60, flush=True)

        instances_path = f"{INSTANCES_DIR}/{ds}.json"
        linearrag_path = f"{RESULTS_DIR}/{ds}-linearrag.json"

        if not Path(linearrag_path).exists():
            print(f"[SKIP] LinearRAG result not found: {linearrag_path}", flush=True)
            continue

        instances = json.load(open(instances_path))
        linearrag_data = json.load(open(linearrag_path))

        # Build queries in same order as LinearRAG results
        inst_lookup = {
            inst.get("instance_id") or inst.get("id"): inst for inst in instances
        }
        queries = []
        for r in linearrag_data["results"]:
            inst = inst_lookup.get(r["instance_id"])
            queries.append(inst.get("question") or inst.get("query") or "" if inst else "")

        # --- BM25 retrieve top-100 ---
        print(f"Running BM25 (top-{SPARSE_TOP_K})...", flush=True)
        t0 = time.time()
        bm25_results = bm25.retrieve(queries, top_k=SPARSE_TOP_K)
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        # --- TF-IDF retrieve top-100 ---
        print(f"Running TF-IDF (top-{SPARSE_TOP_K})...", flush=True)
        t0 = time.time()
        tfidf_results = tfidf.retrieve(queries, top_k=SPARSE_TOP_K)
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        # --- 3-way RRF fusion ---
        print(f"3-way RRF fusion (BM25 + TF-IDF + LinearRAG)...", flush=True)
        t0 = time.time()
        enhanced_results = []

        for i, lr_record in enumerate(linearrag_data["results"]):
            bm25_list = [
                {"skill_id": sid, "score": sc, "rank": r + 1}
                for r, (sid, sc) in enumerate(bm25_results[i])
            ]
            tfidf_list = [
                {"skill_id": sid, "score": sc, "rank": r + 1}
                for r, (sid, sc) in enumerate(tfidf_results[i])
            ]
            lr_list = lr_record["retrieved"]
            # Add rank if missing in cached results
            for j, item in enumerate(lr_list):
                if "rank" not in item:
                    item["rank"] = j + 1

            # 3-way RRF fusion
            merged = multi_rrf_merge(
                rankings=[bm25_list, tfidf_list, lr_list],
                weights=[1.0, 1.0, 1.0],  # equal weights
                top_k=TOP_K,
                k_rrf=60,
            )

            enhanced_results.append({
                "instance_id": lr_record["instance_id"],
                "retrieved": merged,
            })

        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        # --- Compute metrics ---
        metrics = compute_metrics(enhanced_results, instances)

        # --- Save output ---
        output_path = f"{OUTPUT_DIR}/{ds}-rrf3way.json"
        output = {
            "metadata": {
                "dataset": ds,
                "retriever": "rrf3way_bm25_tfidf_linearrag",
                "top_k": TOP_K,
                "corpus_size": len(corpus_ids),
                "n_queries": len(enhanced_results),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "extra": {
                    "fusion": "RRF (k=60)",
                    "rankers": ["bm25", "tfidf", "linearrag"],
                    "weights": [1.0, 1.0, 1.0],
                    "bm25_top_k": SPARSE_TOP_K,
                    "tfidf_top_k": SPARSE_TOP_K,
                    "bm25_params": {"k1": 1.5, "b": 0.75},
                },
            },
            "results": enhanced_results,
            "metrics": metrics,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nMetrics for {ds}:", flush=True)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)
        print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
