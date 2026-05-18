#!/usr/bin/env python3
"""Use the official src/sragents/retrieve/hybrid.py round_robin_merge with LinearRAG.

Pipeline:
  1. Run BM25, save to schema-compliant file
  2. Adapt cached LinearRAG to schema (add gold_skill_ids)
  3. Call round_robin_merge() from hybrid.py
  4. Save and report metrics

This validates the official hybrid.py path end-to-end on all 6 datasets.
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragents.corpus import load_corpus_dict
from sragents.retrieve.bm25 import BM25Retriever
from sragents.retrieve.hybrid import round_robin_merge
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults

TOP_K = 50
MAX_CHARS = 3000
INSTANCES_DIR = "data/bench/instances"
CORPUS_PATH = "data/bench/corpus/corpus.json"
LINEARRAG_DIR = "results/retrieval_all"

# Output dirs - keep BM25 + adapted LinearRAG + merged hybrid separate
BM25_DIR = "results/retrieval_bm25"
ADAPTED_LR_DIR = "results/retrieval_linearrag_adapted"
HYBRID_OFFICIAL_DIR = "results/retrieval_hybrid_official"

os.makedirs(BM25_DIR, exist_ok=True)
os.makedirs(ADAPTED_LR_DIR, exist_ok=True)
os.makedirs(HYBRID_OFFICIAL_DIR, exist_ok=True)


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


def save_bm25_results(bm25_results, queries_meta, dataset, corpus_size, output_path):
    """Save BM25 results in schema format."""
    records = []
    for i, (inst_id, gold) in enumerate(queries_meta):
        records.append(RetrievalRecord(
            instance_id=inst_id,
            gold_skill_ids=gold,
            retrieved=[{"skill_id": sid, "score": float(sc)}
                       for sid, sc in bm25_results[i]],
        ))
    rr = RetrievalResults(
        retriever="bm25",
        top_k=TOP_K,
        corpus_size=corpus_size,
        records=records,
        dataset=dataset,
    )
    rr.dump(Path(output_path))


def adapt_linearrag_to_schema(dataset, queries_meta, corpus_size, output_path):
    """Re-save cached LinearRAG with gold_skill_ids field (required by hybrid.py)."""
    src = f"{LINEARRAG_DIR}/{dataset}-linearrag.json"
    if not Path(src).exists():
        return False
    data = json.load(open(src))
    gold_lookup = {iid: gold for iid, gold in queries_meta}

    records = []
    for r in data["results"]:
        iid = r["instance_id"]
        gold = gold_lookup.get(iid, [])
        retrieved = [{"skill_id": x["skill_id"], "score": float(x.get("score", 0.0))}
                     for x in r["retrieved"]]
        records.append(RetrievalRecord(
            instance_id=iid,
            gold_skill_ids=gold,
            retrieved=retrieved,
        ))

    rr = RetrievalResults(
        retriever="linearrag",
        top_k=TOP_K,
        corpus_size=corpus_size,
        records=records,
        dataset=dataset,
    )
    rr.dump(Path(output_path))
    return True


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"
    ]

    # --- Build BM25 once ---
    print(f"Loading corpus...", flush=True)
    t0 = time.time()
    corpus_ids, corpus_texts = build_corpus_text()
    print(f"  {len(corpus_ids)} skills in {time.time() - t0:.1f}s", flush=True)

    print(f"\nBuilding BM25 index...", flush=True)
    t0 = time.time()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_ids, corpus_texts)
    print(f"BM25 ready in {time.time() - t0:.1f}s", flush=True)

    summary = {}

    for ds in datasets:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {ds}", flush=True)
        print("=" * 60, flush=True)

        # Load instances + gold labels
        instances = json.load(open(f"{INSTANCES_DIR}/{ds}.json"))
        inst_lookup = {inst.get("instance_id") or inst.get("id"): inst for inst in instances}

        # Match LinearRAG instance order
        lr_path = f"{LINEARRAG_DIR}/{ds}-linearrag.json"
        if not Path(lr_path).exists():
            print(f"[SKIP] No LinearRAG cache: {lr_path}", flush=True)
            continue
        lr_data = json.load(open(lr_path))

        queries = []
        queries_meta = []  # [(instance_id, gold_skill_ids)]
        for r in lr_data["results"]:
            iid = r["instance_id"]
            inst = inst_lookup.get(iid)
            if not inst:
                continue
            query_text = inst.get("question") or inst.get("query") or ""
            gold = inst.get("skill_annotations") or inst.get("gold_skill_ids") or []
            queries.append(query_text)
            queries_meta.append((iid, gold))

        # --- BM25 retrieve top-50 ---
        print(f"Running BM25 ({len(queries)} queries, top-{TOP_K})...", flush=True)
        t0 = time.time()
        bm25_results = bm25.retrieve(queries, top_k=TOP_K)
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        # Save BM25 in schema format
        bm25_path = f"{BM25_DIR}/{ds}-bm25.json"
        save_bm25_results(bm25_results, queries_meta, ds, len(corpus_ids), bm25_path)

        # Adapt LinearRAG to schema (add gold_skill_ids)
        adapted_lr_path = f"{ADAPTED_LR_DIR}/{ds}-linearrag.json"
        adapt_linearrag_to_schema(ds, queries_meta, len(corpus_ids), adapted_lr_path)

        # --- Call official round_robin_merge ---
        print(f"Running official round_robin_merge() from src/sragents/retrieve/hybrid.py...", flush=True)
        t0 = time.time()
        merged = round_robin_merge(
            file_a=Path(bm25_path),
            file_b=Path(adapted_lr_path),
            top_k=TOP_K,
        )
        print(f"  Done in {time.time() - t0:.1f}s", flush=True)

        # Save merged hybrid
        hybrid_path = f"{HYBRID_OFFICIAL_DIR}/{ds}-hybrid-official.json"
        merged.dump(Path(hybrid_path))

        # Report
        print(f"\nMetrics for {ds}:", flush=True)
        for k, v in merged.metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)
        summary[ds] = merged.metrics

        print(f"Saved: {hybrid_path}", flush=True)

    # --- Final summary table ---
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY (official hybrid.py + LinearRAG)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Dataset':<14} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@50':>8} {'nDCG@10':>10}")
    print('-' * 80)
    for ds, m in summary.items():
        print(f"{ds:<14} {m.get('Recall@1', 0):>8.4f} {m.get('Recall@5', 0):>8.4f} "
              f"{m.get('Recall@10', 0):>8.4f} {m.get('Recall@50', 0):>8.4f} "
              f"{m.get('nDCG@10', 0):>10.4f}")


if __name__ == "__main__":
    main()
