"""K-means-routed retrieval pool builder.

Pipeline per query:
  1. BGE-encode query (with prefix).
  2. Cosine similarity vs the 300 normalized centroids → pick top-M clusters.
  3. Candidate set = ALL skills inside those M clusters.
  4. Within the candidate set:
     - Score with BGE (cosine query × skill_emb).
     - Score with BM25 (sparse score, full 26k → filter to candidates).
  5. Reciprocal Rank Fusion → top-100.

Saves one pool JSON per dataset and sweeps several M values to find optimal:
  results/pool/kmeans_routed_M{m}-{ds}.json

Also reports Recall@K vs the existing full-RRF extended pool.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sragents.corpus import load_corpus, skill_text
from sragents.retrieve import get
from sragents.retrieve.fusion import multi_rrf_merge
from sragents.retrieve.metrics import compute_retrieval_metrics
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults

DATASETS = ["theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench"]
M_VALUES = [10, 30, 50, 100]
TOP_K_OUT = 100
TOP_K_BM25_FULL = 500     # BM25 candidates before filter
RRF_K = 60

# ---- Load shared resources ----
print("Loading corpus + cached embeddings + clusters...", flush=True)
corpus = load_corpus(Path("data/bench/corpus/corpus.json"))
ids = [s["skill_id"] for s in corpus]
texts = [skill_text(s) for s in corpus]
corpus_size = len(corpus)
sid_to_idx = {sid: i for i, sid in enumerate(ids)}

corpus_emb = np.load("results/bge/corpus_emb.npy")     # (26262, 768) normalized
centroids = np.load("results/clusters_centroids.npy")  # (300, 768) normalized
cluster_index = {int(k): v for k, v in json.loads(
    Path("results/clusters_index.json").read_text()).items()}
clusters = json.loads(Path("results/clusters.json").read_text())
print(f"  centroids {centroids.shape}, clusters {len(cluster_index)}, corpus {corpus_size}",
      flush=True)

# ---- Build BM25 once (re-used for all datasets) ----
print("Building BM25 index (one-shot)...", flush=True)
t0 = time.time()
bm25 = get("bm25")
bm25.build_index(ids, texts)
print(f"  BM25 build = {time.time()-t0:.1f}s", flush=True)

# ---- Load BGE model + query prefix ----
from sentence_transformers import SentenceTransformer
print("Loading BGE model...", flush=True)
bge = SentenceTransformer("BAAI/bge-base-en-v1.5")
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def per_query_pool(
    q_emb: np.ndarray,           # (768,) normalized BGE embedding
    bm25_full_scores: np.ndarray,  # (corpus_size,) BM25 scores for this query
    m: int,
    top_k: int,
    gold_set: set[str],
) -> list[dict]:
    """Build cluster-routed RRF top-K candidates for a single query."""
    # 1. Top-M clusters by centroid similarity.
    centroid_sims = q_emb @ centroids.T              # (300,)
    top_clusters = np.argsort(centroid_sims)[::-1][:m]

    # 2. Candidate skill indices.
    cand_idx_list: list[int] = []
    for c in top_clusters:
        cand_idx_list.extend(cluster_index[int(c)])
    cand_idx = np.unique(np.array(cand_idx_list, dtype=np.int64))

    # 3. BGE rank inside candidate set.
    cand_emb = corpus_emb[cand_idx]
    bge_scores = cand_emb @ q_emb                    # (n_cand,)
    bge_order = np.argsort(bge_scores)[::-1]
    bge_ranked = [(int(cand_idx[i]), float(bge_scores[i])) for i in bge_order]

    # 4. BM25 rank inside candidate set.
    bm25_cand = bm25_full_scores[cand_idx]
    bm25_order = np.argsort(bm25_cand)[::-1]
    bm25_ranked = [(int(cand_idx[i]), float(bm25_cand[i])) for i in bm25_order]

    # 5. RRF fuse.
    bm25_list = [{"skill_id": ids[i]} for i, _ in bm25_ranked]
    bge_list = [{"skill_id": ids[i]} for i, _ in bge_ranked]
    fused = multi_rrf_merge([bm25_list, bge_list], top_k=top_k, k_rrf=RRF_K)

    # Lookup helpers.
    bm25_lookup = {ids[i]: (r + 1, sc) for r, (i, sc) in enumerate(bm25_ranked)}
    bge_lookup = {ids[i]: (r + 1, sc) for r, (i, sc) in enumerate(bge_ranked)}
    enriched = []
    for entry in fused:
        sid = entry["skill_id"]
        row = {
            "skill_id": sid,
            "score": float(entry["score"]),
            "rank": int(entry["rank"]),
            "is_gold": sid in gold_set,
            "rrf_rank": int(entry["rank"]),
            "rrf_score": float(entry["score"]),
        }
        if sid in bm25_lookup:
            r, sc = bm25_lookup[sid]
            row["bm25_rank"] = r; row["bm25_score"] = float(sc)
        if sid in bge_lookup:
            r, sc = bge_lookup[sid]
            row["bge_rank"] = r; row["bge_score"] = float(sc)
        enriched.append(row)
    miss = bool(gold_set) and not (gold_set & {e["skill_id"] for e in enriched})
    for e in enriched:
        e["candidate_recall_miss"] = miss
    return enriched, len(cand_idx)


for ds in DATASETS:
    instances = json.loads(Path(f"data/bench/instances/{ds}.json").read_text())
    queries = [i["question"] for i in instances]
    gold = {i["instance_id"]: i["skill_annotations"] for i in instances}

    # Encode queries once.
    print(f"\n=== {ds} : {len(queries)} queries ===", flush=True)
    t0 = time.time()
    q_embs = bge.encode([BGE_PREFIX + q for q in queries],
                        batch_size=128, show_progress_bar=False,
                        normalize_embeddings=True)
    enc_t = time.time() - t0

    # BM25 full scores for each query.
    t0 = time.time()
    bm25_top_all = bm25.retrieve(queries, top_k=corpus_size)  # full ranking
    bm25_full_score = np.zeros((len(queries), corpus_size), dtype=np.float32)
    for qi, lst in enumerate(bm25_top_all):
        for sid, sc in lst:
            bm25_full_score[qi, sid_to_idx[sid]] = sc
    bm25_t = time.time() - t0
    print(f"  BGE encode {enc_t:.1f}s  BM25 full {bm25_t:.1f}s", flush=True)

    for m in M_VALUES:
        out_path = Path(f"results/pool/kmeans_routed_M{m}-{ds}.json")
        records: list[RetrievalRecord] = []
        avg_cand = 0
        t0 = time.time()
        for qi, inst in enumerate(instances):
            gold_set = set(gold[inst["instance_id"]])
            enriched, n_cand = per_query_pool(
                q_embs[qi], bm25_full_score[qi], m, TOP_K_OUT, gold_set,
            )
            avg_cand += n_cand
            records.append(RetrievalRecord(
                instance_id=inst["instance_id"],
                gold_skill_ids=gold[inst["instance_id"]],
                retrieved=enriched,
            ))
        wall = time.time() - t0
        eval_recs = [{"gold_skill_ids": r.gold_skill_ids, "retrieved": r.retrieved}
                     for r in records]
        metrics = compute_retrieval_metrics(eval_recs, top_k=TOP_K_OUT)
        RetrievalResults(
            retriever=f"pool_kmeans_routed_M{m}",
            top_k=TOP_K_OUT, corpus_size=corpus_size, records=records,
            metrics=metrics, dataset=ds,
            extra={
                "track": "kmeans_routed",
                "m_clusters": m,
                "avg_candidate_size": avg_cand / len(queries),
                "wall_sec_per_query": wall / len(queries),
                "wall_sec_total": wall,
                "rrf_k": RRF_K,
                "bge_encode_sec": enc_t,
                "bm25_full_sec": bm25_t,
            },
        ).dump(out_path)
        print(f"  M={m:3d} avg_cand={avg_cand/len(queries):6.0f}  "
              f"R@1={metrics['Recall@1']*100:5.2f}  R@10={metrics['Recall@10']*100:5.2f}  "
              f"R@100={metrics['Recall@100']*100:5.2f}  wall={wall:.1f}s",
              flush=True)
print("\nALL DONE", flush=True)
