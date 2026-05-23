"""Compute normalized centroids from cached corpus_emb + clusters.

Saves:
  results/clusters_centroids.npy   (300, 768) float32, L2-normalized
  results/clusters_index.json      {cluster_id: [skill_idx, ...]} for fast lookup
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

corpus_emb = np.load("results/bge/corpus_emb.npy")
ids = json.loads(Path("results/bge/corpus_ids.json").read_text())
clusters = json.loads(Path("results/clusters.json").read_text())

print(f"Corpus: {corpus_emb.shape}, clusters: {len(set(clusters.values()))}")

cluster_idx = np.array([clusters[sid] for sid in ids], dtype=np.int32)
n_clusters = int(cluster_idx.max()) + 1

centroids = np.zeros((n_clusters, corpus_emb.shape[1]), dtype=np.float32)
counts = np.zeros(n_clusters, dtype=np.int32)
for i, c in enumerate(cluster_idx):
    centroids[c] += corpus_emb[i]
    counts[c] += 1
centroids /= np.maximum(counts[:, None], 1)
norms = np.linalg.norm(centroids, axis=1, keepdims=True)
centroids = centroids / np.maximum(norms, 1e-12)

np.save("results/clusters_centroids.npy", centroids)

cluster_index: dict[int, list[int]] = {}
for i, c in enumerate(cluster_idx):
    cluster_index.setdefault(int(c), []).append(i)

Path("results/clusters_index.json").write_text(json.dumps(cluster_index))
print(f"Saved centroids ({centroids.shape}), index, "
      f"sizes min={counts.min()} median={int(np.median(counts))} max={counts.max()}")
