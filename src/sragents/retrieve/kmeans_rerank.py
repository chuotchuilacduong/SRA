"""K-means cluster-aware rerankers for BM25 results.

Two rerankers are provided:

KMeansReranker
    Clusters all skills together.  Centroid affinity is used as the
    semantic signal alongside the normalized BM25 score.

    final_score = alpha * bm25_norm + (1 - alpha) * cluster_affinity_norm

GoldKMeansReranker
    Clusters only gold (non-web) skills so centroids are anchored to the
    "answer" semantic space.  Web/noise skills are then assigned post-hoc
    to their nearest gold centroid.  Scoring uses a propagation formula:

        affinity = cosine(query, centroid[c]) * cosine(skill_emb, centroid[c])

    The first term measures query-cluster relevance; the second measures
    how central the skill is within its cluster.  Their product naturally
    decays as skills move away from the centroid ("propagates outward").

    final_score = alpha * bm25_norm + (1 - alpha) * affinity_norm
"""

import time
from pathlib import Path

import numpy as np


class KMeansReranker:
    """Reranks BM25 candidates using K-means cluster-based semantic affinity.

    Call :meth:`build_index` once over the full corpus, then call
    :meth:`rerank` per query.  Corpus embeddings and cluster assignments
    are cached to ``cache_dir`` so subsequent runs skip re-encoding.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        n_clusters: int = 300,
        alpha: float = 0.5,
        batch_size: int = 256,
        cache_dir: str | None = None,
        random_state: int = 42,
    ):
        self._model_path = model_name_or_path
        self._query_prefix = query_prefix
        self.n_clusters = n_clusters
        self.alpha = alpha
        self._batch_size = batch_size
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._random_state = random_state
        self._model = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading model: {self._model_path}")
            self._model = SentenceTransformer(self._model_path)

    def _cache_key(self) -> str:
        return f"kmeans_k{self.n_clusters}"

    def _cache_paths(self) -> tuple:
        if self._cache_dir is None:
            return None, None, None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key()
        return (
            self._cache_dir / f"{key}_emb.npy",
            self._cache_dir / f"{key}_labels.npy",
            self._cache_dir / f"{key}_centroids.npy",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Encode corpus and fit K-means.  Results are cached to ``cache_dir``."""
        self._corpus_ids = corpus_ids
        self._id_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(corpus_ids)}

        emb_path, label_path, centroid_path = self._cache_paths()

        if (
            emb_path is not None
            and emb_path.exists()
            and label_path.exists()
            and centroid_path.exists()
        ):
            print(f"  Loading cached K-means index ({self._cache_key()})...")
            self._cluster_labels = np.load(label_path)
            self._centroids = np.load(centroid_path)
            print(f"  Loaded: {len(corpus_ids)} skills, {self.n_clusters} clusters")
            return

        self._load_model()

        print(f"  Encoding corpus ({len(corpus_texts)} docs)...", flush=True)
        t0 = time.time()
        corpus_emb = self._model.encode(
            corpus_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        print(f"  Encoded in {time.time() - t0:.1f}s")

        print(f"  Fitting K-means (K={self.n_clusters})...", end=" ", flush=True)
        t0 = time.time()
        from sklearn.cluster import KMeans
        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self._random_state,
            n_init="auto",
        )
        km.fit(corpus_emb)
        self._cluster_labels = km.labels_

        # Normalize centroids so dot product == cosine similarity
        raw = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-9)
        self._centroids = raw / norms
        print(f"{time.time() - t0:.1f}s")

        if emb_path is not None:
            np.save(emb_path, corpus_emb)
            np.save(label_path, self._cluster_labels)
            np.save(centroid_path, self._centroids)
            print(f"  Cached to {self._cache_dir}")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[tuple[str, float]]:
        """Return candidates re-sorted by blended BM25 + cluster affinity score.

        Args:
            query: Raw query string (prefix is applied internally).
            candidates: List of ``{"skill_id": str, "score": float}`` dicts,
                ordered by BM25 descending.

        Returns:
            ``[(skill_id, final_score), ...]`` sorted by descending final score.
        """
        if not candidates:
            return []

        self._load_model()

        q_emb: np.ndarray = self._model.encode(
            [self._query_prefix + query],
            normalize_embeddings=True,
        )[0]

        # BM25 scores, min-max normalized within this candidate set
        bm25 = np.array([c["score"] for c in candidates], dtype=np.float32)
        lo, hi = bm25.min(), bm25.max()
        bm25_norm = (bm25 - lo) / (hi - lo) if hi > lo else np.ones_like(bm25)

        # Cluster affinity: cosine(query, centroid[cluster of skill])
        aff = np.empty(len(candidates), dtype=np.float32)
        for j, c in enumerate(candidates):
            idx = self._id_to_idx.get(c["skill_id"])
            if idx is None:
                aff[j] = 0.0
            else:
                aff[j] = float(q_emb @ self._centroids[self._cluster_labels[idx]])

        # Normalize affinity from its natural [-1,1] range to [0,1]
        alo, ahi = aff.min(), aff.max()
        aff_norm = (aff - alo) / (ahi - alo) if ahi > alo else np.full_like(aff, 0.5)

        final = self.alpha * bm25_norm + (1.0 - self.alpha) * aff_norm
        order = np.argsort(final)[::-1]
        return [(candidates[i]["skill_id"], float(final[i])) for i in order]


class GoldKMeansReranker:
    """Gold-anchored K-means reranker.

    K-means is fit **only on gold (non-web) skills** so cluster centroids
    are anchored to the semantic space of actual answers.  Web/noise skills
    are then assigned post-hoc to their nearest gold centroid without
    influencing it.

    Scoring uses a propagation formula that jointly measures:
      - how relevant the cluster is to the query
      - how central the skill is within that cluster

    This naturally boosts skills near the centroid ("answer core") over
    peripheral ones that happen to land in the same cluster.

    affinity = cosine(q, centroid[c]) * cosine(skill_emb, centroid[c])
    final    = alpha * bm25_norm + (1 - alpha) * affinity_norm
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        n_clusters: int = 100,
        alpha: float = 0.5,
        batch_size: int = 256,
        cache_dir: str | None = None,
        random_state: int = 42,
    ):
        self._model_path = model_name_or_path
        self._query_prefix = query_prefix
        self.n_clusters = n_clusters
        self.alpha = alpha
        self._batch_size = batch_size
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._random_state = random_state
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading model: {self._model_path}")
            self._model = SentenceTransformer(self._model_path)

    def _cache_paths(self):
        if self._cache_dir is None:
            return None, None, None, None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"gold_kmeans_k{self.n_clusters}"
        return (
            self._cache_dir / f"{key}_emb.npy",       # all corpus embeddings
            self._cache_dir / f"{key}_labels.npy",     # cluster label per skill
            self._cache_dir / f"{key}_centroids.npy",  # gold-anchored centroids
            self._cache_dir / f"{key}_meta.npy",       # n_gold scalar
        )

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Encode corpus, cluster gold skills, assign web skills post-hoc."""
        self._corpus_ids = corpus_ids
        self._id_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(corpus_ids)}

        gold_mask = np.array(
            [not cid.startswith("web_") for cid in corpus_ids], dtype=bool
        )
        n_gold = int(gold_mask.sum())
        n_web = len(corpus_ids) - n_gold
        print(f"  Corpus split: {n_gold} gold + {n_web} web skills")

        emb_path, label_path, centroid_path, meta_path = self._cache_paths()

        if (
            emb_path is not None
            and all(p.exists() for p in (emb_path, label_path, centroid_path, meta_path))
        ):
            print(f"  Loading cached gold K-means index (K={self.n_clusters})...")
            self._corpus_emb = np.load(emb_path)
            self._cluster_labels = np.load(label_path)
            self._centroids = np.load(centroid_path)
            print(f"  Loaded: {len(corpus_ids)} skills, {self.n_clusters} clusters")
            return

        self._load_model()

        print(f"  Encoding corpus ({len(corpus_texts)} docs)...", flush=True)
        t0 = time.time()
        corpus_emb = self._model.encode(
            corpus_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        print(f"  Encoded in {time.time() - t0:.1f}s")

        gold_emb = corpus_emb[gold_mask]

        # Clamp K to at most the number of gold skills
        k = min(self.n_clusters, n_gold)
        if k < self.n_clusters:
            print(f"  Note: clamping K from {self.n_clusters} to {k} (only {n_gold} gold skills)")

        print(f"  Fitting K-means on {n_gold} gold skills (K={k})...", end=" ", flush=True)
        t0 = time.time()
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=self._random_state, n_init="auto")
        km.fit(gold_emb)

        # Normalize centroids for cosine similarity
        raw = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True).clip(min=1e-9)
        centroids = raw / norms
        print(f"{time.time() - t0:.1f}s")

        # Assign ALL skills (gold + web) to nearest centroid
        # scores shape: (n_corpus, k)
        print("  Assigning all skills to nearest centroid...", end=" ", flush=True)
        t0 = time.time()
        sim = corpus_emb @ centroids.T          # (N, k), cosine since both normalized
        cluster_labels = np.argmax(sim, axis=1).astype(np.int32)
        print(f"{time.time() - t0:.1f}s")

        self._corpus_emb = corpus_emb
        self._cluster_labels = cluster_labels
        self._centroids = centroids

        if emb_path is not None:
            np.save(emb_path, corpus_emb)
            np.save(label_path, cluster_labels)
            np.save(centroid_path, centroids)
            np.save(meta_path, np.array([n_gold]))
            print(f"  Cached to {self._cache_dir}")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[tuple[str, float]]:
        """Return candidates re-sorted by gold-anchored propagation score.

        affinity(skill) = cosine(query, centroid[c]) * cosine(skill_emb, centroid[c])

        Skills near the centroid of a query-relevant cluster score highest;
        peripheral skills in the same cluster score progressively lower.
        """
        if not candidates:
            return []

        self._load_model()

        q_emb: np.ndarray = self._model.encode(
            [self._query_prefix + query],
            normalize_embeddings=True,
        )[0].astype(np.float32)

        # BM25 scores, min-max normalized
        bm25 = np.array([c["score"] for c in candidates], dtype=np.float32)
        lo, hi = bm25.min(), bm25.max()
        bm25_norm = (bm25 - lo) / (hi - lo) if hi > lo else np.ones_like(bm25)

        # Propagation affinity: query-cluster sim × skill-centroid sim
        aff = np.empty(len(candidates), dtype=np.float32)
        for j, c in enumerate(candidates):
            idx = self._id_to_idx.get(c["skill_id"])
            if idx is None:
                aff[j] = 0.0
            else:
                cluster_id = self._cluster_labels[idx]
                centroid = self._centroids[cluster_id]
                query_cluster_sim = float(q_emb @ centroid)
                skill_centroid_sim = float(self._corpus_emb[idx] @ centroid)
                # Product: high only when BOTH query and skill are close to centroid
                aff[j] = query_cluster_sim * skill_centroid_sim

        alo, ahi = aff.min(), aff.max()
        aff_norm = (aff - alo) / (ahi - alo) if ahi > alo else np.full_like(aff, 0.5)

        final = self.alpha * bm25_norm + (1.0 - self.alpha) * aff_norm
        order = np.argsort(final)[::-1]
        return [(candidates[i]["skill_id"], float(final[i])) for i in order]
