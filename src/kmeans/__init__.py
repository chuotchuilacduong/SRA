"""M4-v2 — Cluster-Aware RRF Reranker.

Larger-pool, soft-cluster, reliability-calibrated, adaptive-weight reranking on
top of an RRF (BM25+BGE) candidate pool. See ``IMPLEMENTATION_NOTES.md`` for the
decision log and ``docs/kmeans/implementation_m4_v2 (1).md`` for the full spec.

This package reuses ``sragents`` for metrics / corpus / schema and only adds the
M4-v2-specific scoring, ablation, and reporting logic.
"""

__all__ = ["config", "io", "embeddings", "rrf_pool", "cluster_stats",
           "prf", "soft_cluster", "adaptive_alpha", "scorer",
           "evaluate", "ablation_runner", "report_writer"]
