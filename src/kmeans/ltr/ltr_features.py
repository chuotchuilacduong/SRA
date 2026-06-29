"""LTR feature assembly (plan §5.5).

Turns the per-query base table (+ QSC local clustering) into a flat
(query, candidate) feature matrix with grouped columns, one row per candidate in
the RRF top-M pool. Cached per dataset as .npz so train/predict reuse it.

Feature groups (selected per LTR variant): retrieval, m4, a7, qsc, confidence.
"""

from __future__ import annotations

import numpy as np

from ..common import io
from .qsc import qsc_features, LocalClustering

# Per-candidate features sourced from base_table["arrays"].
_ARRAY_FEATS = [
    "rrf_score", "rrf_norm", "rrf_rank", "bm25_score", "bm25_norm", "bm25_rank",
    "bge_score", "bge_norm", "bge_rank", "direct_cosine_q_s",
    "inv_rrf_rank", "inv_bm25_rank", "inv_bge_rank", "missing_bm25_flag", "missing_bge_flag",
    "m4_score", "m4_rank", "m4_affinity", "m4_aff_norm",
    "global_cluster_id", "global_cluster_size", "global_cluster_cohesion",
    "global_cluster_reliability", "prf_score", "prf_rank", "cosine_refined_query_skill",
]
# Query-level features (broadcast to every candidate of the query).
_QUERY_FEATS = [
    "prf_query_shift_norm", "prf_safe_set_size", "prf_safe_set_mean_rrf",
    "rrf_margin_1_10", "rrf_margin_1_5", "bm25_bge_top10_overlap",
    "bm25_bge_top20_overlap", "bm25_bge_top50_overlap",
    "query_cluster_entropy_global", "query_length_tokens",
]
_QSC_FEATS = [
    "qsc_local_affinity", "qsc_local_aff_norm", "qsc_local_prior", "qsc_local_prior_norm",
    "qsc_local_centrality", "qsc_local_cluster_id", "qsc_local_cluster_size",
    "qsc_local_cluster_rank_by_prior", "qsc_local_cluster_rank_by_affinity",
]
# bge_ft (fine-tuned retriever sr-emb-bge-v1) signal. NOT produced by the base-table
# build (needs the model's corpus/query embeddings) — added post-hoc by
# scripts/add_bgeft_features.py into a SEPARATE cache (ltr_features_bgeft/). Kept as a
# distinct group appended LAST in ALL_FEATURES so existing 45-col caches and every prior
# column_indices() selection stay valid; selecting groups WITHOUT "bge_ft" reproduces
# the baseline L6 exactly.
_BGEFT_FEATS = [
    "bgeft_cosine", "bgeft_rank", "bgeft_inv_rank", "bgeft_rank_norm",
    "bgeft_is_top1", "bgeft_margin_top1", "bgeft_z",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "retrieval": ["rrf_score", "rrf_norm", "rrf_rank", "bm25_score", "bm25_norm",
                  "bm25_rank", "bge_score", "bge_norm", "bge_rank", "direct_cosine_q_s",
                  "inv_rrf_rank", "inv_bm25_rank", "inv_bge_rank",
                  "missing_bm25_flag", "missing_bge_flag"],
    "m4": ["m4_score", "m4_rank", "m4_affinity", "m4_aff_norm", "global_cluster_id",
           "global_cluster_size", "global_cluster_cohesion", "global_cluster_reliability"],
    "a7": ["prf_score", "prf_rank", "prf_query_shift_norm", "prf_safe_set_size",
           "prf_safe_set_mean_rrf", "cosine_refined_query_skill"],
    "qsc": list(_QSC_FEATS),
    "confidence": ["rrf_margin_1_10", "rrf_margin_1_5", "bm25_bge_top10_overlap",
                   "bm25_bge_top20_overlap", "bm25_bge_top50_overlap",
                   "query_cluster_entropy_global", "query_length_tokens"],
    "bge_ft": list(_BGEFT_FEATS),
}
ALL_FEATURES = [f for g in ["retrieval", "m4", "a7", "qsc", "confidence", "bge_ft"]
                for f in FEATURE_GROUPS[g]]


def feature_row(e: dict, local: LocalClustering) -> tuple[np.ndarray, np.ndarray]:
    """Build (X_q (n,F), y_q (n,)) for one query from its base entry + local clustering."""
    n = len(e["skill_ids"])
    qf = qsc_features(e, local)
    arrays = e["arrays"]
    qfeats = e["query_feats"]
    gold = set(e["gold_skill_ids"] or [])
    cols = []
    for name in ALL_FEATURES:
        if name in _QSC_FEATS:
            cols.append(qf[name])
        elif name in _QUERY_FEATS:
            cols.append(np.full(n, qfeats[name], dtype=np.float32))
        elif name in _BGEFT_FEATS:
            # placeholder: real values are filled by scripts/add_bgeft_features.py,
            # which needs the sr-emb-bge-v1 embeddings (not available at base-build time).
            cols.append(np.zeros(n, dtype=np.float32))
        else:
            cols.append(np.asarray(arrays[name], dtype=np.float32))
    X = np.stack(cols, axis=1).astype(np.float32)
    y = np.array([1 if s in gold else 0 for s in e["skill_ids"]], dtype=np.int32)
    return X, y


class FeatureAccumulator:
    """Streams per-query feature rows into a packable table (memory-light)."""

    def __init__(self, ds: str):
        self.ds = ds
        self._X, self._y = [], []
        self.group_sizes, self.instance_ids, self.skill_ids, self.gold = [], [], [], []

    def add(self, e: dict, local: LocalClustering) -> None:
        X, y = feature_row(e, local)
        self._X.append(X); self._y.append(y)
        self.group_sizes.append(len(e["skill_ids"]))
        self.instance_ids.append(e["instance_id"])
        self.skill_ids.extend(e["skill_ids"])
        self.gold.append(list(e["gold_skill_ids"] or []))

    def table(self) -> dict:
        return {
            "dataset": self.ds,
            "X": np.concatenate(self._X, axis=0),
            "y": np.concatenate(self._y, axis=0),
            "group_sizes": np.array(self.group_sizes, dtype=np.int64),
            "instance_ids": self.instance_ids,
            "skill_ids": self.skill_ids,
            "gold": self.gold,
            "feature_names": ALL_FEATURES,
        }


def column_indices(groups: list[str]) -> list[int]:
    """Indices into ALL_FEATURES for the union of the given groups."""
    keep = [f for g in groups for f in FEATURE_GROUPS[g]]
    return [ALL_FEATURES.index(f) for f in ALL_FEATURES if f in set(keep)]


def save_features(path, table: dict) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, X=table["X"], y=table["y"], group_sizes=table["group_sizes"],
        instance_ids=np.array(table["instance_ids"], dtype=object),
        skill_ids=np.array(table["skill_ids"], dtype=object),
        feature_names=np.array(table["feature_names"], dtype=object),
        gold=np.array([",".join(g) for g in table["gold"]], dtype=object),
        dataset=table["dataset"])


def load_features(path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "dataset": str(d["dataset"]),
        "X": d["X"], "y": d["y"], "group_sizes": d["group_sizes"],
        "instance_ids": list(d["instance_ids"]),
        "skill_ids": list(d["skill_ids"]),
        "gold": [g.split(",") if g else [] for g in d["gold"]],
        "feature_names": list(d["feature_names"]),
    }
