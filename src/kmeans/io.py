"""IO + artifact loading for M4-v2.

Centralizes every on-disk read so the rest of the package never hard-codes a
path. Validates row-index alignment between ``corpus_emb``, ``corpus_ids`` and
``clusters`` on load (Step 1 of the spec).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import M4V2Config

logger = logging.getLogger(__name__)


# --- query / instance normalization -----------------------------------------
# The repo stores queries with fields {instance_id, question, skill_annotations}.
# The spec's contract is {query_id, query, gold_skill_ids}. We normalize on load.

def load_instances(cfg: M4V2Config, ds: str) -> list[dict]:
    """Return normalized instances for dataset ``ds``.

    Each dict: ``{instance_id, dataset, query, gold_skill_ids}`` (plus the raw
    ``eval_data`` passed through untouched).
    """
    path = cfg.paths["instances_dir"] / f"{ds}.json"
    raw = json.loads(path.read_text())
    out = []
    for r in raw:
        out.append({
            "instance_id": r["instance_id"],
            "dataset": r.get("dataset", ds),
            "query": r["question"],
            "gold_skill_ids": list(r.get("skill_annotations") or []),
            "eval_data": r.get("eval_data", {}),
        })
    return out


def instances_by_id(cfg: M4V2Config, datasets: list[str]) -> dict[str, dict]:
    """``instance_id -> instance dict`` across the given datasets (for metrics)."""
    out: dict[str, dict] = {}
    for ds in datasets:
        for inst in load_instances(cfg, ds):
            out[inst["instance_id"]] = inst
    return out


# --- corpus / embeddings / clusters ------------------------------------------

@dataclass
class Artifacts:
    """Everything aligned to the corpus row index ``i`` (0..N-1)."""

    corpus_ids: list[str]                 # row i -> skill_id
    sid_to_idx: dict[str, int]            # skill_id -> row i
    corpus_emb: np.ndarray                # (N, d) float32, L2-normalized
    centroids: np.ndarray                 # (K, d) float32, L2-normalized
    skill_cluster: np.ndarray             # (N,) int32, cluster id per row
    N: int
    K: int
    d: int

    def cluster_of(self, skill_id: str) -> int:
        return int(self.skill_cluster[self.sid_to_idx[skill_id]])

    def emb_of(self, skill_id: str) -> np.ndarray:
        return self.corpus_emb[self.sid_to_idx[skill_id]]


def load_artifacts(cfg: M4V2Config) -> Artifacts:
    """Load corpus embeddings, centroids and cluster labels with alignment checks."""
    corpus_ids = json.loads(cfg.paths["corpus_ids"].read_text())
    sid_to_idx = {sid: i for i, sid in enumerate(corpus_ids)}
    corpus_emb = np.load(cfg.paths["corpus_emb"]).astype(np.float32)
    centroids = np.load(cfg.paths["centroids"]).astype(np.float32)
    clusters = json.loads(cfg.paths["clusters"].read_text())

    N, d = corpus_emb.shape
    K = centroids.shape[0]
    if len(corpus_ids) != N:
        raise ValueError(
            f"corpus_ids ({len(corpus_ids)}) != corpus_emb rows ({N})")
    if centroids.shape[1] != d:
        raise ValueError(
            f"centroid dim {centroids.shape[1]} != embedding dim {d}")

    # cluster label per corpus row (Step 1 alignment: every id must be present)
    missing = [sid for sid in corpus_ids if sid not in clusters]
    if missing:
        raise ValueError(
            f"{len(missing)} skills missing a cluster label (e.g. {missing[:3]})")
    skill_cluster = np.array([int(clusters[sid]) for sid in corpus_ids],
                             dtype=np.int32)
    if int(skill_cluster.max()) + 1 > K:
        raise ValueError(
            f"max cluster id {skill_cluster.max()} >= K={K} centroids")
    if K != cfg.kmeans_k:
        logger.warning("centroids K=%d but config kmeans_k=%d; using K=%d",
                       K, cfg.kmeans_k, K)

    # Verify L2-normalization (spec Step 4: load and verify, do not re-normalize).
    cnorms = np.linalg.norm(centroids, axis=1)
    if np.max(np.abs(cnorms - 1.0)) > 1e-4:
        logger.warning("centroids not unit-norm (max dev %.2e); re-normalizing",
                       float(np.max(np.abs(cnorms - 1.0))))
        centroids = centroids / np.maximum(cnorms[:, None], 1e-12)
    enorms = np.linalg.norm(corpus_emb, axis=1)
    if np.max(np.abs(enorms - 1.0)) > 1e-3:
        logger.warning("corpus_emb not unit-norm (max dev %.2e); re-normalizing",
                       float(np.max(np.abs(enorms - 1.0))))
        corpus_emb = corpus_emb / np.maximum(enorms[:, None], 1e-12)

    logger.info("artifacts: N=%d skills, d=%d, K=%d clusters", N, d, K)
    return Artifacts(corpus_ids, sid_to_idx, corpus_emb, centroids,
                     skill_cluster, N, K, d)


# --- query embedding cache ----------------------------------------------------

def load_query_embeddings(cfg: M4V2Config, ds: str) -> dict[str, np.ndarray]:
    """Return ``instance_id -> e_q`` (L2-normalized) from the cached npy + ids."""
    emb = np.load(cfg.query_emb_path(ds)).astype(np.float32)
    ids = json.loads(cfg.query_emb_ids_path(ds).read_text())
    if len(ids) != emb.shape[0]:
        raise ValueError(f"{ds}: query emb rows {emb.shape[0]} != ids {len(ids)}")
    return {qid: emb[i] for i, qid in enumerate(ids)}


def save_query_embeddings(cfg: M4V2Config, ds: str, ids: list[str],
                          emb: np.ndarray) -> None:
    p = cfg.query_emb_path(ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, emb.astype(np.float32))
    cfg.query_emb_ids_path(ds).write_text(json.dumps(ids))


# --- small helpers ------------------------------------------------------------

def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))


def write_jsonl(path: str | Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
