"""Augment cached L6 feature tables with the bge_ft (fine-tuned retriever) signal.

Reads the baseline 45-col tables `results/m4_v2/cache/ltr_features/{ds}.npz`, computes
the 7 `FEATURE_GROUPS["bge_ft"]` features per (query, candidate) row from the
`sr-emb-bge-v1` embeddings, appends them, and writes a SEPARATE cache
`results/m4_v2/cache/ltr_features_bgeft/{ds}.npz` (52 cols, feature_names=ALL_FEATURES).
The default cache is left untouched, so the §20 baseline L6 stays reproducible.

bge_ft features (within each query's candidate pool), u=qft(query), v=sft(skill),
both L2-normalized (so u·v = cosine):
  bgeft_cosine, bgeft_rank, bgeft_inv_rank, bgeft_rank_norm,
  bgeft_is_top1, bgeft_margin_top1 (cos-max), bgeft_z (within-pool z-score).

Query embeddings are cached at `results/m4_v2/cache/query_emb_ft/{ds}.npy` for reuse.
GPU is used only for the one-time query encode (~5.4k short queries → seconds). Idempotent.

Run: python src/kmeans/scripts/add_bgeft_features.py [--datasets ...]
"""

import argparse
import json
from pathlib import Path

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
import _bootstrap  # noqa: F401
_bootstrap.setup_logging()

import numpy as np                                       # noqa: E402
from sragents.config import PROJECT_ROOT                 # noqa: E402
from kmeans.qsc_ltr_runner import load_ext               # noqa: E402
from kmeans import io, ltr_features                       # noqa: E402

CACHE = PROJECT_ROOT / "results" / "m4_v2" / "cache"
SRC = CACHE / "ltr_features"
DST = CACHE / "ltr_features_bgeft"
QEMB = CACHE / "query_emb_ft"
MODEL = PROJECT_ROOT / "results" / "models" / "sr-emb-bge-v1"
CORPUS_EMB = MODEL / "corpus_emb.npy"
CORPUS_IDS = PROJECT_ROOT / "results" / "bge" / "corpus_ids.json"

BASELINE_GROUPS = ["retrieval", "m4", "a7", "qsc", "confidence"]
N_BASE = len([f for g in BASELINE_GROUPS for f in ltr_features.FEATURE_GROUPS[g]])  # 45
BGEFT = ltr_features.FEATURE_GROUPS["bge_ft"]  # 7 names, the output column order


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_corpus():
    emb = np.load(CORPUS_EMB).astype(np.float32)         # (N, 768), row-aligned to corpus_ids
    ids = json.loads(CORPUS_IDS.read_text())
    return emb, {s: i for i, s in enumerate(ids)}


def encode_queries(ds, cfg, st):
    """instance_id -> normalized query embedding; cached per dataset."""
    QEMB.mkdir(parents=True, exist_ok=True)
    npy, idf = QEMB / f"{ds}.npy", QEMB / f"{ds}_ids.json"
    insts = io.load_instances(cfg, ds)
    if npy.exists() and idf.exists():
        ids = json.loads(idf.read_text())
        arr = np.load(npy).astype(np.float32)
        if len(ids) == arr.shape[0]:
            return dict(zip(ids, arr))
    ids = [r["instance_id"] for r in insts]
    texts = [r["query"] for r in insts]
    arr = st.encode(texts, batch_size=256, normalize_embeddings=True,
                    show_progress_bar=False).astype(np.float32)
    np.save(npy, arr)
    idf.write_text(json.dumps(ids))
    return dict(zip(ids, arr))


def bgeft_block(sids, u, corpus_emb, cidx):
    """(n, 7) bge_ft feature block for one query's pool, columns in BGEFT order."""
    n = len(sids)
    V = np.zeros((n, corpus_emb.shape[1]), dtype=np.float32)
    miss = 0
    for j, s in enumerate(sids):
        i = cidx.get(s)
        if i is None:
            miss += 1
        else:
            V[j] = corpus_emb[i]
    cos = (V @ u).astype(np.float32)                     # cosine (missing skills -> 0)
    order = np.argsort(-cos, kind="stable")
    rank = np.empty(n, dtype=np.float32)
    rank[order] = np.arange(1, n + 1, dtype=np.float32)
    top = float(cos[order[0]])
    std = float(cos.std())
    z = ((cos - float(cos.mean())) / std).astype(np.float32) if std > 1e-12 \
        else np.zeros(n, dtype=np.float32)
    block = np.column_stack([
        cos,                                # bgeft_cosine
        rank,                               # bgeft_rank
        1.0 / rank,                         # bgeft_inv_rank
        rank / float(n),                    # bgeft_rank_norm
        (rank == 1).astype(np.float32),     # bgeft_is_top1
        cos - top,                          # bgeft_margin_top1  (<=0)
        z,                                  # bgeft_z
    ]).astype(np.float32)
    return block, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    ext, cfg = load_ext()
    datasets = args.datasets or ext["datasets"]
    assert len(BGEFT) == 7, BGEFT

    from sentence_transformers import SentenceTransformer
    dev = _device()
    print(f"[add_bgeft] device={dev}  model={MODEL.name}", flush=True)
    st = SentenceTransformer(str(MODEL), device=dev)
    corpus_emb, cidx = load_corpus()
    print(f"[add_bgeft] corpus_emb={corpus_emb.shape} ids={len(cidx)}", flush=True)

    for ds in datasets:
        t = ltr_features.load_features(SRC / f"{ds}.npz")
        nbase = t["X"].shape[1]
        assert nbase == N_BASE, f"{ds}: expected {N_BASE} base cols, got {nbase}"
        qemb = encode_queries(ds, cfg, st)

        blocks, off, miss_tot, miss_u = [], 0, 0, 0
        sizes, sids_all = t["group_sizes"], t["skill_ids"]
        for i, iid in enumerate(t["instance_ids"]):
            n = int(sizes[i]); sl = slice(off, off + n); off += n
            u = qemb.get(iid)
            if u is None:
                miss_u += 1
                blocks.append(np.zeros((n, 7), dtype=np.float32))
                continue
            blk, miss = bgeft_block(sids_all[sl], u, corpus_emb, cidx)
            miss_tot += miss
            blocks.append(blk)
        extra = np.concatenate(blocks, axis=0)
        assert extra.shape[0] == t["X"].shape[0]

        t["X"] = np.concatenate([t["X"], extra], axis=1).astype(np.float32)
        t["feature_names"] = list(ltr_features.ALL_FEATURES)
        ltr_features.save_features(DST / f"{ds}.npz", t)
        print(f"  {ds:14} {t['X'].shape[0]:6d} rows x {t['X'].shape[1]} cols "
              f"(+7 bge_ft)  miss_skill_emb={miss_tot} miss_query_emb={miss_u}", flush=True)
    print(f"[add_bgeft] done -> {DST}", flush=True)


if __name__ == "__main__":
    main()
