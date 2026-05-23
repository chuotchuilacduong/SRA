"""End-to-end retrieval evaluation: BM25, BGE zero-shot, trained SR-Emb."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from sra_router.corpus import SkillCorpus
from sra_router.encoder import QUERY_PREFIX, SkillEncoder, pick_device
from sra_router.instances import SRAInstance
from sra_router.metrics import aggregate, ndcg_at_k, recall_at_k


def _per_query_metrics(ranked_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    return {
        "Recall@1":  recall_at_k(ranked_ids, gold_ids, 1),
        "Recall@5":  recall_at_k(ranked_ids, gold_ids, 5),
        "Recall@10": recall_at_k(ranked_ids, gold_ids, 10),
        "Recall@50": recall_at_k(ranked_ids, gold_ids, 50),
        "nDCG@10":   ndcg_at_k(ranked_ids, gold_ids, 10),
    }


def evaluate_ranker(
    ranker: Callable[[list[str], int], list[list[tuple[str, float]]]],
    eval_instances: list[SRAInstance],
    *,
    top_k: int = 50,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    queries = [inst.query for inst in eval_instances]
    ranked = ranker(queries, top_k)
    per_q: list[dict[str, float]] = []
    by_dataset: dict[str, list[dict[str, float]]] = {}
    for inst, lst in zip(eval_instances, ranked):
        ids = [sid for sid, _ in lst]
        m = _per_query_metrics(ids, inst.gold_skill_ids)
        per_q.append(m)
        by_dataset.setdefault(inst.dataset, []).append(m)
    overall = aggregate(per_q)
    by_ds = {ds: aggregate(rows) for ds, rows in sorted(by_dataset.items())}
    return overall, by_ds


def make_bm25_ranker(corpus: SkillCorpus):
    from sragents.retrieve.bm25 import BM25Retriever
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    retriever = BM25Retriever()
    retriever.build_index(ids, texts)
    return retriever.retrieve


def make_dense_ranker(
    corpus: SkillCorpus,
    model_name: str,
    *,
    query_prefix: str = "",
    device: str | None = None,
    batch_size: int = 64,
):
    """Return a callable matching the BM25 retriever's interface."""
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    dev = pick_device(device)
    model = SentenceTransformer(model_name, device=str(dev))
    print(f"  Encoding corpus ({len(texts)}) with {model_name}...")
    corpus_emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_tensor=True, show_progress_bar=True).to(dev)

    def _retrieve(queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        prefixed = [query_prefix + q for q in queries]
        q_emb = model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_tensor=True, show_progress_bar=False).to(dev)
        sims = (q_emb @ corpus_emb.T).cpu().numpy()
        out = []
        for i in range(sims.shape[0]):
            top = np.argpartition(-sims[i], range(top_k))[:top_k]
            top = top[np.argsort(-sims[i, top])]
            out.append([(ids[j], float(sims[i, j])) for j in top])
        return out

    return _retrieve


def make_sr_emb_ranker(
    corpus: SkillCorpus,
    checkpoint_dir: str,
    *,
    device: str | None = None,
    batch_size: int = 64,
):
    """Reuse the trained SkillEncoder."""
    enc = SkillEncoder(model_name=checkpoint_dir, device=pick_device(device))
    enc.eval()
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    print(f"  Encoding corpus ({len(texts)}) with trained SR-Emb...")
    corpus_emb = enc.encode_eval(texts, is_query=False, batch_size=batch_size).cpu().numpy()

    def _retrieve(queries: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        q_emb = enc.encode_eval(queries, is_query=True, batch_size=batch_size).cpu().numpy()
        sims = q_emb @ corpus_emb.T
        out = []
        for i in range(sims.shape[0]):
            top = np.argpartition(-sims[i], range(top_k))[:top_k]
            top = top[np.argsort(-sims[i, top])]
            out.append([(ids[j], float(sims[i, j])) for j in top])
        return out

    return _retrieve


def run_all(
    corpus: SkillCorpus,
    eval_instances: list[SRAInstance],
    *,
    sr_emb_ckpt: str | None,
    output_path: str | Path,
    top_k: int = 50,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    print("\n=== BM25 ===")
    bm25 = make_bm25_ranker(corpus)
    overall, by_ds = evaluate_ranker(bm25, eval_instances, top_k=top_k)
    results["BM25"] = {"overall": overall, "by_dataset": by_ds}

    print("\n=== BGE zero-shot (bge-small-en-v1.5) ===")
    bge = make_dense_ranker(
        corpus, "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    )
    overall, by_ds = evaluate_ranker(bge, eval_instances, top_k=top_k)
    results["BGE-small-zero-shot"] = {"overall": overall, "by_dataset": by_ds}

    if sr_emb_ckpt and Path(sr_emb_ckpt).exists():
        print("\n=== SR-Emb (fine-tuned) ===")
        sr = make_sr_emb_ranker(corpus, sr_emb_ckpt)
        overall, by_ds = evaluate_ranker(sr, eval_instances, top_k=top_k)
        results["SR-Emb-finetuned"] = {"overall": overall, "by_dataset": by_ds}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved comparison → {output_path}")
    return results
