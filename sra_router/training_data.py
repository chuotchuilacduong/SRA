"""Build (query, positive, hard_negs) triples for encoder training.

Pipeline:
  1. Filter training instances to those with at least one gold skill in the corpus.
  2. For each (query, gold_id), mine 4-source negatives, then run the
     3-layer false-negative filter.
  3. Cache the result to a JSONL so reruns skip the expensive encode pass.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sra_router.corpus import SkillCorpus
from sra_router.false_neg_filter import filter_false_negatives
from sra_router.instances import SRAInstance
from sra_router.negative_mining import (
    mine_negatives_for_query,
    precompute_bm25_topk,
    precompute_dense_topk,
)


def build_training_pairs(
    train_instances: list[SRAInstance],
    corpus: SkillCorpus,
    cache_path: str | Path,
    *,
    base_dense_model: str = "BAAI/bge-small-en-v1.5",
    n_per_source: tuple[int, int, int, int] = (4, 3, 2, 1),
    apply_false_neg_filter: bool = True,
    seed: int = 42,
    force: bool = False,
) -> list[dict]:
    """Return a list of ``{query, positive_id, hard_neg_ids, dataset}`` dicts."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return [json.loads(l) for l in cache_path.read_text().splitlines() if l.strip()]

    queries = [inst.query for inst in train_instances]
    print("Mining BM25 top-50...")
    bm25_topk = precompute_bm25_topk(corpus, queries, top_k=50)
    print("Mining dense top-50...")
    dense_topk, _, corpus_emb = precompute_dense_topk(
        corpus, queries, model_name=base_dense_model, top_k=50,
    )
    id_to_idx = {sid: i for i, sid in enumerate(corpus.ids())}
    skill_emb_lookup: dict[str, np.ndarray] = {sid: corpus_emb[i] for sid, i in id_to_idx.items()}

    rng = random.Random(seed)
    n_sem, n_lex, n_tax, n_rand = n_per_source

    out: list[dict] = []
    drop_totals = {"name_dup": 0, "trigram": 0, "cosine": 0}
    for i, inst in enumerate(tqdm(train_instances, desc="Mining negatives")):
        gold = set(inst.gold_skill_ids)
        if not any(g in corpus for g in gold):
            continue

        # Closures over precomputed lists for this query.
        bm25_lst = bm25_topk[i]
        dense_lst = dense_topk[i]
        sem_fn = lambda q, k, lst=dense_lst: lst[:k]
        lex_fn = lambda q, k, lst=bm25_lst: lst[:k]

        neg_ids = mine_negatives_for_query(
            inst.query, gold, corpus,
            semantic_topk_fn=sem_fn,
            bm25_topk_fn=lex_fn,
            n_semantic=n_sem, n_lexical=n_lex, n_taxonomy=n_tax, n_random=n_rand,
            rng=rng,
        )
        if apply_false_neg_filter:
            neg_ids, drops = filter_false_negatives(
                neg_ids, list(gold), corpus,
                embeddings=skill_emb_lookup,
            )
            for k, v in drops.items():
                drop_totals[k] += v

        for gold_id in inst.gold_skill_ids:
            if gold_id not in corpus:
                continue
            out.append({
                "instance_id": inst.instance_id,
                "dataset": inst.dataset,
                "query": inst.query,
                "positive_id": gold_id,
                "hard_neg_ids": neg_ids,
            })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"  Wrote {len(out)} pairs → {cache_path}")
    print(f"  False-neg drop counts: {drop_totals}")
    return out
