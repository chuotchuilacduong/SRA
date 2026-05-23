"""Phase 1 — 4-source hard negative mining.

Per the plan: 4 semantic + 3 lexical + 2 taxonomy + 1 random = 10 negatives
per query, drawn from complementary signal sources so the encoder learns to
discriminate against all of them simultaneously.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

from sra_router.corpus import SkillCorpus


def _sample_non_gold(candidate_ids: list[str], gold_ids: set[str], k: int,
                     rng: random.Random) -> list[str]:
    pool = [s for s in candidate_ids if s not in gold_ids]
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def mine_negatives_for_query(
    query: str,
    gold_ids: set[str],
    corpus: SkillCorpus,
    semantic_topk_fn: Callable[[str, int], list[str]],
    bm25_topk_fn: Callable[[str, int], list[str]],
    n_semantic: int = 4,
    n_lexical: int = 3,
    n_taxonomy: int = 2,
    n_random: int = 1,
    rng: random.Random | None = None,
) -> list[str]:
    """Return up to ``n_semantic + n_lexical + n_taxonomy + n_random`` ids.

    Each call is independent — pass a seeded ``rng`` for reproducibility.
    """
    rng = rng or random.Random()
    sem_pool = semantic_topk_fn(query, 50)
    sem = _sample_non_gold(sem_pool, gold_ids, n_semantic, rng)

    lex_pool = bm25_topk_fn(query, 50)
    lex = _sample_non_gold(lex_pool, gold_ids, n_lexical, rng)

    gold_sources = {corpus[g].source for g in gold_ids if g in corpus}
    tax_pool = [
        sid for src in gold_sources for sid in corpus.ids_in_source(src)
        if sid not in gold_ids
    ]
    tax = rng.sample(tax_pool, min(n_taxonomy, len(tax_pool))) if tax_pool else []

    rand_pool = corpus.ids_not_in_sources(gold_sources | {"distractor"})
    # If gold sources cover all sources, fall back to any non-gold id.
    if not rand_pool:
        rand_pool = [sid for sid in corpus.ids() if sid not in gold_ids]
    rand = rng.sample(rand_pool, min(n_random, len(rand_pool))) if rand_pool else []

    seen: set[str] = set(gold_ids)
    out: list[str] = []
    for sid in sem + lex + tax + rand:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def precompute_bm25_topk(
    corpus: SkillCorpus, queries: list[str], top_k: int = 50,
) -> list[list[str]]:
    """Build a single BM25 index over the corpus and return top-K per query.

    Uses the same tokenizer as the in-tree ``sragents`` BM25 retriever so the
    negatives mirror what the BM25 baseline would surface at inference time.
    """
    from sragents.retrieve.bm25 import BM25Retriever

    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    retriever = BM25Retriever()
    retriever.build_index(ids, texts)
    raw = retriever.retrieve(queries, top_k=top_k)
    return [[sid for sid, _ in lst] for lst in raw]


def precompute_dense_topk(
    corpus: SkillCorpus,
    queries: list[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    top_k: int = 50,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    """Return (top-K ids per query, query embs, corpus embs).

    The cached corpus + query matrices are reused by the trainer to avoid a
    second encode pass.
    """
    from sentence_transformers import SentenceTransformer

    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]

    model = SentenceTransformer(model_name, device=device)
    print(f"  Encoding {len(texts)} skills with {model_name}...")
    corpus_emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              show_progress_bar=True, convert_to_numpy=True)
    print(f"  Encoding {len(queries)} queries...")
    query_emb = model.encode(queries, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=True, convert_to_numpy=True)
    sims = query_emb @ corpus_emb.T  # [n_query, n_corpus]
    out = []
    for i in range(sims.shape[0]):
        top = np.argpartition(-sims[i], range(top_k))[:top_k]
        top = top[np.argsort(-sims[i, top])]
        out.append([ids[j] for j in top])
    return out, query_emb, corpus_emb
