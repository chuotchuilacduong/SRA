"""Score fusion methods for combining complementary retrieval signals.

Strategy:
    BM25 (sparse, lexical)      <- exact keyword/phrase matching, rare terms
    Dense embeddings (semantic) <- meaning, paraphrase, synonyms
    LinearRAG (graph+dense)     <- entity-graph propagation, multi-hop

Each captures DIFFERENT aspects of relevance:
    - BM25 strong, Dense weak:  exact code/API names, math symbols, rare jargon
    - BM25 weak,   Dense strong: paraphrased queries, semantic similarity
    - LinearRAG:                 structured entity relationships

Reciprocal Rank Fusion (RRF, Cormack et al. 2009) fuses these by
RANK rather than raw score, sidestepping the score-scale mismatch
between BM25 (raw float, e.g. 25.7) and dense cosine (e.g. 0.85).

Reference:
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    "Reciprocal Rank Fusion outperforms Condorcet and individual
    Rank Learning Methods." SIGIR '09.
"""

from __future__ import annotations


def rrf_merge(
    list_a: list[dict],
    list_b: list[dict],
    top_k: int = 50,
    k_rrf: int = 60,
) -> list[dict]:
    """Two-way Reciprocal Rank Fusion.

    Formula: score(doc) = sum_{i in lists} 1 / (k_rrf + rank_i(doc))

    The constant k_rrf=60 is the robust default from Cormack et al. (2009).
    Higher k diminishes the impact of top ranks (smoother fusion);
    lower k emphasizes top ranks more aggressively.

    Args:
        list_a: First ranked list of dicts with 'skill_id' key.
        list_b: Second ranked list of dicts with 'skill_id' key.
        top_k: Number of results to return.
        k_rrf: RRF constant (default 60).

    Returns:
        Merged list of {'skill_id', 'score', 'rank'} dicts.
    """
    return multi_rrf_merge([list_a, list_b], top_k=top_k, k_rrf=k_rrf)


def weighted_rrf_merge(
    list_a: list[dict],
    list_b: list[dict],
    weight_a: float = 1.0,
    weight_b: float = 1.0,
    top_k: int = 50,
    k_rrf: int = 60,
) -> list[dict]:
    """Two-way Weighted RRF.

    Equal weights reduce to standard RRF. Useful when one retriever is
    empirically more reliable than the other for a domain.
    """
    return multi_rrf_merge(
        [list_a, list_b],
        weights=[weight_a, weight_b],
        top_k=top_k,
        k_rrf=k_rrf,
    )


def multi_rrf_merge(
    rankings: list[list[dict]],
    weights: list[float] | None = None,
    top_k: int = 50,
    k_rrf: int = 60,
) -> list[dict]:
    """N-way Reciprocal Rank Fusion of complementary retrievers.

    Designed for combining sparse + dense + graph-based signals:
        rankings = [bm25_ranks, dense_ranks, linearrag_ranks]

    Higher RRF score = ranked highly by MULTIPLE retrievers
    (cumulative evidence across complementary signals).

    Args:
        rankings: List of ranked lists, each [{'skill_id', ...}, ...].
        weights: Per-ranker weights (default: all 1.0 = equal trust).
                 Higher weight = retriever more trusted.
        top_k: Number of results to return.
        k_rrf: RRF constant (default 60, from Cormack et al.).

    Returns:
        Top-k merged list of {'skill_id', 'score', 'rank'}.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    assert len(weights) == len(rankings), \
        f"weights ({len(weights)}) must match rankings ({len(rankings)})"

    scores: dict[str, float] = {}
    for lst, weight in zip(rankings, weights):
        for rank, item in enumerate(lst):
            sid = item["skill_id"]
            scores[sid] = scores.get(sid, 0.0) + weight / (k_rrf + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [
        {"skill_id": sid, "score": sc, "rank": i + 1}
        for i, (sid, sc) in enumerate(ranked)
    ]


def complementary_fusion(
    bm25_ranks: list[dict],
    dense_ranks: list[dict],
    linearrag_ranks: list[dict] | None = None,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    linearrag_weight: float = 1.0,
    top_k: int = 50,
    k_rrf: int = 60,
) -> list[dict]:
    """Convenience wrapper documenting the COMPLEMENTARY fusion strategy.

    This function explicitly names the three signal types so callers
    can reason about what each component contributes:

      - BM25 (sparse, lexical):
          - Strong: exact keyword matches, rare technical terms,
            code API names, math symbols, named entities
          - Weak: paraphrased queries, synonyms

      - Dense (semantic, BGE/Contriever):
          - Strong: semantic similarity, paraphrase robustness,
            natural-language scenarios
          - Weak: rare keywords, code-symbol exact match

      - LinearRAG (graph + dense + entity propagation, optional):
          - Strong: multi-hop reasoning, entity relationship
          - Weak: domain-specific symbolic content (e.g. logic)

    Args:
        bm25_ranks: BM25 top-k ranking.
        dense_ranks: Dense (BGE/Contriever) top-k ranking.
        linearrag_ranks: Optional LinearRAG ranking (omit if not used).
        *_weight: Per-signal weight; 1.0 means equal trust.
        top_k: Number of results to return.
        k_rrf: RRF constant.

    Returns:
        Fused top-k list.
    """
    rankings = [bm25_ranks, dense_ranks]
    weights = [bm25_weight, dense_weight]
    if linearrag_ranks is not None:
        rankings.append(linearrag_ranks)
        weights.append(linearrag_weight)
    return multi_rrf_merge(rankings, weights=weights, top_k=top_k, k_rrf=k_rrf)
