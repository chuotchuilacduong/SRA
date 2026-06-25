"""α-CE β=0.7 fusion reranker — integrates the OPTIMAL Method 7 from
docs/FINAL_METHODS_AND_RESULTS.md directly into the inference path.

Why this exists
---------------
The baseline ``CrossEncoderReranker`` (sragents/retrieve/cross_rerank.py)
only outputs RAW CE logits (= Method 5: "CE v3 H100").

The OPTIMAL method on the full bench is Method 7 — the *post-hoc* α-CE
β=0.7 fusion that blends normalized CE with Stage-1 score:

    score★(s) = β · ĉe(s) + (1 − β) · Ŝ1(s),    β = 0.7

Where ``ĉe`` and ``Ŝ1`` are per-query min-max normalized CE / Stage-1
scores (see ``experiments/fuse_ce_stage1.py``).

Currently this β-blend lives in the offline experiments script and is
NOT applied during inference — production retrieval ships only Method 5.
``BetaCEReranker`` closes that gap: it runs the same CE forward pass,
then applies the β-blend in-place before returning, so callers get the
optimal ranking with the same call signature as ``rerank()``.

Macro gains carried over (from FINAL_METHODS_AND_RESULTS.md, §C.1):
  R@10  : CE-only 82.64  →  α-CE β=0.7  86.06  (+3.42 pp)
  nDCG@10: 69.74        →  74.67       (+4.93 pp)

Cost
----
Zero extra model calls. Just O(top_k) array ops per query. Same latency
as plain CE (~1.2 s / query CPU, MiniLM-L6).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from sragents.retrieve.cross_rerank import CrossEncoderReranker


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Per-query min-max normalize to [0, 1]; degenerate → 0.5."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


class BetaCEReranker(CrossEncoderReranker):
    """Cross-encoder reranker with built-in α-CE β=0.7 score fusion.

    Inherits everything from :class:`CrossEncoderReranker` (model load,
    SkillPacker, MaxPChunker, device policy). The only change is how the
    final per-candidate score is computed:

        score★ = β · minmax(ce_logits) + (1 − β) · minmax(stage1_scores)

    Stage-1 scores come from the ``score`` field that was on the
    candidate dict BEFORE CE rerank — i.e. the upstream RRF / RRF+KMeans
    score. ``rerank()`` reads it via ``c["score"]`` for each input
    candidate, exactly as the post-hoc fuser does.

    Args (additional vs base class):
        beta: Blend weight for CE. Default 0.7 = Method 7 optimum.
              Set ``beta=1.0`` to reproduce plain CE (Method 5).
    """

    def __init__(self, *args, beta: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = float(beta)

    # ------------------------------------------------------------- API

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 50,
        batch_size: int = 32,
        text_max_chars: int = 3000,
    ) -> list[dict]:
        """Cross-encoder rerank with α-CE β-blend.

        Steps
        -----
        1. Cache each candidate's Stage-1 ``score`` (upstream RRF / α-KM).
        2. Run plain CE rerank via the parent class to get CE logits.
        3. Per-query min-max normalize both CE logits and Stage-1 scores.
        4. Blend:  final = β · ĉe + (1 − β) · Ŝ1.
        5. Re-sort by final score, attach provenance fields, return top-k.
        """
        if not candidates:
            return []

        # 1. Snapshot Stage-1 scores BEFORE CE overwrites them
        stage1_score_by_id: dict[str, float] = {
            c["skill_id"]: float(c.get("score", 0.0)) for c in candidates
        }

        # 2. Get CE-only ranking (parent class) — we want all candidates back,
        #    not just top-k, so we can blend over the full pool then truncate.
        ce_ranked = super().rerank(
            query=query,
            candidates=candidates,
            top_k=len(candidates),  # keep full pool for fair blend normalization
            batch_size=batch_size,
            text_max_chars=text_max_chars,
        )
        if not ce_ranked:
            return []

        # 3. Per-query min-max normalize on the FULL pool returned by CE
        ce_logits = np.array(
            [c["score"] for c in ce_ranked], dtype=np.float32,
        )
        s1_logits = np.array(
            [stage1_score_by_id.get(c["skill_id"], 0.0) for c in ce_ranked],
            dtype=np.float32,
        )
        ce_norm = _minmax(ce_logits)
        s1_norm = _minmax(s1_logits)

        # 4. β-blend
        final = self.beta * ce_norm + (1.0 - self.beta) * s1_norm
        order = np.argsort(final)[::-1]

        # 5. Build output rows with provenance
        out: list[dict] = []
        for new_rank, idx in enumerate(order[:top_k], start=1):
            base = dict(ce_ranked[int(idx)])
            base.update({
                "score": float(final[idx]),
                "rank": new_rank,
                "ce_score_raw": float(ce_logits[idx]),
                "ce_score_norm": float(ce_norm[idx]),
                "stage1_score_norm": float(s1_norm[idx]),
                "fusion_beta": self.beta,
            })
            out.append(base)
        return out
