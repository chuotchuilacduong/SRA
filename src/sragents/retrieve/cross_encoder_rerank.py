"""Cross-encoder passage re-ranker (RocketQAv2-style inference).

At inference the algorithm is:
  for each (query, candidate_passage) pair → cross-encoder score → sort descending.

The cross-encoder attends over the full concatenation
  [CLS] query [SEP] passage [SEP]
capturing fine-grained query-passage interactions that a dual-encoder cannot.

Optional BM25 blending (alpha > 0):
  final = (1 - alpha) * norm(ce_score) + alpha * norm(bm25_score)
Default alpha=0.0 → pure cross-encoder ranking.
"""

from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    """Re-ranks BM25 candidates using a cross-encoder model.

    Call :meth:`build_index` once over the full corpus (builds a text lookup),
    then call :meth:`rerank` per query.  The cross-encoder cannot pre-encode
    passages independently, so there is no corpus-level caching.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 512,
        alpha: float = 0.0,
    ):
        self._model_path = model_name_or_path
        self._batch_size = batch_size
        self._max_length = max_length
        self.alpha = alpha
        self._model = None
        self._corpus_texts: dict[str, str] = {}

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"  Loading cross-encoder: {self._model_path}")
            self._model = CrossEncoder(
                self._model_path,
                max_length=self._max_length,
            )

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Build a skill_id → text lookup.  No heavy computation needed."""
        self._corpus_texts = dict(zip(corpus_ids, corpus_texts))

    def rerank(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[tuple[str, float]]:
        """Return candidates re-sorted by cross-encoder score.

        Args:
            query: Raw query string.
            candidates: List of ``{"skill_id": str, "score": float}`` dicts
                ordered by BM25 score descending.

        Returns:
            ``[(skill_id, final_score), ...]`` sorted by descending score.
        """
        if not candidates:
            return []

        self._load_model()

        pairs = [
            (query, self._corpus_texts.get(c["skill_id"], ""))
            for c in candidates
        ]
        ce_scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.alpha > 0.0:
            bm25 = np.array([c["score"] for c in candidates], dtype=np.float32)
            lo, hi = bm25.min(), bm25.max()
            bm25_norm = (bm25 - lo) / (hi - lo) if hi > lo else np.ones_like(bm25)

            clo, chi = ce_scores.min(), ce_scores.max()
            ce_norm = (ce_scores - clo) / (chi - clo) if chi > clo else np.ones_like(ce_scores)

            final = (1.0 - self.alpha) * ce_norm + self.alpha * bm25_norm
        else:
            final = ce_scores

        order = np.argsort(final)[::-1]
        return [(candidates[i]["skill_id"], float(final[i])) for i in order]
