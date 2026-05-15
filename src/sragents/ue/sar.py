"""SAR uncertainty estimator for adaptive retrieval decisions.

Implements all three methods from:
  "Shifting Attention to Relevance for Information Retrieval-Augmented
   Large Language Models" (ACL 2024, arXiv:2307.01379)

Five modes (``mode`` arg):

  ``pe``
      Predictive Entropy (PE, Eq. 1) — baseline used in the paper.
      Score = Σ_i -log p(z_i | z_{<i}, x)  averaged over K generations.
      Needs logprobs; no local model.

  ``token_sar``
      TOKENSAR (Eq. 6-8) — shifts attention to semantically important tokens.
      Score = Σ_i R̃_T(z_i) · (-log p(z_i))  averaged over K generations,
      where R̃_T is the normalized cross-encoder relevance of each token.
      Needs logprobs + cross-encoder (slow: O(T) encoder calls per sample).

  ``sent_sar``
      SENTSAR (Eq. 9-10) — shifts attention to semantically consistent
      sentences across K samples.
      Score = (1/K) Σ_j -log(p(s_j|x) + Σ_{k≠j} g(s_j,s_k)·p(s_k|x) / t)
      Needs logprobs + cross-encoder (with question prefix for similarity).
      Best speed/accuracy trade-off.

  ``sar``
      Full SAR (Eq. 11) — combines token- and sentence-level shifting.
      Replaces p(s_j|x) in SENTSAR with p'(s_j|x) = exp(-TOKENSAR(s_j)).
      Needs logprobs + cross-encoder + sentence encoder.

  ``consistency``
      No-logprobs fallback: 1 − mean pairwise cosine similarity across K
      samples.  Works with any API.  Automatically used when logprobs are
      unavailable (e.g. Timely API).

Score ranges:
  - ``pe`` / ``token_sar``: nats, roughly 0.5–3.0  → threshold ~1.0–2.0
  - ``sent_sar`` / ``sar``: can be negative (numerically shifted); tune threshold
  - ``consistency``: [0, 1]  → threshold ~0.3–0.6

All thresholds are model- and dataset-dependent; calibrate on a held-out split.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# SAR sentence-level temperature (paper default §5.1: t = 0.001).
# Very small t makes the relevance term dominate; larger t balances it.
_DEFAULT_SAR_TEMPERATURE = 0.001


@dataclass
class SARResult:
    """Uncertainty probe result for one query."""

    score: float          # uncertainty score (higher = more uncertain)
    method: str           # mode actually used (may differ after fallback)
    n_generations: int    # number of valid samples that contributed
    needs_retrieval: bool # score > threshold


class SARScorer:
    """Estimate LLM uncertainty to decide whether external skills are needed.

    Args:
        mode: Scoring strategy — see module docstring.
        n_samples: Number of stochastic samples (K in the paper; paper uses 5).
        temperature: Sampling temperature for probe generations.
        threshold: Score above which the model is deemed uncertain enough
            to warrant skill retrieval.  Calibrate on a validation split.
        sar_temperature: The ``t`` parameter in Eq. (9)/(11); paper uses 0.001.
        importance_model: Cross-encoder model for sentence/token similarity
            (sent_sar, token_sar, sar).  Paper uses stsb-roberta-large.
    """

    def __init__(
        self,
        mode: str = "sent_sar",
        n_samples: int = 5,
        temperature: float = 0.7,
        threshold: float = 1.5,
        sar_temperature: float = _DEFAULT_SAR_TEMPERATURE,
        importance_model: str = "cross-encoder/stsb-roberta-large",
    ) -> None:
        valid = ("pe", "token_sar", "sent_sar", "sar", "consistency")
        if mode not in valid:
            raise ValueError(f"Unknown mode {mode!r}. Choose one of {valid}.")

        self.mode = mode
        self.n_samples = max(2, n_samples)  # need ≥2 for sentence-level
        self.temperature = temperature
        self.threshold = threshold
        self.sar_temperature = sar_temperature

        self._cross_encoder: Any = None
        self._sent_encoder: Any = None

        # sent_sar, token_sar, and sar all use the cross-encoder:
        #   sent_sar / sar: pairwise sentence similarities g(s_i, s_j)
        #   token_sar / sar: token-level importance weights
        if mode in ("sent_sar", "token_sar", "sar"):
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415
                self._cross_encoder = CrossEncoder(importance_model, num_labels=1)
                logger.info("Loaded cross-encoder: %s", importance_model)
            except Exception as exc:
                logger.warning(
                    "Cannot load cross-encoder (%s); sent_sar/sar will fall back"
                    " to consistency.", exc
                )
                if self.mode in ("sent_sar", "sar"):
                    self.mode = "consistency"
                elif self.mode == "token_sar":
                    self.mode = "pe"

        if self.mode == "consistency":
            self._load_sent_encoder()

    def _load_sent_encoder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._sent_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("Cannot load sentence encoder (%s).", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        question: str,
        client: Any,
        model: str,
        max_tokens: int = 256,
    ) -> SARResult:
        """Compute uncertainty score for *question*.

        Args:
            question: Full prompt to probe (no skills injected).
            client: OpenAI-compatible client.
            model: Model identifier.
            max_tokens: Token budget per probe generation (keep short).
        """
        if self.mode == "consistency":
            return self._consistency_score(question, client, model, max_tokens)

        generations, had_missing = self._collect_generations(
            question, client, model, max_tokens
        )

        if not generations:
            if had_missing:
                logger.warning(
                    "API returned no logprobs; switching to consistency mode."
                )
                self.mode = "consistency"
                if self._sent_encoder is None:
                    self._load_sent_encoder()
                return self._consistency_score(question, client, model, max_tokens)
            return SARResult(
                score=float("inf"), method=self.mode,
                n_generations=0, needs_retrieval=True,
            )

        if self.mode == "pe":
            raw = self._pe(generations)
        elif self.mode == "token_sar":
            raw = self._tokensar(generations, question)
        elif self.mode == "sent_sar":
            raw = self._sentsar(generations, use_token_weights=False, question=question)
        else:  # sar (full)
            raw = self._sentsar(generations, use_token_weights=True, question=question)

        return SARResult(
            score=raw,
            method=self.mode,
            n_generations=len(generations),
            needs_retrieval=raw > self.threshold,
        )

    # ------------------------------------------------------------------
    # Generation collection
    # ------------------------------------------------------------------

    def _collect_generations(
        self, question: str, client: Any, model: str, max_tokens: int
    ) -> tuple[list[dict], bool]:
        """Collect K stochastic generations with logprobs.

        Returns:
            (list of generation dicts, had_missing_logprobs).
            Each dict has: text, lp_content, log_prob, n_tokens.
        """
        results: list[dict] = []
        missing = 0

        for _ in range(self.n_samples):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=1,
            )
            choice = response.choices[0]
            lp_obj = getattr(choice, "logprobs", None)
            lp_content = lp_obj.content if lp_obj is not None else None

            if not lp_content:
                missing += 1
                continue

            log_prob = sum(lp.logprob for lp in lp_content)
            results.append({
                "text": choice.message.content or "",
                "lp_content": lp_content,
                "log_prob": log_prob,
                "n_tokens": len(lp_content),
            })

        return results, missing > 0

    # ------------------------------------------------------------------
    # PE (Eq. 1) — predictive entropy baseline
    # ------------------------------------------------------------------

    def _pe(self, generations: list[dict]) -> float:
        """Sum of negative log-probs across K generations (PE, Eq. 1)."""
        scores = [
            sum(-lp.logprob for lp in g["lp_content"])
            for g in generations
        ]
        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # TOKENSAR (Eq. 6-8)
    # ------------------------------------------------------------------

    def _tokensar(self, generations: list[dict], question: str) -> float:
        """Token-shifted PE: importance-weighted neg-log-prob per generation."""
        import torch  # noqa: PLC0415

        scores: list[float] = []
        for g in generations:
            tokens = [lp.token for lp in g["lp_content"]]
            entropies = torch.tensor(
                [-lp.logprob for lp in g["lp_content"]], dtype=torch.float32
            )
            weights = self._token_importance(question, g["text"], tokens)
            weights = weights / (weights.sum() + 1e-8)
            scores.append(float((weights * entropies).sum()))

        return sum(scores) / len(scores)

    def _token_importance(
        self, question: str, text: str, tokens: list[str]
    ) -> "torch.Tensor":
        """R̃_T: 1 − cross_encoder(q+full, q+text_without_token) per token."""
        import torch  # noqa: PLC0415

        importance: list[float] = []
        for tok in tokens:
            reduced = text.replace(tok, "", 1)
            sim = float(
                self._cross_encoder.predict([[question + text, question + reduced]])[0]
            )
            importance.append(max(0.0, 1.0 - sim))
        return torch.tensor(importance, dtype=torch.float32)

    # ------------------------------------------------------------------
    # SENTSAR (Eq. 9-10) and full SAR (Eq. 11)
    # ------------------------------------------------------------------

    def _sentence_similarities(self, question: str, texts: list[str]) -> list[list[float]]:
        """Pairwise sentence similarities via cross-encoder with question prefix.

        Mirrors get_sentence_similarities.py from the original SAR codebase:
            gen_i = question + texts[i]
            gen_j = question + texts[j]
            similarity = cross_encoder.predict([gen_i, gen_j])

        Returns a K×K list-of-lists; diagonal is 1.0 (self-similarity unused).
        """
        K = len(texts)
        sim: list[list[float]] = [[1.0] * K for _ in range(K)]
        for i in range(K):
            for j in range(i + 1, K):
                pair = [question + texts[i], question + texts[j]]
                val = float(self._cross_encoder.predict([pair])[0])
                sim[i][j] = val
                sim[j][i] = val
        return sim

    def _sentsar(
        self,
        generations: list[dict],
        use_token_weights: bool,
        question: str,
    ) -> float:
        """Compute SENTSAR or full SAR score.

        SENTSAR (Eq. 9-10):
            E_S(s_j) = -log(p(s_j|x) + Σ_{k≠j} g(s_j,s_k)·p(s_k|x) / t)
            score    = (1/K) Σ_j E_S(s_j)

        Full SAR (Eq. 11):
            Same formula but p(s_j|x) replaced by p'(s_j|x) = exp(-TOKENSAR(s_j)).

        Numerical stability: all probabilities are shifted by exp(max_log_p)
        so they stay in (0, 1].  The threshold absorbs this constant offset.
        """
        import torch  # noqa: PLC0415

        K = len(generations)
        texts = [g["text"] for g in generations]

        # Pairwise sentence similarities g(s_i, s_j) via cross-encoder + question prefix
        sim_matrix = self._sentence_similarities(question, texts)

        # log p(s_j|x) — either raw (SENTSAR) or token-shifted (full SAR)
        if use_token_weights:
            # Full SAR: log p'(s_j|x) = -TOKENSAR(s_j, x)
            log_p = []
            for g in generations:
                tokens = [lp.token for lp in g["lp_content"]]
                entropies = torch.tensor(
                    [-lp.logprob for lp in g["lp_content"]], dtype=torch.float32
                )
                weights = self._token_importance(question, g["text"], tokens)
                weights = weights / (weights.sum() + 1e-8)
                log_p.append(-float((weights * entropies).sum()))
        else:
            # SENTSAR: log p(s_j|x) = sum of token log-probs
            log_p = [g["log_prob"] for g in generations]

        # Numerical stability: shift so max log-prob maps to 0
        max_lp = max(log_p)
        probs = [math.exp(lp - max_lp) for lp in log_p]  # in (0, 1]

        # Compute E_S(s_j) for each generation j (no clamping — matches paper)
        scores: list[float] = []
        for j in range(K):
            relevance = sum(
                sim_matrix[j][k] * probs[k]
                for k in range(K)
                if k != j
            )
            val = probs[j] + relevance / self.sar_temperature
            scores.append(-math.log(max(val, 1e-300)))

        return sum(scores) / K

    # ------------------------------------------------------------------
    # Consistency (no-logprobs fallback, not from paper)
    # ------------------------------------------------------------------

    def _consistency_score(
        self, question: str, client: Any, model: str, max_tokens: int
    ) -> SARResult:
        """1 − mean pairwise cosine similarity across K stochastic samples."""
        responses: list[str] = []
        for _ in range(self.n_samples):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                responses.append(resp.choices[0].message.content or "")
            except Exception as exc:
                logger.warning("Probe generation failed: %s", exc)

        n = len(responses)
        if n < 2:
            return SARResult(
                score=float("inf"), method="consistency",
                n_generations=n, needs_retrieval=True,
            )

        if self._sent_encoder is not None:
            from sentence_transformers import util as st_util  # noqa: PLC0415
            emb = self._sent_encoder.encode(responses, convert_to_tensor=True)
            sim_matrix = st_util.cos_sim(emb, emb)
            pairwise_sim = float((sim_matrix.sum() - n) / (n * (n - 1)))
            score = 1.0 - pairwise_sim
        else:
            score = self._rouge1_inconsistency(responses)

        return SARResult(
            score=score,
            method="consistency",
            n_generations=n,
            needs_retrieval=score > self.threshold,
        )

    @staticmethod
    def _rouge1_inconsistency(responses: list[str]) -> float:
        def f1(a: str, b: str) -> float:
            ta, tb = set(a.lower().split()), set(b.lower().split())
            if not ta or not tb:
                return 0.0
            return 2 * len(ta & tb) / (len(ta) + len(tb))

        n = len(responses)
        pairs = n * (n - 1) / 2
        total = sum(
            f1(responses[i], responses[j])
            for i in range(n)
            for j in range(i + 1, n)
        )
        return 1.0 - (total / pairs if pairs else 0.0)
