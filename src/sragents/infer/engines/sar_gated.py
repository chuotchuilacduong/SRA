"""SAR-gated engine: adaptive retrieval via uncertainty estimation.

Flow for each instance:

  1. **Probe** — call the LLM without skills using a short ``ue_max_tokens``
     budget to measure its uncertainty (SAR score).
  2. **Gate** — compare the score against ``ue_threshold``:
       - uncertain (score > threshold) *and* skills available → re-run with
         the full skill context injected.
       - confident (score ≤ threshold) *or* no skills → return the skill-free
         final answer directly.
  3. **Record** — store the UE score and decision in ``InferenceResult.meta``
     for offline analysis.

The probe budget (``ue_max_tokens``) should be short (default 256) to keep
latency low; the model only needs to produce enough tokens for the semantic
signal, not a complete answer.

Five UE modes are available (pass via ``--engine-arg ue_mode=...``):

  ``sent_sar``  (default, recommended)
      SENTSAR (Eq. 9-10 from paper): -log(p(s_j|x) + sentence_relevance/t).
      Requires logprobs + a local sentence encoder (all-MiniLM-L6-v2).
      Best accuracy/speed trade-off; paper shows it beats PE and SE.

  ``pe``
      Predictive Entropy baseline: mean neg-log-prob per token.
      Requires logprobs; no local model.

  ``token_sar``
      TOKENSAR (Eq. 6-8): token-importance-weighted neg-log-prob.
      Requires logprobs + local cross-encoder.  Slowest (O(T) encoder calls).

  ``sar``
      Full SAR (Eq. 11): combines token- and sentence-level shifting.
      Requires logprobs + cross-encoder + sentence encoder.

  ``consistency``
      No-logprobs fallback: pairwise semantic dissimilarity across N samples.
      Works with any API; score in [0, 1]; threshold ~0.3–0.6.
      Automatically engaged when logprobs are unavailable (e.g. Timely API).

CLI usage::

    sragents infer \\
        --engine sar_gated \\
        --provider topk \\
        --engine-arg ue_mode=predictive_entropy \\
        --engine-arg ue_threshold=1.5 \\
        --engine-arg ue_n_samples=5
"""

from sragents.infer.base import InferenceResult, register_engine
from sragents.llm import chat, get_extra_body
from sragents.prompts import build_prompt
from sragents.ue.sar import SARScorer


@register_engine("sar_gated")
class SARGatedEngine:
    """Adaptive retrieval engine gated by SAR uncertainty estimation.

    Args:
        temperature: Sampling temperature for the *final* answer generation.
        max_tokens: Max tokens for the final answer.
        thinking: Enable chain-of-thought/thinking mode on supporting models.
        ue_mode: SAR scoring mode.  See module docstring.
        ue_n_samples: Number of stochastic probe generations.
        ue_temperature: Sampling temperature for probe generations.
        ue_threshold: Uncertainty threshold above which skills are injected.
        ue_max_tokens: Max tokens for each probe generation (keep short).
    """

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        thinking: bool = False,
        ue_mode: str = "sent_sar",
        ue_n_samples: int = 5,
        ue_temperature: float = 0.7,
        ue_threshold: float = 1.5,
        ue_max_tokens: int = 256,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.ue_max_tokens = ue_max_tokens
        self._scorer = SARScorer(
            mode=ue_mode,
            n_samples=ue_n_samples,
            temperature=ue_temperature,
            threshold=ue_threshold,
        )

    def run(
        self,
        instance: dict,
        skills: list[dict],
        client,
        model: str,
        **kwargs,
    ) -> InferenceResult:
        # Step 1: probe without skills to measure uncertainty
        _, probe_prompt = build_prompt(instance, skills=[])
        ue_result = self._scorer.score(
            probe_prompt, client, model, max_tokens=self.ue_max_tokens
        )

        extra = get_extra_body(model, thinking=self.thinking)
        use_skills = ue_result.needs_retrieval and bool(skills)

        # Step 2: final answer — with or without skills
        if use_skills:
            skill_texts = [s["content"] for s in skills if s.get("content")]
            system, user = build_prompt(instance, skills=skill_texts)
            skill_ids = [s["skill_id"] for s in skills]
        else:
            system, user = build_prompt(instance, skills=[])
            skill_ids = []

        try:
            model_output = chat(
                client,
                model,
                user,
                system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra,
            )
        except Exception as exc:
            # Context overflow when skills are injected — fall back to skill-free call
            if use_skills and ("context length" in str(exc).lower() or "400" in str(exc)):
                import logging
                logging.getLogger(__name__).warning(
                    "Context overflow with skills; retrying without skills. (%s)", exc
                )
                system, user = build_prompt(instance, skills=[])
                skill_ids = []
                use_skills = False
                model_output = chat(
                    client,
                    model,
                    user,
                    system=system,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=extra,
                )
            else:
                raise

        return InferenceResult(
            raw_output=model_output,
            transcript=None,
            skill_ids_used=skill_ids,
            meta={
                "ue_score": ue_result.score,
                "ue_method": ue_result.method,
                "ue_n_generations": ue_result.n_generations,
                "used_retrieval": use_skills,
            },
        )
