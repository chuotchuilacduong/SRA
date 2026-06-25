"""KL-distilled DE + CE + β=0.7 fusion reranker (Sơ đồ 3).

Extends :class:`BetaCEReranker` so the **CE checkpoint** comes from the
RocketQAv2 joint-trained ``cross_encoder/`` directory (i.e. the CE that
was KL-distilled jointly with a dual-encoder). The DE half of the pair
lives in ``dual_encoder/`` and is consumed by Stage 1 separately
(:mod:`sragents.retrieve.dense` with ``model_path=...``).

The β=0.7 score blend at inference is identical to Method 7:

    score★(s) = β · ĉe(s) + (1 − β) · Ŝ1(s),   β = 0.7

What changes vs Method 7 is **upstream**: now CE and Stage 1's DE side
were jointly trained with::

    L = KL(p̃_DE ‖ p̃_CE) + L_sup(CE, gold_idx)

so the two streams are no longer independent at training time — Stage 1
DE has been pulled toward the CE-defined relevance distribution.

This keeps the inference interface a drop-in for ``CrossEncoderReranker``.
"""
from __future__ import annotations

from pathlib import Path

from sragents.retrieve.beta_ce_rerank import BetaCEReranker
from sragents.retrieve.skill_packer import SkillPacker


class KLBetaCEReranker(BetaCEReranker):
    """β-blend reranker whose CE is the RocketQAv2 KL-distilled checkpoint.

    Conceptually the *only* novelty vs :class:`BetaCEReranker` is the
    factory that points at ``<rocketqa_out>/cross_encoder``. Kept as a
    distinct class so call sites and configs document the lineage.
    """

    @classmethod
    def from_rocketqav2(
        cls,
        model_dir: str | Path,
        *,
        beta: float = 0.7,
        device: str = "auto",
        max_length: int = 256,
        packer: SkillPacker | None = None,
        corpus: dict | None = None,
    ) -> "KLBetaCEReranker":
        """Load CE half of a RocketQAv2 joint checkpoint.

        Expects ``model_dir`` to contain ``cross_encoder/`` (and usually
        a sibling ``dual_encoder/`` consumed by Stage 1).
        """
        ce_dir = Path(model_dir) / "cross_encoder"
        if not ce_dir.exists():
            raise FileNotFoundError(
                f"{ce_dir} not found — pass the RocketQAv2 output root, not the CE dir"
            )
        return cls(
            model_name=str(ce_dir), device=device, max_length=max_length,
            packer=packer, corpus=corpus, beta=beta,
        )
