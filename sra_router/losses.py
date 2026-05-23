"""Losses used by the bi-encoder trainer."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_with_hard_negs(
    q_emb: torch.Tensor,           # [B, d]
    p_emb: torch.Tensor,           # [B, d]
    hn_emb: torch.Tensor,          # [B*N, d]
    n_hard_per_q: int,
    tau: float = 0.05,
) -> torch.Tensor:
    """In-batch InfoNCE + per-query hard negatives.

    The denominator for query i contains:
      - its own positive (col i in the in-batch block),
      - all other in-batch positives (acting as in-batch negatives),
      - the i-th block of ``n_hard_per_q`` hard negatives.

    Out-of-batch positives belonging to other queries' hard negs are NOT
    shared across queries; this is the standard per-query hard-negative
    pattern used by SkillRouter §3.1.
    """
    B = q_emb.shape[0]
    in_batch_logits = q_emb @ p_emb.T / tau          # [B, B]
    hn_logits = q_emb @ hn_emb.T / tau               # [B, B*N]
    # Mask out hard negatives belonging to other queries so each query only
    # competes against its own N hard negs.
    if n_hard_per_q > 0:
        mask = torch.full((B, B * n_hard_per_q), float("-inf"), device=q_emb.device)
        for i in range(B):
            mask[i, i * n_hard_per_q:(i + 1) * n_hard_per_q] = 0.0
        hn_logits = hn_logits + mask
        logits = torch.cat([in_batch_logits, hn_logits], dim=1)
    else:
        logits = in_batch_logits
    labels = torch.arange(B, device=q_emb.device)
    return F.cross_entropy(logits, labels)
