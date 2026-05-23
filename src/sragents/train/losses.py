"""Loss functions for cross-encoder fine-tuning.

Three options:

* ``bce``       — pointwise binary cross-entropy (default; matches HYRR).
* ``pairwise``  — margin ranking loss over (positive, negative) pairs
                  grouped by query.
* ``listwise``  — softmax-CE over all candidates for a query (per-query group).

The trainer chooses one via config. Each callable consumes the same
``(logits, labels, query_ids)`` triple so they're interchangeable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, labels: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    """Pointwise BCE — ``query_ids`` is unused but accepted for symmetry."""
    return F.binary_cross_entropy_with_logits(logits.float(), labels.float())


def pairwise_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    """Margin loss over (pos, neg) pairs sharing a query_id.

    For each query group, every positive vs. every negative contributes
    ``max(0, margin - s_pos + s_neg)``. Falls back to BCE if the batch
    has no usable pairs.
    """
    loss = logits.new_zeros(())
    count = 0
    for qid in torch.unique(query_ids):
        mask = query_ids == qid
        s = logits[mask]
        l = labels[mask]
        pos = s[l > 0.5]
        neg = s[l < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = margin - pos.unsqueeze(1) + neg.unsqueeze(0)
        loss = loss + F.relu(diff).mean()
        count += 1
    if count == 0:
        return bce_loss(logits, labels, query_ids)
    return loss / count


def listwise_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-query softmax-CE.

    For each query group, target = labels normalized to a distribution
    over candidates (uniform over positives, zero on negatives). Falls
    back to BCE if any group has zero positives.
    """
    loss = logits.new_zeros(())
    count = 0
    for qid in torch.unique(query_ids):
        mask = query_ids == qid
        s = logits[mask]
        l = labels[mask].float()
        if l.sum() <= 0:
            continue
        target = l / l.sum()
        log_p = F.log_softmax(s, dim=0)
        loss = loss + -(target * log_p).sum()
        count += 1
    if count == 0:
        return bce_loss(logits, labels, query_ids)
    return loss / count


def infonce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss per query group.

    For each query group with at least 1 positive:
        L_q = -log( sum_{i in pos} exp(s_i / τ) /  sum_{j} exp(s_j / τ) )

    Equivalent to a temperatured listwise softmax where positive mass
    is summed before the log. Lower τ -> sharper distribution (harder
    pressure on hardest negatives).

    Falls back to BCE if no group has any positives.
    """
    loss = logits.new_zeros(())
    count = 0
    for qid in torch.unique(query_ids):
        mask = query_ids == qid
        s = logits[mask] / temperature
        l = labels[mask].float()
        if l.sum() <= 0:
            continue
        # logsumexp(pos) - logsumexp(all)
        # mask out negatives in pos-LSE by adding -inf where label = 0
        neg_inf_mask = (l < 0.5).float() * (-1e9)
        log_num = torch.logsumexp(s + neg_inf_mask, dim=0)
        log_den = torch.logsumexp(s, dim=0)
        loss = loss + (log_den - log_num)
        count += 1
    if count == 0:
        return bce_loss(logits, labels, query_ids)
    return loss / count


LOSSES = {
    "bce": bce_loss,
    "pairwise": pairwise_margin_loss,
    "listwise": listwise_softmax_loss,
    "infonce": infonce_loss,
}


def get_loss(name: str):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; available: {list(LOSSES)}")
    return LOSSES[name]
