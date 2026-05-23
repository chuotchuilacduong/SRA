"""Phase 2 — 3-layer false negative filter.

Layers:
 1. Skill name dedup against gold names (case-folded).
 2. Trigram-Jaccard over the body > 0.6 → drop.
 3. Cosine similarity of base-encoder embeddings vs. any gold > 0.92 → drop.

Each layer is cheap relative to the next, so they run in order and short-circuit.
"""

from __future__ import annotations

import numpy as np

from sra_router.corpus import Skill, SkillCorpus


def _trigrams(s: str) -> set[str]:
    s = s.lower()
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def trigram_jaccard(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def filter_false_negatives(
    neg_ids: list[str],
    gold_ids: list[str],
    corpus: SkillCorpus,
    *,
    embeddings: dict[str, np.ndarray] | None = None,
    jaccard_threshold: float = 0.6,
    cosine_threshold: float = 0.92,
) -> tuple[list[str], dict[str, int]]:
    """Return (kept_ids, layer_dropcounts).

    ``embeddings`` is an optional ``{skill_id: l2-normalised vector}`` map. If
    omitted, layer 3 is skipped (which still leaves layers 1 + 2 active).
    """
    gold_skills: list[Skill] = [corpus[g] for g in gold_ids if g in corpus]
    gold_names = {g.name.lower() for g in gold_skills if g.name}

    drops = {"name_dup": 0, "trigram": 0, "cosine": 0}
    kept: list[str] = []
    gold_embs: np.ndarray | None = None
    if embeddings is not None:
        vecs = [embeddings[g] for g in gold_ids if g in embeddings]
        if vecs:
            gold_embs = np.stack(vecs)

    for nid in neg_ids:
        if nid not in corpus:
            continue
        n = corpus[nid]

        # Layer 1: name dedup
        if n.name and n.name.lower() in gold_names:
            drops["name_dup"] += 1
            continue

        # Layer 2: trigram Jaccard on body
        if any(trigram_jaccard(n.content, g.content) > jaccard_threshold for g in gold_skills):
            drops["trigram"] += 1
            continue

        # Layer 3: embedding cosine
        if gold_embs is not None and nid in embeddings:
            sim = float(np.max(gold_embs @ embeddings[nid]))
            if sim > cosine_threshold:
                drops["cosine"] += 1
                continue

        kept.append(nid)
    return kept, drops
