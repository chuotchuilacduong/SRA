"""HYRR-style hybrid hard-negative mining for cross-encoder training.

Each training pair is one (query, skill, label) tuple.

For every gold-bearing query we emit:

* the positive(s)                                  ``label=1``, ``source=positive``
* ``n_bm25_hard``   BM25-ranked candidates         ``source=bm25_hard``
* ``n_bge_hard``    BGE-ranked candidates          ``source=bge_hard``
                    (falls back to ``rrf_hard`` if BGE provenance absent)
* ``n_random``      uniformly-sampled corpus ids   ``source=random``
* (optional) ``n_cluster_hard`` from same cluster  ``source=cluster_hard``

Hard constraint: **no negative may share a skill_id with the query's
gold annotations** — verified at sample time, not after the fact. This is
critical for multi-label datasets where naive top-rank mining leaks
positives into negatives and silently inflates loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from sragents.retrieve.schema import load as load_results

# Each pool entry: {skill_id, score, rank, bm25_rank?, bge_rank?, rrf_rank?, ...}
PoolEntry = dict


@dataclass
class NegativeMix:
    bm25_hard: int = 4
    bge_hard: int = 4
    random: int = 2
    cluster_hard: int = 0

    def total(self) -> int:
        return self.bm25_hard + self.bge_hard + self.random + self.cluster_hard


@dataclass
class TrainPair:
    instance_id: str
    skill_id: str
    label: int
    negative_source: str
    question: str
    skill: dict
    gold_skill_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "skill_id": self.skill_id,
            "label": self.label,
            "negative_source": self.negative_source,
            "question": self.question,
            "skill": self.skill,
            "gold_skill_ids": self.gold_skill_ids,
        }


@dataclass
class NegativeSampler:
    """Mine hybrid hard negatives from a candidate pool.

    Attrs:
        mix: How many of each kind to draw.
        seed: RNG seed for random + cluster sampling.
        clusters: Optional ``skill_id -> cluster_id``. If absent, the
            ``cluster_hard`` slot degrades to extra ``random`` draws.
        rrf_fallback_for_bge: If a pool entry lacks ``bge_rank`` (i.e. the
            extended-track wasn't used to build it), treat ``rrf_rank``
            as the dense-side ranker for sampling purposes.
    """

    mix: NegativeMix = field(default_factory=NegativeMix)
    seed: int = 42
    clusters: dict[str, int] | None = None
    rrf_fallback_for_bge: bool = True

    # ------------------------------------------------------------ sampling

    def sample_for_query(
        self,
        instance_id: str,
        question: str,
        gold_skill_ids: list[str],
        pool: list[PoolEntry],
        corpus: dict[str, dict],
    ) -> list[TrainPair]:
        rng = random.Random(_query_seed(self.seed, instance_id))
        gold_set = set(gold_skill_ids)
        pairs: list[TrainPair] = []

        # Positives — one row per gold (skip those missing from corpus).
        for gid in gold_skill_ids:
            sk = corpus.get(gid)
            if sk is None:
                continue
            pairs.append(TrainPair(
                instance_id=instance_id, skill_id=gid, label=1,
                negative_source="positive", question=question,
                skill=sk, gold_skill_ids=gold_skill_ids,
            ))

        used: set[str] = set(gold_set)

        # BM25-hard
        bm25_ranked = _sort_by(pool, "bm25_rank")
        for sid in _take_n(bm25_ranked, used, gold_set, self.mix.bm25_hard):
            pairs.append(_neg(instance_id, sid, question, corpus, gold_skill_ids, "bm25_hard"))
            used.add(sid)

        # BGE-hard (or RRF-hard fallback)
        bge_field = "bge_rank" if any("bge_rank" in p for p in pool) else (
            "rrf_rank" if self.rrf_fallback_for_bge else None
        )
        if bge_field is not None:
            bge_ranked = _sort_by(pool, bge_field)
            source = "bge_hard" if bge_field == "bge_rank" else "rrf_hard"
            for sid in _take_n(bge_ranked, used, gold_set, self.mix.bge_hard):
                pairs.append(_neg(instance_id, sid, question, corpus, gold_skill_ids, source))
                used.add(sid)

        # Random — uniformly from corpus, excluding golds and already-used
        corpus_ids = list(corpus.keys())
        wanted = self.mix.random
        attempts = 0
        while wanted > 0 and attempts < wanted * 20:
            attempts += 1
            sid = rng.choice(corpus_ids)
            if sid in gold_set or sid in used:
                continue
            pairs.append(_neg(instance_id, sid, question, corpus, gold_skill_ids, "random"))
            used.add(sid)
            wanted -= 1

        # Cluster-hard — same cluster as a gold but not gold itself.
        if self.mix.cluster_hard and self.clusters:
            cluster_targets: set[int] = {
                self.clusters[g] for g in gold_skill_ids if g in self.clusters
            }
            cands = [s for s in corpus_ids
                     if self.clusters.get(s) in cluster_targets
                     and s not in gold_set and s not in used]
            rng.shuffle(cands)
            for sid in cands[: self.mix.cluster_hard]:
                pairs.append(_neg(instance_id, sid, question, corpus, gold_skill_ids, "cluster_hard"))
                used.add(sid)

        return pairs


# ---------------------------------------------------------------- helpers


def _query_seed(base: int, instance_id: str) -> int:
    return (base * 1_000_003 + hash(instance_id)) & 0x7FFF_FFFF


def _sort_by(pool: list[PoolEntry], field: str) -> list[str]:
    """Return skill_ids ranked by ``field`` ascending (rank 1 first).

    Pool entries that lack ``field`` are dropped from this view (we don't
    want them polluting BM25-hard with non-BM25 candidates, etc.).
    """
    items = [(p[field], p["skill_id"]) for p in pool if field in p]
    items.sort(key=lambda x: x[0])
    return [sid for _, sid in items]


def _take_n(ranked: Iterable[str], used: set[str], gold: set[str], n: int) -> list[str]:
    """Take the next ``n`` ranked ids that are not gold and not already used."""
    out: list[str] = []
    for sid in ranked:
        if sid in gold or sid in used:
            continue
        out.append(sid)
        used.add(sid)  # within-call dedup (caller adds again, idempotent)
        if len(out) == n:
            break
    return out


def _neg(instance_id, sid, question, corpus, gold, source) -> TrainPair:
    sk = corpus.get(sid, {"skill_id": sid})
    return TrainPair(
        instance_id=instance_id, skill_id=sid, label=0,
        negative_source=source, question=question,
        skill=sk, gold_skill_ids=gold,
    )


# ------------------------------------------------------- batch driver


def mine_from_pool(
    pool_path,
    corpus: dict[str, dict],
    instances_by_id: dict[str, dict],
    *,
    question_field: str = "question",
    mix: NegativeMix | None = None,
    seed: int = 42,
    clusters: dict[str, int] | None = None,
) -> list[TrainPair]:
    """Mine hybrid hard negatives from a saved pool JSON.

    ``pool_path`` is a :class:`sragents.retrieve.schema.RetrievalResults`
    file (i.e. produced by ``sragents build-pool``).
    """
    data = load_results(pool_path)
    sampler = NegativeSampler(
        mix=mix or NegativeMix(),
        seed=seed,
        clusters=clusters,
    )
    out: list[TrainPair] = []
    for r in data["results"]:
        inst = instances_by_id.get(r["instance_id"])
        if inst is None:
            continue
        gold = r.get("gold_skill_ids") or []
        if not gold:
            continue
        question = inst.get(question_field) or ""
        pool = r["retrieved"]
        out.extend(sampler.sample_for_query(
            instance_id=r["instance_id"],
            question=question,
            gold_skill_ids=gold,
            pool=pool,
            corpus=corpus,
        ))
    return out
