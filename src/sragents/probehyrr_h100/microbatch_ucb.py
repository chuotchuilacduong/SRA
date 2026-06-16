"""Micro-batch UCB engine for candidate scheduling (spec §5).

CPU-only library (no CLI). Used by:
  - build_ucb_tasks.py  — pick an arm per query, then the best candidate in it.
  - label_builder.py    — map a label to a reward and update arm statistics.

Design note (micro-batch, not online UCB): within one round all selections use the
arm statistics frozen at the previous round's end. To keep warm-up and exploration
from collapsing onto a single arm within a round, the scheduler increments a
*provisional* per-arm count as it assigns probes (see build_ucb_tasks). The helpers
here therefore take an explicit ``working_n`` / ``t`` so the caller controls that.
"""

from __future__ import annotations

import math

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import load_json, sha256_short

# ---------------------------------------------------------------------------
# Candidate eligibility (§5.4)
# ---------------------------------------------------------------------------

def _imputed(cand: dict, key: str) -> int:
    """bm25_rank / dense_rank with the MISSING_RANK sentinel for nulls."""
    v = cand.get(key)
    return C.MISSING_RANK if v is None else int(v)


def gold_features(m4_rec: dict) -> tuple[int | None, int | None]:
    """Return (gold_rank_m4, gold_cluster_id) using the in-pool gold entry if present."""
    gold_rank = m4_rec.get("gold_rank_m4")
    gold_cluster = None
    for c in m4_rec.get("top50", []):
        if c.get("is_gold"):
            gold_cluster = c.get("cluster_id")
            if gold_rank is None:
                gold_rank = c.get("rank_m4")
            break
    return gold_rank, gold_cluster


def eligible_candidates(m4_rec: dict, probed_skill_ids: set[str], corpus: dict) -> list[dict]:
    """Top-50 entries probeable by UCB (§5.4 skip rules).

    Skip: gold (probed as anchor), m4_top1 rank-1 (probed as anchor), already-probed
    skills, empty skill content, duplicate skill_id.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for c in m4_rec.get("top50", []):
        sid = c["skill_id"]
        if c.get("is_gold"):
            continue
        if c.get("rank_m4") == 1:           # m4_top1 anchor
            continue
        if sid in probed_skill_ids or sid in seen:
            continue
        s = corpus.get(sid)
        if not (s and s.get("content")):    # empty skill content
            continue
        seen.add(sid)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Candidate priority inside an arm (§5.5)
# ---------------------------------------------------------------------------

def pick_candidate_for_arm(
    arm: str,
    eligible: list[dict],
    gold_rank: int | None,
    gold_cluster: int | None,
    gold_in_top50: bool,
    regime: dict,
    seed: str,
) -> dict | None:
    """Choose the single best eligible candidate for ``arm``, or None if the arm
    is empty / not applicable for this query."""

    def lowest_rank(cands):
        return min(cands, key=lambda c: c["rank_m4"]) if cands else None

    if arm == "rank_2_to_10_non_gold":
        c = [x for x in eligible if 2 <= x["rank_m4"] <= 10]
        return lowest_rank(c)

    if arm == "rank_11_to_50_non_gold":
        c = [x for x in eligible if 11 <= x["rank_m4"] <= 50]
        if not c:
            return None
        if gold_in_top50 and gold_rank is not None:
            return min(c, key=lambda x: (abs(x["rank_m4"] - gold_rank), x["rank_m4"]))
        return lowest_rank(c)

    if arm == "above_gold_boundary":
        if gold_rank is None:
            return None
        c = [x for x in eligible if x["rank_m4"] < gold_rank]
        return max(c, key=lambda x: x["rank_m4"]) if c else None  # closest above gold

    if arm == "below_gold_boundary":
        if gold_rank is None:
            return None
        c = [x for x in eligible if x["rank_m4"] > gold_rank]
        return min(c, key=lambda x: x["rank_m4"]) if c else None  # closest below gold

    if arm == "same_cluster_as_gold_non_gold":
        if gold_cluster is None:
            return None
        c = [x for x in eligible if x.get("cluster_id") == gold_cluster]
        return lowest_rank(c)

    if arm == "high_bm25_low_dense":
        c = [x for x in eligible if _imputed(x, "bm25_rank") < _imputed(x, "dense_rank")]
        if not c:
            return None
        # largest conflict, tie-break on lowest bm25 rank
        return max(c, key=lambda x: (_imputed(x, "dense_rank") - _imputed(x, "bm25_rank"),
                                     -_imputed(x, "bm25_rank")))

    if arm == "high_dense_low_bm25":
        c = [x for x in eligible if _imputed(x, "dense_rank") < _imputed(x, "bm25_rank")]
        if not c:
            return None
        return max(c, key=lambda x: (_imputed(x, "bm25_rank") - _imputed(x, "dense_rank"),
                                     -_imputed(x, "dense_rank")))

    if arm == "no_load_high_rank_risk":
        if not regime.get("no_load_opportunity"):
            return None
        c = [x for x in eligible if 2 <= x["rank_m4"] <= 10]
        return lowest_rank(c)

    if arm == "gold_absent_high_rank":
        if not regime.get("gold_absent_top50"):
            return None
        c = [x for x in eligible if 1 <= x["rank_m4"] <= 10]
        return lowest_rank(c)

    if arm == "random_tail_control":
        c = [x for x in eligible if 21 <= x["rank_m4"] <= 50]
        if not c:
            return None
        # deterministic "random": seed by qid/round so builds are reproducible/resumable
        idx = int(sha256_short(seed, 12), 16) % len(c)
        return sorted(c, key=lambda x: x["rank_m4"])[idx]

    return None


def candidate_actions(m4_rec, regime, probed_skill_ids, corpus, seed) -> dict[str, dict]:
    """Map each schedulable arm -> its chosen candidate (only arms with a candidate)."""
    eligible = eligible_candidates(m4_rec, probed_skill_ids, corpus)
    if not eligible:
        return {}
    gold_rank, gold_cluster = gold_features(m4_rec)
    gold_in_top50 = bool(m4_rec.get("gold_in_top50", False))
    actions: dict[str, dict] = {}
    for arm in C.SCHEDULABLE_ARMS:
        cand = pick_candidate_for_arm(arm, eligible, gold_rank, gold_cluster,
                                      gold_in_top50, regime, f"{seed}::{arm}")
        if cand is not None:
            actions[arm] = cand
    return actions


# ---------------------------------------------------------------------------
# UCB score + arm selection (§5.6)
# ---------------------------------------------------------------------------

def ucb_score(mean_reward: float, n_arm: float, t: float) -> float:
    if n_arm <= 0:
        return float("inf")          # force exploration of unseen arms
    return mean_reward + C.UCB_C * math.sqrt(math.log(max(t, 1.0)) / n_arm)


def select_arm(candidate_arms, stats, working_n, t) -> str:
    """Pick an arm among ``candidate_arms`` (those with a candidate for this query).

    Warm-up first: if any candidate arm is still below WARMUP_PER_ARM (counting the
    in-round provisional ``working_n``), take the least-probed one. Otherwise true UCB.
    """
    def wn(arm):
        return stats["arm_stats"].get(arm, {}).get("n", 0) + working_n.get(arm, 0)

    under = [a for a in candidate_arms if wn(a) < C.WARMUP_PER_ARM]
    if under:
        return min(under, key=lambda a: (wn(a), C.SCHEDULABLE_ARMS.index(a)))

    def score(arm):
        mean = stats["arm_stats"].get(arm, {}).get("mean_reward", 0.0)
        return ucb_score(mean, wn(arm), t)

    return max(candidate_arms, key=lambda a: (score(a), -C.SCHEDULABLE_ARMS.index(a)))


# ---------------------------------------------------------------------------
# Reward + state (§5.7)
# ---------------------------------------------------------------------------

def reward_for_label(label: str) -> float:
    return C.REWARD_TABLE.get(label, 0.0)


def blank_state(split: str) -> dict:
    def arms():
        return {a: {"n": 0, "sum_reward": 0.0, "mean_reward": 0.0} for a in C.SCHEDULABLE_ARMS}
    return {
        "split": split,
        "round": 0,
        "global_t": 0,
        "arm_stats": arms(),
        "arm_stats_by_dataset": {},
    }


def load_state(path, split: str) -> dict:
    from pathlib import Path
    if Path(path).exists():
        return load_json(path)
    return blank_state(split)


def save_state(path, state: dict) -> None:
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2))


def update_stats(state: dict, arm: str, dataset: str, reward: float) -> None:
    """Apply one observed reward to global + per-dataset arm statistics."""
    def bump(table: dict):
        s = table.setdefault(arm, {"n": 0, "sum_reward": 0.0, "mean_reward": 0.0})
        s["n"] += 1
        s["sum_reward"] += reward
        s["mean_reward"] = s["sum_reward"] / s["n"]

    bump(state["arm_stats"])
    by_ds = state["arm_stats_by_dataset"].setdefault(dataset, {})
    bump(by_ds)
    state["global_t"] += 1
