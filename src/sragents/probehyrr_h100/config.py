"""Constants and paths for the CE-ProbeHYRR H100 pipeline (single source of truth)."""

from __future__ import annotations

import math
from pathlib import Path

from sragents.config import PROJECT_ROOT, RESULTS_DIR

# --- scope ---------------------------------------------------------------
# valid_4 first build (TheoremQA + LogicBench + MedCalc-Bench + CHAMP).
# ToolQA / BigCodeBench are phase-2 (need tool-loop / execution sandbox).
VALID_4 = ["theoremqa", "logicbench", "medcalcbench", "champ"]

# The split protocol whose train/dev/test we adopt. query_gen is the usable one
# (~70/10/20); no_leak/skill_gen has an EMPTY train set and cannot be used.
SPLIT_PROTOCOL = "query_gen"
SPLITS = ["train", "dev", "test"]

# --- generator -----------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-8B"
ENABLE_THINKING = False
TEMPERATURE = 0.0
TOP_P = 1.0

# Per-dataset output caps.
#
# 2026-06-16 RAISED from the spec §2.2 values (logicbench 128 / medcalc 256 / champ 512).
# Root cause of the v0 label defect: enable_thinking=False does NOT make Qwen3-8B terse —
# it still writes long step-by-step working and ignores "return only the final answer", so
# the small caps cut off the FINAL answer. The verifier then extracts a wrong intermediate
# token (a step index "4.", an input date) -> verifier_score=0 -> systematic FALSE NEGATIVES.
# Measured correct-rate truncated-vs-complete (v0): champ 2%/71%, logicbench 29%/83%,
# medcalc 8%/58%. theoremqa (cap already 1024) truncated only 2% and is the only clean set.
# These caps target truncated_rate < MAX_TRUNCATED_RATE on every dataset. Re-confirm on a
# pilot and tune down only if throughput needs it AND the truncation gate still passes.
MAX_NEW_TOKENS = {
    "logicbench": 1024,
    "medcalcbench": 1024,
    "champ": 1536,
    "theoremqa": 1024,
}
DEFAULT_MAX_NEW_TOKENS = 1024

# Acceptance gate: max fraction of generations allowed to hit the token cap
# (finish_reason="length"). Above this, labels are silently corrupted (see note above).
MAX_TRUNCATED_RATE = 0.10

# --- M4 candidate pool ----------------------------------------------------
M4_POOL_TMPL = "results/pool/hybrid_km_alpha30-{ds}.json"  # top-100 RRF+KMeans pool == M4
TOP50 = 50
# Sentinel rank for a candidate absent from a branch's ranked list (bm25/dense).
MISSING_RANK = TOP50 + 1

# --- splits on disk -------------------------------------------------------
SPLIT_FILE_TMPL = "results/splits/{ds}-{protocol}.json"

# --- output layout --------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
OUT_SPLITS_DIR = DATA_DIR / "splits"                 # data/splits/{split}.jsonl
OUT_CORPUS_JSONL = DATA_DIR / "corpus" / "skills.jsonl"
M4_CACHE_DIR = RESULTS_DIR / "m4"                     # results/m4/m4_top50_{split}.jsonl
RUN_ROOT = RESULTS_DIR / "ce_probehyrr_h100"
TASKS_DIR = RUN_ROOT / "tasks"
GEN_DIR = RUN_ROOT / "generations"
VERIFIED_DIR = RUN_ROOT / "verified"
LOGS_DIR = RUN_ROOT / "logs"
STATE_DIR = RUN_ROOT / "state"


def m4_cache_path(split: str) -> Path:
    return M4_CACHE_DIR / f"m4_top50_{split}.jsonl"


def split_jsonl_path(split: str) -> Path:
    return OUT_SPLITS_DIR / f"{split}.jsonl"


def anchor_tasks_path(split: str) -> Path:
    return TASKS_DIR / f"anchor_probe_tasks_{split}.jsonl"


def generation_config(dataset: str) -> dict:
    """The per-task generation_config block stamped into every task row."""
    return {
        "model": MODEL_ID,
        "enable_thinking": ENABLE_THINKING,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS.get(dataset, DEFAULT_MAX_NEW_TOKENS),
    }


# --- Stages 4-9: micro-batch UCB candidate probing (spec §5) -------------
UCB_C = math.sqrt(2)          # exploration constant (§5.6)
WARMUP_PER_ARM = 50           # global warm-up probes per arm before true UCB (§5.6)
BUDGET_PER_QUERY = 3          # B: max candidate probes per query (§5.2)
NUM_ROUNDS = 3                # micro-batch UCB rounds (§5.2)
RANK_FRIEND_CUTOFF = 10       # rank_m4 <= this looks "convincing" -> strict/weak false-friend split

# Arms for Option A (§5.3). A1 (m4_top1) is already an anchor and never scheduled.
UCB_ARMS = [
    "m4_top1_anchor_already_done",
    "rank_2_to_10_non_gold",
    "rank_11_to_50_non_gold",
    "above_gold_boundary",
    "below_gold_boundary",
    "same_cluster_as_gold_non_gold",
    "high_bm25_low_dense",
    "high_dense_low_bm25",
    "no_load_high_rank_risk",
    "gold_absent_high_rank",
    "random_tail_control",
]
# Arms UCB is allowed to schedule (exclude the already-done anchor arm).
SCHEDULABLE_ARMS = [a for a in UCB_ARMS if a != "m4_top1_anchor_already_done"]

# Label -> reward (§5.7). Higher = more informative for data acquisition.
REWARD_TABLE = {
    "helpful": 1.0,
    "gold_verified_helpful": 1.0,
    "harmful": 1.2,
    "strict_false_friend": 1.0,
    "bad_candidate_when_skill_needed": 0.8,
    "no_load_preferred": 0.6,
    "safe_but_unneeded": 0.3,
    "weak_false_friend": 0.3,
    "neutral": 0.0,
    "ambiguous": -0.5,
    "unverifiable": -1.0,
}
# Labels whose skill should rank HIGH (positive) for the utility cross-encoder.
POSITIVE_LABELS = {"helpful", "gold_verified_helpful"}
# Labels dropped from the training pairs entirely (too noisy to learn from).
DROP_LABELS = {"ambiguous", "unverifiable", "gold_verified_harmful"}


def query_regime_labels_path(split: str) -> Path:
    return RUN_ROOT / f"query_regime_labels_{split}.jsonl"


def ucb_tasks_path(round_: int, split: str) -> Path:
    return TASKS_DIR / f"ucb_probe_tasks_round_{round_}_{split}.jsonl"


def ucb_outputs_path(round_: int, split: str) -> Path:
    return GEN_DIR / f"ucb_outputs_round_{round_}_{split}.jsonl"


def ucb_verified_path(round_: int, split: str) -> Path:
    return VERIFIED_DIR / f"ucb_verified_round_{round_}_{split}.jsonl"


def ucb_probe_logs_path(split: str) -> Path:
    return LOGS_DIR / f"ucb_candidate_probe_logs_{split}.jsonl"


def ucb_state_path(split: str) -> Path:
    return STATE_DIR / f"ucb_state_{split}.json"


def utility_pairs_path(split: str) -> Path:
    return RUN_ROOT / f"utility_pairs_{split}.jsonl"


def utility_train_groups_path(split: str) -> Path:
    return RUN_ROOT / f"utility_train_groups_{split}.jsonl"
