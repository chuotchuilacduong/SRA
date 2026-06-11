"""Constants — single source of truth for the skill-probe hidden-state study."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "Qwen/Qwen3-4B"
# Asserted at model-load time so a wrong checkpoint fails fast (see hf_generator).
EXPECT_NUM_LAYERS = 36          # transformer blocks; hidden_states tuple has +1 (embeddings) = 37
EXPECT_HIDDEN_SIZE = 2560

DATASETS = ["logicbench", "medcalcbench", "theoremqa", "champ"]
METHODS = ["M4", "CE-HYRR"]

N_QUERIES = 10
SEED = 42
TOP_K = 10                      # candidates probed per (query, method)
MAX_NEW_TOKENS = 2048           # matches probe_once.py default
LAYER_POLICY = "all"            # "all" -> keep 37 layers; "every4" -> [0,4,...,36]

OUTPUT_ROOT = ROOT / "results" / "skill_probe_hidden"

# Candidate sources (reuse the same files the rest of the repo uses).
POOL_TMPL = "results/pool/hybrid_km_alpha30-{ds}.json"            # M4, top-100, full split
RERANK_TMPL = "results/rerank/test-kmeans_M30-{ds}-ce-v3-top20.json"  # CE-HYRR, top-20, TEST split

# "score" means different things per method — kept for provenance, never compared numerically.
SCORE_MEANING = {"M4": "hybrid blend 0..1 (higher=better)", "CE-HYRR": "CE logit (higher=better)"}


def layer_indices(num_layers_plus_embed: int, policy: str = LAYER_POLICY) -> list[int]:
    """Which layer indices to keep from the [num_layers+1, hidden] stack."""
    n = num_layers_plus_embed
    if policy == "every4":
        return list(range(0, n, 4)) + ([n - 1] if (n - 1) % 4 else [])
    return list(range(n))  # "all"
