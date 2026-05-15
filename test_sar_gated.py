"""Smoke-test for the sar_gated engine.

Supports all UE modes: pe, sent_sar, token_sar, sar, consistency.
Works with any OpenAI-compatible backend (vLLM, OpenAI, Timely).

Usage (local vLLM):
    python test_sar_gated.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --api-base http://localhost:8000/v1 \\
        --ue-mode sent_sar \\
        --threshold 0.5

Usage (Timely API / no logprobs):
    python test_sar_gated.py --ue-mode consistency --threshold 0.45
"""

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from openai import OpenAI
from sragents.llm import create_llm_client
from sragents.infer.engines.sar_gated import SARGatedEngine
from sragents.infer.providers.topk import TopKProvider

INSTANCES_PATH = Path("SRA-Bench/instances/theoremqa.json")
RETRIEVAL_PATH = Path("results/retrieval_test/theoremqa-linearrag-1000.json")
CORPUS_PATH    = Path("SRA-Bench/corpus/corpus_1000.json")

_DEFAULT_THRESHOLD = {
    "pe":          2.0,
    "token_sar":   2.0,
    "sent_sar":    0.0,
    "sar":         0.0,
    "consistency": 0.45,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int,   default=5)
    ap.add_argument("--model",      default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--api-base",   default=None)
    ap.add_argument("--api-key",    default="EMPTY")
    ap.add_argument("--ue-mode",    default="sent_sar",
                    choices=["pe", "token_sar", "sent_sar", "sar", "consistency"])
    ap.add_argument("--threshold",  type=float, default=None)
    ap.add_argument("--n-samples",  type=int,   default=5)
    ap.add_argument("--topk",       type=int,   default=3)
    ap.add_argument("--max-tokens", type=int,   default=256)
    args = ap.parse_args()

    threshold = args.threshold if args.threshold is not None \
        else _DEFAULT_THRESHOLD[args.ue_mode]

    W = 72  # output width

    print(f"\n{'='*W}")
    print(f"  SAR-GATED ENGINE TEST")
    print(f"{'='*W}")
    print(f"  Model     : {args.model}")
    print(f"  API base  : {args.api_base or '(from .env)'}")
    print(f"  UE mode   : {args.ue_mode}")
    print(f"  Threshold : {threshold}  "
          f"(score > threshold → uncertain → use skills)")
    print(f"  Samples   : {args.n_samples} probe generations per instance")
    print(f"  Top-K     : {args.topk} skills")
    print(f"  Instances : {args.n}")
    print(f"{'='*W}\n")

    instances = json.loads(INSTANCES_PATH.read_text())[: args.n]

    if args.api_base:
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    else:
        client = create_llm_client()

    engine = SARGatedEngine(
        ue_mode=args.ue_mode,
        ue_n_samples=args.n_samples,
        ue_temperature=0.8,
        ue_threshold=threshold,
        ue_max_tokens=64,
        temperature=0.3,
        max_tokens=args.max_tokens,
    )

    provider = TopKProvider(
        source=str(RETRIEVAL_PATH),
        corpus_path=str(CORPUS_PATH),
        k=args.topk,
    )

    n_used_retrieval = 0
    t0 = time.time()

    for i, inst in enumerate(instances):
        iid       = inst["instance_id"]
        question  = inst["question"]
        answer    = inst["eval_data"].get("answer", "?")

        skills = provider.provide(inst)
        result = engine.run(inst, skills, client, model=args.model)

        m      = result.meta
        used   = m.get("used_retrieval", False)
        score  = m.get("ue_score", 0.0)
        method = m.get("ue_method", args.ue_mode)
        n_gen  = m.get("ue_n_generations", args.n_samples)
        if used:
            n_used_retrieval += 1

        # ── per-instance block ──────────────────────────────────────
        print(f"{'─'*W}")
        print(f"[{i+1}/{args.n}]  {iid}")
        print()

        # Question (wrapped)
        print("  QUESTION:")
        for line in textwrap.wrap(question, width=W-4):
            print(f"    {line}")
        print()

        # UE decision
        score_str = f"{score:.4f}" if score != float("inf") else "inf"
        decision  = "RETRIEVE (use skills)" if used else "direct  (no skills)"
        print(f"  UE SCORE  : {score_str}   [method={method}, K={n_gen}]")
        print(f"  THRESHOLD : {threshold}")
        print(f"  DECISION  : {decision}")
        if method != args.ue_mode:
            print(f"  FALLBACK  : {args.ue_mode} → {method}")
        print()

        # Model output (wrapped)
        model_out = (result.raw_output or "").strip()
        print("  MODEL OUTPUT:")
        for line in textwrap.wrap(model_out, width=W-4) or ["(empty)"]:
            print(f"    {line}")
        print()

        # Ground truth
        print(f"  CORRECT ANSWER : {answer}")
        print()

    # ── summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"{'='*W}")
    print(f"  SUMMARY")
    print(f"{'='*W}")
    print(f"  Used retrieval : {n_used_retrieval}/{args.n}  "
          f"({100*n_used_retrieval/args.n:.0f}%)")
    print(f"  Elapsed        : {elapsed:.1f}s  (~{elapsed/args.n:.1f}s per instance)")
    print(f"  Threshold      : {threshold}  "
          f"({'sent_sar scores are typically negative' if 'sar' in args.ue_mode else ''})")
    print()


if __name__ == "__main__":
    main()
