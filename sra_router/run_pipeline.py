"""End-to-end driver: build pairs → train encoder → evaluate.

Scaled-down for laptop / no-GPU execution. Defaults:
  * 4 single-label datasets only (TheoremQA + LogicBench + ToolQA + MedCalcBench)
  * BGE-small-en-v1.5 backbone (384-dim, 33M params)
  * Train pairs are mined from the SRA-Bench instances themselves with an 80/20
    demo split. NOT the paper protocol — see SRA_SKILL_ROUTER_RESULTS.md.

Run:
  python -m sra_router.run_pipeline --output-dir results/sra_router/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sra_router.corpus import SkillCorpus
from sra_router.evaluate import run_all
from sra_router.instances import load_instances, stratified_split
from sra_router.plot import plot_convergence
from sra_router.train_encoder import TrainConfig, train
from sra_router.training_data import build_training_pairs


DEFAULT_DATASETS = ("theoremqa", "logicbench", "toolqa", "medcalcbench")


def collect_instances(datasets: tuple[str, ...], instance_dir: Path):
    out = []
    for ds in datasets:
        out.extend(load_instances(instance_dir / f"{ds}.json"))
    return out


def subsample(items: list, n: int, seed: int = 0):
    if n <= 0 or len(items) <= n:
        return items
    import random as _r
    rng = _r.Random(seed)
    return rng.sample(items, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bench/corpus/corpus.json")
    ap.add_argument("--instances-dir", default="data/bench/instances")
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-cap", type=int, default=600,
                    help="Cap on training instances (per fold). 0 = no cap.")
    ap.add_argument("--eval-cap", type=int, default=200,
                    help="Cap on eval instances. 0 = no cap.")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-hard-negs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {args.corpus} ...")
    corpus = SkillCorpus.from_json(args.corpus)
    print(f"  Corpus size: {len(corpus)} | distribution: {corpus.summary()}")

    print(f"Loading instances for {args.datasets} ...")
    all_inst = collect_instances(tuple(args.datasets), Path(args.instances_dir))
    print(f"  Total: {len(all_inst)} instances")
    train_full, eval_full = stratified_split(all_inst, eval_frac=0.2, seed=13)
    print(f"  Train: {len(train_full)} | Eval: {len(eval_full)}")

    train_inst = subsample(train_full, args.train_cap, seed=11)
    eval_inst = subsample(eval_full, args.eval_cap, seed=12)
    print(f"  Train (after cap): {len(train_inst)} | Eval (after cap): {len(eval_inst)}")

    pair_cache = out / "training_pairs.jsonl"
    pairs = build_training_pairs(
        train_inst, corpus, pair_cache,
        base_dense_model=args.base_model,
    )
    print(f"  Training pairs: {len(pairs)}")

    if not args.skip_training:
        cfg = TrainConfig(
            base_model=args.base_model,
            batch_size=args.batch_size,
            n_hard_negs=args.n_hard_negs,
            max_steps=args.max_steps,
            lr=args.lr,
            device=args.device,
            eval_subset_size=min(len(eval_inst), 200),
        )
        # Use the same eval subset throughout for an apples-to-apples curve.
        eval_subset = eval_inst[: cfg.eval_subset_size]
        train(pairs, corpus, eval_subset, config=cfg, output_dir=out / "encoder")
        plot_convergence(
            out / "encoder" / "train_log.jsonl",
            out / "encoder" / "eval_log.jsonl",
            out / "convergence.png",
        )

    if not args.skip_eval:
        ckpt = out / "encoder" / "encoder"
        results = run_all(
            corpus, eval_inst,
            sr_emb_ckpt=str(ckpt) if ckpt.exists() else None,
            output_path=out / "retrieval_eval.json",
            top_k=50,
        )
        print("\n=== Summary ===")
        for name, payload in results.items():
            print(f"  {name}: {payload['overall']}")


if __name__ == "__main__":
    main()
