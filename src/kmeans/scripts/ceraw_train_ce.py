"""CE-Raw Phase 2 — fine-tune the cross-encoder on the raw (no stage-1) pairs.

Thin driver around the repo's proven trainer (``sragents.train.train_cross_encoder`` — the
same code that produced ce-joint-v3), so the only thing that changes vs M5/M7 is the
*training data* (full-corpus-mined negatives instead of the M4 @100 pool). Listwise loss,
full-text packing, MiniLM-L6 base — identical recipe to experiments/configs/train_loss_listwise.yaml.

Device: CUDA (H100) > CPU automatically (MPS disabled by the trainer).

    python src/kmeans/scripts/ceraw_train_ce.py \
        --config src/kmeans/configs/ceraw_listwise.yaml \
        --out results/models/ce-raw-v1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.train.train_cross_encoder import TrainConfig, train_cross_encoder

DATA = PROJECT_ROOT / "data" / "ce_raw"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train CE-Raw (listwise, full-corpus negatives)")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "src/kmeans/configs/ceraw_listwise.yaml")
    ap.add_argument("--train-pairs", type=Path, default=DATA / "ce_pairs_train.json")
    ap.add_argument("--dev-pool", type=Path, default=DATA / "dev_pool.json")
    ap.add_argument("--instances", type=Path, default=DATA / "instances_all.json")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/models/ce-raw-v1")
    args = ap.parse_args()

    for p in (args.config, args.train_pairs, args.dev_pool, args.instances):
        if not p.exists():
            raise SystemExit(f"missing required input: {p} (run ceraw_build_dataset.py first)")

    corpus = load_corpus_dict()
    train_pairs = json.loads(args.train_pairs.read_text())
    dev_records = json.loads(args.dev_pool.read_text())["results"]
    instances = json.loads(args.instances.read_text())
    instances_by_id = {i["instance_id"]: i for i in instances}

    config = TrainConfig.from_yaml(args.config)
    n_pos = sum(1 for p in train_pairs if float(p["label"]) > 0.5)
    print(f"[ce-raw] pairs={len(train_pairs)} pos={n_pos} neg={len(train_pairs)-n_pos} "
          f"| dev_queries={len(dev_records)} | loss={config.loss_name} "
          f"base={config.base_model} -> {args.out}", flush=True)

    summary = train_cross_encoder(
        train_pairs=train_pairs,
        dev_pool_records=dev_records,
        corpus=corpus,
        instances_by_id=instances_by_id,
        config=config,
        output_dir=args.out,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
