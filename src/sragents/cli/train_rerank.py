"""``sragents train-rerank`` — fine-tune a cross-encoder reranker."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.data_config import load_data_config
from sragents.train.train_cross_encoder import TrainConfig, train_cross_encoder

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train-rerank", help="Fine-tune a cross-encoder reranker",
        description="Train ms-marco-MiniLM (or any HF cross-encoder) on the "
                    "train_pairs.json produced by `sragents mine-negatives`.",
    )
    p.add_argument("--config", type=Path,
                   default=PROJECT_ROOT / "configs" / "train.yaml")
    p.add_argument("--train-pairs", type=Path, required=True,
                   help="Output of `sragents mine-negatives` (array of pairs)")
    p.add_argument("--dev-pool", type=Path, default=None,
                   help="Optional pool JSON used for dev-time scoring + early stopping")
    p.add_argument("--instances", type=Path, default=None,
                   help="Concatenated instances JSON (needed iff --dev-pool given)")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True,
                   help="Directory to save the best checkpoint into")
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.set_defaults(func=run)


def run(args) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    require_exists(args.train_pairs, "train-pairs")
    data_cfg = load_data_config(args.data_config)
    corpus_dict = load_corpus_dict(args.corpus) if args.corpus else load_corpus_dict()
    train_pairs = json.loads(args.train_pairs.read_text())

    dev_records, instances_by_id = None, {}
    if args.dev_pool is not None:
        require_exists(args.dev_pool, "dev-pool")
        require_exists(args.instances, "instances")
        dev_records = json.loads(args.dev_pool.read_text())["results"]
        instances = json.loads(args.instances.read_text())
        instances_by_id = {data_cfg.instance_query_id(i): i for i in instances}

    config = TrainConfig.from_yaml(args.config)
    summary = train_cross_encoder(
        train_pairs=train_pairs,
        dev_pool_records=dev_records,
        corpus=corpus_dict,
        instances_by_id=instances_by_id,
        config=config,
        output_dir=args.out,
    )
    print(json.dumps(summary, indent=2))
