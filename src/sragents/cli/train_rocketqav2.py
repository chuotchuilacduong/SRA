"""``sragents train-rocketqav2`` — joint dual-encoder + cross-encoder training.

Implements dynamic listwise distillation (RocketQAv2, Ren et al., EMNLP 2021):

    L = KL(p̃_DE ‖ p̃_CE)  +  CrossEntropy(s_CE, positive_idx)

Input is the same ``train_pairs.json`` produced by ``sragents mine-negatives``,
so the pipeline is identical to ``train-rerank`` up to this point:

    sragents build-pool   →  pool.json
    sragents mine-negatives  →  train_pairs.json
    sragents train-rocketqav2  →  dual_encoder/  +  cross_encoder/

Trained checkpoints are standard HuggingFace checkpoints:

    dual_encoder/   — use with: sragents retrieve --retriever bge
                          --retriever-arg model_path=<dir>/dual_encoder
    cross_encoder/  — use with: sragents rerank-topk
                          --model-path <dir>/cross_encoder
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.data_config import load_data_config
from sragents.retrieve.rocketqav2_trainer import RocketQAv2Config, train_rocketqav2

log = logging.getLogger(__name__)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train-rocketqav2",
        help="Joint dual-encoder + cross-encoder training via listwise distillation (RocketQAv2)",
        description=(
            "Jointly fine-tunes a dual-encoder retriever and a cross-encoder reranker "
            "using dynamic listwise distillation (RocketQAv2, EMNLP 2021). "
            "Accepts the same train_pairs.json produced by `sragents mine-negatives`."
        ),
    )
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
        help="Training config YAML (default: configs/train.yaml). "
             "Add a [rocketqav2] section for DE-specific settings.",
    )
    p.add_argument(
        "--train-pairs", type=Path, required=True,
        help="Output of `sragents mine-negatives` (array of train pairs)",
    )
    p.add_argument(
        "--dev-pool", type=Path, default=None,
        help="Optional pool JSON used for dev scoring + early stopping (CE-based)",
    )
    p.add_argument(
        "--instances", type=Path, default=None,
        help="Concatenated instances JSON (required if --dev-pool is given)",
    )
    p.add_argument(
        "--corpus", type=Path, default=None,
        help="Corpus JSON override (default: package default corpus)",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output directory: saves dual_encoder/ and cross_encoder/ subdirs",
    )
    p.add_argument(
        "--data-config", type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
    )
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

    config = RocketQAv2Config.from_yaml(args.config) if args.config.exists() else RocketQAv2Config()

    log.info(
        "RocketQAv2 — de=%s  ce=%s  epochs=%d  batch=%d  lr=%s",
        config.de_model, config.ce_model,
        config.epochs, config.batch_size, config.lr,
    )

    summary = train_rocketqav2(
        train_pairs=train_pairs,
        dev_pool_records=dev_records,
        corpus=corpus_dict,
        instances_by_id=instances_by_id,
        config=config,
        output_dir=args.out,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nDual-encoder  → {args.out}/dual_encoder")
    print(f"Cross-encoder → {args.out}/cross_encoder")
