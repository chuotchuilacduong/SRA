"""``sragents make-splits`` — write train/dev/test instance_id lists."""

from __future__ import annotations

import json
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.data_config import load_data_config
from sragents.train.split_builder import build_split


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "make-splits", help="Build query/skill/leave-domain-out splits",
        description="Generates ID lists for the three generalization protocols. "
                    "Output is a JSON file with keys train/dev/test.",
    )
    p.add_argument("--protocol", choices=["query_gen", "skill_gen", "ldo"], required=True)
    p.add_argument("--instances", type=Path, nargs="+", required=True,
                   help="One or more instances JSON files (concatenated)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--held-out-dataset", default=None,
                   help="Required for protocol=ldo")
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.set_defaults(func=run)


def run(args) -> None:
    data_cfg = load_data_config(args.data_config)
    all_instances: list[dict] = []
    for path in args.instances:
        require_exists(path, "instances")
        all_instances.extend(json.loads(path.read_text()))

    kwargs = {
        "instance_id_field": data_cfg.f_query_id,
        "seed": args.seed,
    }
    if args.protocol == "query_gen":
        kwargs["dev_ratio"] = args.dev_ratio
        kwargs["test_ratio"] = args.test_ratio
    elif args.protocol == "skill_gen":
        kwargs["gold_field"] = data_cfg.f_gold
        kwargs["dev_skill_ratio"] = args.dev_ratio
        kwargs["test_skill_ratio"] = args.test_ratio
    else:  # ldo
        if not args.held_out_dataset:
            import sys
            sys.exit("--held-out-dataset is required for protocol=ldo")
        kwargs["dataset_field"] = data_cfg.f_domain
        kwargs["held_out_dataset"] = args.held_out_dataset
        kwargs["dev_ratio"] = args.dev_ratio

    splits = build_split(args.protocol, all_instances, **kwargs)
    splits.dump(args.out)
    print(f"Wrote {args.out}  train={len(splits.train)} dev={len(splits.dev)} test={len(splits.test)}")
