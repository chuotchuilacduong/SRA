"""``sragents probe`` — annotate a pool with failure-probe labels.

Writes ``{instance_id: [probe_labels]}`` JSON useful for slicing
post-hoc per-bucket metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.data_config import load_data_config
from sragents.eval.failure_probes import ProbeConfig, annotate_pool


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "probe", help="Annotate a pool with failure-probe labels",
        description="Heuristic per-query labels: formula_heavy, code_heavy, "
                    "multi_label, long_skill, lexical/semantic_confounder.",
    )
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--eval-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "eval.yaml")
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.pool, "pool")
    require_exists(args.instances, "instances")
    data_cfg = load_data_config(args.data_config)
    cfg = ProbeConfig.from_yaml(args.eval_config)
    corpus = load_corpus_dict(args.corpus) if args.corpus else load_corpus_dict()
    pool = json.loads(args.pool.read_text())["results"]
    instances = json.loads(args.instances.read_text())
    by_id = {data_cfg.instance_query_id(i): i for i in instances}

    annotations = annotate_pool(
        pool, by_id, corpus, cfg,
        question_field=data_cfg.f_query,
        gold_field=data_cfg.f_gold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(annotations, indent=2))
    # Coarse summary
    counts: dict[str, int] = {}
    for labels in annotations.values():
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
    print(f"Annotated {len(annotations)} queries; wrote {args.output}")
    for lab, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lab}: {c}")
