"""Stage: (re-)evaluate M4-v2 from the per-variant JSONL outputs.

Recomputes metrics from `results/m4_v2/{variant}/{dataset}.jsonl` and rebuilds
`results/m4_v2/ablation/{variant}.json` + `_all_eval.json` without re-scoring.
Handy if scoring is already done and only the evaluation/report needs refreshing.

Usage:
    python src/kmeans/scripts/evaluate_m4_v2.py [--config CFG] [--datasets ...]
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from kmeans.config import M4V2Config    # noqa: E402
from kmeans import io, evaluate         # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    cfg = M4V2Config.load(args.config)
    datasets = args.datasets or cfg.datasets
    all_eval = {}
    for v in cfg.variants:
        recs_by_ds = {}
        for ds in datasets:
            recs = _read_jsonl(cfg.paths["out_dir"] / v.slug / f"{ds}.jsonl")
            if recs:
                recs_by_ds[ds] = recs
        if not recs_by_ds:
            continue
        ev = evaluate.eval_variant(cfg, recs_by_ds)
        all_eval[v.id] = {"id": v.id, "name": v.name, "pool": v.pool,
                          "prf": v.prf, "affinity": v.affinity, "alpha": v.alpha,
                          "eval": ev}
        io.write_json(cfg.paths["ablation_dir"] / f"{v.id}.json", all_eval[v.id])
        print(f"{v.id}: macro R@10={ev['macro'].get('Recall@10', 0)*100:.2f} "
              f"nDCG@10={ev['macro'].get('nDCG@10', 0)*100:.2f}")
    io.write_json(cfg.paths["ablation_dir"] / "_all_eval.json",
                  {"datasets": datasets, "variants": all_eval})
    print(f"evaluated {len(all_eval)} variants -> {cfg.paths['ablation_dir']}/_all_eval.json")


if __name__ == "__main__":
    main()
