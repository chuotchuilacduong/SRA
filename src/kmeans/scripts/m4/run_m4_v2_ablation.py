"""Stage: run the M4-v2 ablation (variants A0-A8 × datasets), evaluate, log.

Builds query embeddings + cluster stats + fused pools as needed, scores every
variant, writes per-variant JSONL outputs and per-variant metrics, and the run
summary. Use --variants to run a subset (e.g. quick A1 validation).

Usage:
    python src/kmeans/scripts/run_m4_v2_ablation.py \
        [--config CFG] [--datasets ...] [--variants A1 A8 ...] \
        [--force-embeddings] [--force-stats]
"""

import argparse

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from kmeans.config import M4V2Config       # noqa: E402
from kmeans import ablation_runner         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--force-embeddings", action="store_true")
    ap.add_argument("--force-stats", action="store_true")
    args = ap.parse_args()

    cfg = M4V2Config.load(args.config)
    out = ablation_runner.run(
        cfg, datasets=args.datasets, variant_ids=args.variants,
        force_embeddings=args.force_embeddings, force_stats=args.force_stats)
    print(f"ablation done: {len(out['eval'])} variants, "
          f"{out['summary']['total_wall_sec']}s")


if __name__ == "__main__":
    main()
