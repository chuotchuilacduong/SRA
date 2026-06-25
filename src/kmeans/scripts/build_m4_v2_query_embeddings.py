"""Stage: encode + cache per-dataset BGE query embeddings.

The repo never cached query vectors; M4-v2 needs them (soft affinity, PRF,
adaptive-α). Encodes with the same model + prefix the corpus pipeline used.

Usage:
    python src/kmeans/scripts/build_m4_v2_query_embeddings.py \
        [--config CFG] [--datasets ds1 ds2 ...] [--force]
"""

import argparse

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from kmeans.config import M4V2Config    # noqa: E402
from kmeans import embeddings           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = M4V2Config.load(args.config)
    datasets = args.datasets or cfg.datasets
    total = 0
    for ds in datasets:
        total += embeddings.build_dataset_query_embeddings(cfg, ds, force=args.force)
    print(f"query embeddings ready for {datasets} ({total} queries)")


if __name__ == "__main__":
    main()
