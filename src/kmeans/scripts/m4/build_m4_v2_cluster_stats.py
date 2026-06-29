"""Stage: compute cluster reliability statistics (spec Step 4). Runs once, offline.

Usage:
    python src/kmeans/scripts/build_m4_v2_cluster_stats.py [--config CFG] [--force]
"""

import argparse

import _bootstrap  # noqa: F401  (sys.path + logging)

_bootstrap.setup_logging()

from kmeans.config import M4V2Config            # noqa: E402
from kmeans import io, cluster_stats            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = M4V2Config.load(args.config)
    art = io.load_artifacts(cfg)
    rel = cluster_stats.build_and_save(cfg, art, force=args.force)
    print(f"cluster stats -> {cfg.paths['cluster_stats']} "
          f"(K={art.K}, rel mean={float(rel.mean()):.3f})")


if __name__ == "__main__":
    main()
