"""Stage: write all M4-v2 reports/tables from the evaluated ablation.

Reads `results/m4_v2/ablation/_all_eval.json` (produced by run_ablation or
evaluate_m4_v2) and writes the comparison/ablation/delta/error-analysis tables
and `FULL_M4_V2_RESULTS.md`.

Usage:
    python src/kmeans/scripts/write_m4_v2_report.py [--config CFG]
"""

import argparse

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from kmeans.config import M4V2Config    # noqa: E402
from kmeans import report_writer        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = M4V2Config.load(args.config)
    report_writer.write_all(cfg)
    print("reports written to results/comparisons/, results/analysis/, FULL_M4_V2_RESULTS.md")


if __name__ == "__main__":
    main()
