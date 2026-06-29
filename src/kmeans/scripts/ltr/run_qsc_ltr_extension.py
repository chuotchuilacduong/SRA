"""Run the full QSC + LTR extension end-to-end and write all reports.

Stages (plan §15): QSC scoring (Q0-Q6) + LTR feature build -> LTR train/predict
(L0-L7) on a query-level split -> evaluate -> write metrics/delta/error-analysis
and append Section 14 to FULL_M4_V2_RESULTS.md.

Usage:
    python src/kmeans/scripts/run_qsc_ltr_extension.py [--config CFG] [--datasets ...]
                                                       [--report-only]
"""

import argparse

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from kmeans import qsc_ltr_runner, qsc_ltr_report   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="skip run; regenerate reports from consolidated_eval.json")
    args = ap.parse_args()

    if not args.report_only:
        cons = qsc_ltr_runner.run(args.config, datasets=args.datasets)
        print(f"QSC variants: {len(cons['qsc'])}, LTR variants: {len(cons['ltr'])}, "
              f"split={cons['split_sizes']}")
    qsc_ltr_report.write_all(args.config)
    print("reports: results/comparisons/qsc_ltr_extension_*, "
          "results/analysis/qsc_ltr_error_analysis.md, FULL_M4_V2_RESULTS.md (§14)")


if __name__ == "__main__":
    main()
