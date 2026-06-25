"""Run Cross-Encoder Method 7 at pool depth 500 (CE@500) and integrate into the
5-fold CV tables (§15-§19). No CE retraining — only inference on the deeper pool.

Modes:
  --mode gate  : reproduce Method7@100 from hybrid_km_alpha70 top-100 and compare
                 to the published fused_alpha_beta70 metrics (faithfulness gate).
  --mode full  : CE-rerank the RRF top-500 (Stage1=M4), evaluate, add
                 "Method7-CE@500" to results/qsc_ltr/cv/cv_consolidated.json
                 (per-fold + full + significance L6@500 vs CE@500), regenerate reports.

Usage: python src/kmeans/scripts/run_ce500.py --mode {gate,full} [--workers N] [--datasets ...]
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

from sragents.config import PROJECT_ROOT                     # noqa: E402
from kmeans.qsc_ltr_runner import load_ext                   # noqa: E402
from kmeans import ce500, evaluate, cv, cv_report            # noqa: E402

CE500_DIR = PROJECT_ROOT / "results" / "qsc_ltr" / "ce500"
CONS = PROJECT_ROOT / "results" / "qsc_ltr" / "cv" / "cv_consolidated.json"
METRICS = cv_report.METRICS


def gate(cfg, ext, datasets, workers):
    q = ce500.queries_from_hybrid_km(cfg, datasets, top_k=100)
    print(f"gate: {len(q)} queries over {datasets}", flush=True)
    recs = ce500.run_parallel(q, workers=workers, beta=ce500.BETA, top_k=100)
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r)
    ok = True
    for ds in datasets:
        m = evaluate.eval_records(cfg, by_ds[ds])
        pub = json.loads((PROJECT_ROOT / f"results/rerank/fused_alpha_beta70-{ds}.json").read_text())["metrics"]
        print(f"\n[{ds}] repro vs published Method7:")
        for k in ["Recall@1", "Recall@10", "Recall@50", "Recall@100", "nDCG@10"]:
            d = abs(m.get(k, 0) - pub.get(k, 0)) * 100
            flag = "" if d < 0.5 else "  <-- MISMATCH"
            if d >= 0.5:
                ok = False
            print(f"  {k:12} repro={m.get(k,0)*100:6.2f}  pub={pub.get(k,0)*100:6.2f}  Δ={d:.3f}{flag}")
    print("\nGATE:", "PASS" if ok else "FAIL (reproduction differs >0.5pp)")
    return ok


def full(cfg, ext, datasets, workers):
    CE500_DIR.mkdir(parents=True, exist_ok=True)
    q = ce500.queries_from_feature_tables(cfg, ext, datasets, pool_size=500)
    print(f"CE@500: reranking {len(q)} queries × up to 500 candidates ...", flush=True)
    recs = ce500.run_parallel(q, workers=workers, beta=ce500.BETA, top_k=cfg.output_top_k)
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, rs in by_ds.items():
        (CE500_DIR / f"{ds}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
    integrate(cfg, ext, datasets, by_ds)


def integrate(cfg, ext, datasets, by_ds):
    """Add Method7-CE@500 to the CV consolidated and regenerate §15-§19."""
    c = json.loads(CONS.read_text())
    tables = {ds: __import__("kmeans.ltr_features", fromlist=["load_features"]).load_features(
        PROJECT_ROOT / ext["reporting"]["cache_dir"] / f"{ds}.npz") for ds in datasets}
    fold_of = cv.make_folds(tables)
    fold_test_ids = {i: cv.fold_splits(fold_of, i)[2] for i in range(cv.N_FOLDS)}

    per_fold = [{m: round(cv.eval_on_ids(cfg, by_ds, fold_test_ids[i])["macro"].get(m, 0) * 100, 2)
                 for m in METRICS} for i in range(cv.N_FOLDS)]
    full_eval = cv.eval_on_ids(cfg, by_ds, None)
    c["baselines"]["Method7-CE@500"] = {
        "per_fold": per_fold,
        "fold_stats": {m: cv.fold_mean_std(per_fold, m) for m in METRICS},
        "full_macro": {m: round(full_eval["macro"].get(m, 0) * 100, 2) for m in METRICS},
        "full_by_dataset": {ds: {m: round(mm.get(m, 0) * 100, 2) for m in METRICS}
                            for ds, mm in full_eval["by_dataset"].items()}}

    # significance: L6@500 (OOF) vs CE@500, and CE@500 vs CE@100 for context
    l6_oof = cv.assemble_oof(PROJECT_ROOT / "results/qsc_ltr/cv", "L6")
    ce500_recs = [r for rs in by_ds.values() for r in rs]
    c["significance"]["L6full_vs_Method7@500"] = {
        "Recall@10": cv.paired_bootstrap(cv.per_query_metric(l6_oof, "recall", 10),
                                         cv.per_query_metric(ce500_recs, "recall", 10)),
        "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(l6_oof, "ndcg", 10),
                                       cv.per_query_metric(ce500_recs, "ndcg", 10))}
    CONS.write_text(json.dumps(c, indent=2))
    print(f"added Method7-CE@500 to {CONS}")
    cv_report.write_all(CONS)
    print("regenerated §15-§19 with Method7-CE@500")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "full"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    ext, cfg = load_ext()
    datasets = args.datasets or ext["datasets"]
    if args.mode == "gate":
        gate(cfg, ext, datasets or ["champ"], args.workers)
    else:
        full(cfg, ext, datasets, args.workers)


if __name__ == "__main__":
    main()
