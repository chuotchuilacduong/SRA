"""5-fold CV + full LightGBM grid + fair baselines + significance (reviewer-grade).

Runs L0-L7 (L6 full grid) over 5 query-level folds in parallel single-threaded
processes, assembles out-of-fold predictions, evaluates our baselines (A0/A1/A7/
Q2/Q6), the original-paper retrieval baselines (BM25/BGE/RRF/Hybrid/LinearRAG),
and the Cross-Encoder Method 7 on the SAME fold/test queries, runs paired-bootstrap
significance, audits leakage/depth-fairness, and writes the §15-§19 report.

Usage: python src/kmeans/scripts/run_ltr_cv.py [--workers N] [--report-only]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

import numpy as np                                   # noqa: E402
from sragents.config import PROJECT_ROOT             # noqa: E402
from kmeans.qsc_ltr_runner import load_ext           # noqa: E402
from kmeans import cv, ltr_features, cv_report        # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "qsc_ltr" / "cv"
CONS = OUT_DIR / "cv_consolidated.json"

# Original-paper retrieval baselines available as cached per-query outputs.
RETRIEVAL_BASELINES = {
    "BM25": "results/retrieval_bm25/{ds}-bm25.json",
    "BGE": "results/retrieval_dense/{ds}-dense-bge.json",
    "RRF(BM25+BGE)": "results/retrieval_rrf_bm25_dense/{ds}-rrf-bm25-dense.json",
    "Hybrid-official": "results/retrieval_hybrid_official/{ds}-hybrid-official.json",
    "LinearRAG": "results/retrieval_all/{ds}-linearrag.json",
}
METHOD7_TMPL = "results/rerank/fused_alpha_beta70-{ds}.json"
METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100",
           "nDCG@1", "nDCG@5", "nDCG@10"]


def leakage_audit(tables, fold_of):
    audit = {"folds": [], "feature_label_independence": None, "notes": []}
    for i in range(cv.N_FOLDS):
        tr, dev, te = cv.fold_splits(fold_of, i)
        audit["folds"].append({
            "fold": i, "train": len(tr), "dev": len(dev), "test": len(te),
            "train_dev_overlap": len(tr & dev), "train_test_overlap": len(tr & te),
            "dev_test_overlap": len(dev & te)})
    # no feature column equals the label (sample a dataset)
    t = tables["toolqa"]
    eq = [ltr_features.ALL_FEATURES[j] for j in range(t["X"].shape[1])
          if np.array_equal(t["X"][:, j].astype(int), t["y"])]
    audit["feature_label_independence"] = {"cols_equal_to_label": eq, "n_features": len(ltr_features.ALL_FEATURES)}
    audit["notes"] = [
        "Every query is test exactly once (OOF coverage); candidates of a query never cross splits.",
        "LTR ranks the RRF top-500 pool; CE Method 7 and the depth-100 baselines rank top-100.",
        "L6@100 (rerank only RRF top-100) is reported for a depth-matched CE comparison.",
        "Retrieval baselines (BM25/BGE/RRF/Hybrid/LinearRAG) retrieve from the full corpus at their own depth (BM25/Hybrid/LinearRAG top-50 -> R@100 capped).",
    ]
    return audit


def run_pool(ext, workers):
    units = [{"variant": v, "fold_i": i, "ext_path": None,
              "cache_dir": str(PROJECT_ROOT / ext["reporting"]["cache_dir"]),
              "out_dir": str(OUT_DIR), "n_folds": cv.N_FOLDS, "seed": cv.FOLD_SEED}
             for v in ext["ltr"]["variants"] for i in range(cv.N_FOLDS)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(cv.run_unit, u): u for u in units}
        for fut in as_completed(futs):
            u = futs[fut]
            res = fut.result()
            results.append(res)
            print(f"  done {res['variant']} fold{res['fold']}: "
                  f"R@10={res['macro'].get('Recall@10'):.2f} nDCG@10={res['macro'].get('nDCG@10'):.2f}",
                  flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--reuse-ltr", action="store_true",
                    help="reuse cached LTR fold predictions; recompute baselines/sig only")
    args = ap.parse_args()

    ext, cfg = load_ext()
    datasets = ext["datasets"]
    tables = {ds: ltr_features.load_features(PROJECT_ROOT / ext["reporting"]["cache_dir"] / f"{ds}.npz")
              for ds in datasets}
    fold_of = cv.make_folds(tables)
    fold_test_ids = {i: cv.fold_splits(fold_of, i)[2] for i in range(cv.N_FOLDS)}

    if not args.report_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        audit = leakage_audit(tables, fold_of)

        if args.reuse_ltr and CONS.exists():
            old = json.loads(CONS.read_text())
            ltr_methods = old["ltr"]
            ltr_methods["L6"]["_oof_records"] = cv.assemble_oof(OUT_DIR, "L6")
            ltr_methods["L6"]["_oof100_records"] = cv.assemble_oof(OUT_DIR, "L6_d100")
            print("reusing cached LTR fold predictions; recomputing baselines/significance")
            unit_results = []
        else:
            unit_results = run_pool(ext, args.workers)

        # organize LTR results: variant -> {per_fold:[macro per fold], by_dataset_oof, meta_per_fold}
        ltr_methods = {} if not (args.reuse_ltr and CONS.exists()) else ltr_methods
        for v in ([] if (args.reuse_ltr and CONS.exists()) else ext["ltr"]["variants"]):
            vid = v["id"]
            folds = sorted([r for r in unit_results if r["variant"] == vid], key=lambda r: r["fold"])
            per_fold = [r["macro"] for r in folds]
            oof = cv.assemble_oof(OUT_DIR, vid)
            oof_eval = cv.eval_on_ids(cfg, cv._group(oof), None)
            entry = {"per_fold": per_fold,
                     "fold_stats": {m: cv.fold_mean_std(per_fold, m) for m in METRICS},
                     "oof_macro": {m: round(oof_eval["macro"].get(m, 0) * 100, 2) for m in METRICS},
                     "oof_by_dataset": {ds: {m: round(mm.get(m, 0) * 100, 2) for m in METRICS}
                                        for ds, mm in oof_eval["by_dataset"].items()},
                     "best_params_per_fold": [r["meta"].get("params") for r in folds],
                     "feature_importance_per_fold": [r["meta"].get("feature_importances") for r in folds]}
            if "d100_macro" in folds[0]:
                per_fold_d100 = [r.get("d100_macro", {}) for r in folds]
                entry["d100_fold_stats"] = {m: cv.fold_mean_std(per_fold_d100, m) for m in METRICS}
                oof100 = cv.assemble_oof(OUT_DIR, vid + "_d100")
                entry["d100_oof_macro"] = {m: round(cv.eval_on_ids(cfg, cv._group(oof100), None)["macro"].get(m, 0) * 100, 2) for m in METRICS}
                entry["_oof100_records"] = oof100
            entry["_oof_records"] = oof
            ltr_methods[vid] = entry

        # baselines: our (A0/A1/A7/Q2/Q6) + retrieval + Method 7
        from kmeans import io as kio
        gold_by_id = {r["instance_id"]: r["gold_skill_ids"]
                      for ds in datasets for r in kio.load_instances(cfg, ds)}
        baselines = {}
        qsc_dir = PROJECT_ROOT / ext["reporting"]["out_dir"]
        for vid in ["A0", "A1", "A7", "Q2", "Q6"]:
            baselines[vid] = cv.load_our_baseline(cfg, vid, datasets, qsc_dir)
        for name, tmpl in RETRIEVAL_BASELINES.items():
            baselines[name] = cv.load_retrieval_baseline(tmpl, datasets, gold_by_id)
        baselines["Method7-CE"] = cv.load_retrieval_baseline(METHOD7_TMPL, datasets, gold_by_id)

        base_summary = {}
        for name, by_ds in baselines.items():
            if not by_ds:
                continue
            per_fold = [{m: round(cv.eval_on_ids(cfg, by_ds, fold_test_ids[i])["macro"].get(m, 0) * 100, 2)
                         for m in METRICS} for i in range(cv.N_FOLDS)]
            full = cv.eval_on_ids(cfg, by_ds, None)
            base_summary[name] = {
                "per_fold": per_fold,
                "fold_stats": {m: cv.fold_mean_std(per_fold, m) for m in METRICS},
                "full_macro": {m: round(full["macro"].get(m, 0) * 100, 2) for m in METRICS},
                "full_by_dataset": {ds: {m: round(mm.get(m, 0) * 100, 2) for m in METRICS}
                                    for ds, mm in full["by_dataset"].items()},
                "_records": [r for recs in by_ds.values() for r in recs]}

        # significance: L6 (OOF) vs A0/A7/Q6; L6@100 (OOF) vs Method7
        sig = {}
        l6_oof = ltr_methods["L6"]["_oof_records"]
        for base_id in ["A0", "A7", "Q6"]:
            brecs = base_summary[base_id]["_records"]
            sig[f"L6_vs_{base_id}"] = {
                "Recall@10": cv.paired_bootstrap(cv.per_query_metric(l6_oof, "recall", 10),
                                                 cv.per_query_metric(brecs, "recall", 10)),
                "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(l6_oof, "ndcg", 10),
                                               cv.per_query_metric(brecs, "ndcg", 10))}
        if "Method7-CE" in base_summary:
            m7 = base_summary["Method7-CE"]["_records"]
            l6_100 = ltr_methods["L6"].get("_oof100_records", l6_oof)
            for label, recs in [("L6d100_vs_Method7", l6_100), ("L6full_vs_Method7", l6_oof)]:
                sig[label] = {
                    "Recall@10": cv.paired_bootstrap(cv.per_query_metric(recs, "recall", 10),
                                                     cv.per_query_metric(m7, "recall", 10)),
                    "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(recs, "ndcg", 10),
                                                   cv.per_query_metric(m7, "ndcg", 10))}

        # strip heavy per-record lists before serialization
        for e in ltr_methods.values():
            e.pop("_oof_records", None); e.pop("_oof100_records", None)
        for e in base_summary.values():
            e.pop("_records", None)

        consolidated = {"datasets": datasets, "n_folds": cv.N_FOLDS, "seed": cv.FOLD_SEED,
                        "metrics": METRICS, "leakage_audit": audit,
                        "ltr": ltr_methods, "baselines": base_summary, "significance": sig,
                        "fold_test_counts": {i: len(fold_test_ids[i]) for i in range(cv.N_FOLDS)}}
        CONS.write_text(json.dumps(consolidated, indent=2))
        print(f"wrote {CONS}")

    cv_report.write_all(CONS)


if __name__ == "__main__":
    main()
