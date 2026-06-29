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

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).parent.parent))
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


def _baseline_summary(cfg, by_ds, fold_test_ids):
    per_fold = [{m: round(cv.eval_on_ids(cfg, by_ds, fold_test_ids[i])["macro"].get(m, 0) * 100, 2)
                 for m in METRICS} for i in range(cv.N_FOLDS)]
    full = cv.eval_on_ids(cfg, by_ds, None)
    return {"per_fold": per_fold, "fold_stats": {m: cv.fold_mean_std(per_fold, m) for m in METRICS},
            "full_macro": {m: round(full["macro"].get(m, 0) * 100, 2) for m in METRICS},
            "full_by_dataset": {ds: {m: round(mm.get(m, 0) * 100, 2) for m in METRICS}
                                for ds, mm in full["by_dataset"].items()}}


def m5(cfg, ext, datasets, workers):
    """M5 (CE-HYRR, raw CE, NO fusion) @100 and @500, added to the §15-§19 tables."""
    from kmeans import io as kio, ltr_features
    gold_by_id = {r["instance_id"]: r["gold_skill_ids"]
                  for ds in datasets for r in kio.load_instances(cfg, ds)}
    # M5@100 official = full_bench_h100 (raw CE over RRF top-100)
    m5_100 = cv.load_retrieval_baseline("results/rerank/full_bench_h100-{ds}.json", datasets, gold_by_id)

    # M5@500 = pure CE over RRF top-500. Reuse cached records if present (skip the CE pass).
    m5dir = PROJECT_ROOT / "results/qsc_ltr/m5_500"
    if all((m5dir / f"{ds}.jsonl").exists() for ds in datasets):
        print("M5: reuse cached m5_500 records (skip CE pass)")
        by500 = {ds: cv._read_jsonl(m5dir / f"{ds}.jsonl") for ds in datasets}
    else:
        q = ce500.queries_from_feature_tables(cfg, ext, datasets, pool_size=500)
        print(f"M5: pure-CE rerank {len(q)} queries × up to 500 candidates ...", flush=True)
        recs = ce500.run_parallel_ce(q, workers)
        by500, by100d = {}, {}
        m5dir.mkdir(parents=True, exist_ok=True)
        for r in recs:
            ds = r["dataset"]
            top500 = [{"skill_id": c["skill_id"]} for c in r["ranked"][:cfg.output_top_k]]
            top100 = [{"skill_id": c["skill_id"]} for c in r["ranked"] if c["rrf_rank"] <= 100][:cfg.output_top_k]
            by500.setdefault(ds, []).append({"instance_id": r["instance_id"], "dataset": ds,
                                             "gold_skill_ids": r["gold_skill_ids"], "retrieved": top500})
            by100d.setdefault(ds, []).append({"instance_id": r["instance_id"], "dataset": ds,
                                              "gold_skill_ids": r["gold_skill_ids"], "retrieved": top100})
        for ds, rs in by500.items():
            (m5dir / f"{ds}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
        # GATE: derived M5@100 ≈ official full_bench_h100 (both raw CE over RRF top-100)
        ok = True
        print("\n=== GATE: derived M5@100 vs published full_bench_h100 ===")
        for ds in datasets:
            a = evaluate.eval_records(cfg, by100d[ds]); b = evaluate.eval_records(cfg, m5_100[ds])
            for k in ["Recall@1", "Recall@10", "nDCG@10"]:
                if abs(a.get(k, 0) - b.get(k, 0)) * 100 >= 0.5:
                    ok = False
            print(f"  {ds:14} R@1 Δ={abs(a.get('Recall@1',0)-b.get('Recall@1',0))*100:.3f} "
                  f"R@10 Δ={abs(a.get('Recall@10',0)-b.get('Recall@10',0))*100:.3f} "
                  f"nDCG@10 Δ={abs(a.get('nDCG@10',0)-b.get('nDCG@10',0))*100:.3f}")
        print("GATE:", "PASS" if ok else "WARN(>0.5pp)")

    # integrate into consolidated
    c = json.loads(CONS.read_text())
    tables = {ds: ltr_features.load_features(PROJECT_ROOT / ext["reporting"]["cache_dir"] / f"{ds}.npz")
              for ds in datasets}
    fold_of = cv.make_folds(tables)
    fti = {i: cv.fold_splits(fold_of, i)[2] for i in range(cv.N_FOLDS)}
    c["baselines"]["M5-CE@100"] = _baseline_summary(cfg, m5_100, fti)
    c["baselines"]["M5-CE@500"] = _baseline_summary(cfg, by500, fti)
    # significance vs L6 and vs M7
    l6 = cv.assemble_oof(PROJECT_ROOT / "results/qsc_ltr/cv", "L6")
    m5_500_recs = [r for rs in by500.values() for r in rs]
    # M7@500 records are JSONL (one record per line) — read with the jsonl reader
    m7_dir = PROJECT_ROOT / "results/qsc_ltr/ce500"
    m7r = ([r for ds in datasets for r in cv._read_jsonl(m7_dir / f"{ds}.jsonl")]
           if (m7_dir / "champ.jsonl").exists() else [])
    c["significance"]["L6full_vs_M5@500"] = {
        "Recall@10": cv.paired_bootstrap(cv.per_query_metric(l6, "recall", 10),
                                         cv.per_query_metric(m5_500_recs, "recall", 10)),
        "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(l6, "ndcg", 10),
                                       cv.per_query_metric(m5_500_recs, "ndcg", 10))}
    if m7r:
        c["significance"]["M7@500_vs_M5@500"] = {
            "Recall@10": cv.paired_bootstrap(cv.per_query_metric(m7r, "recall", 10),
                                             cv.per_query_metric(m5_500_recs, "recall", 10)),
            "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(m7r, "ndcg", 10),
                                           cv.per_query_metric(m5_500_recs, "ndcg", 10))}
    CONS.write_text(json.dumps(c, indent=2))
    print(f"added M5-CE@100 + M5-CE@500 to {CONS}")
    cv_report.write_all(CONS)
    print("regenerated §15-§19 with M5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "full", "m5"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    ext, cfg = load_ext()
    datasets = args.datasets or ext["datasets"]
    if args.mode == "gate":
        gate(cfg, ext, datasets or ["champ"], args.workers)
    elif args.mode == "m5":
        m5(cfg, ext, datasets, args.workers)
    else:
        full(cfg, ext, datasets, args.workers)


if __name__ == "__main__":
    main()
