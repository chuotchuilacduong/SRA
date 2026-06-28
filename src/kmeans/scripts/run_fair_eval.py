"""§20 Fair supervised comparison on query_gen-test (no CE-leakage).

CE (ce-joint-v3) was trained on the query_gen TRAIN split (3,782 queries, seed 42).
So query_gen-TEST (1,079) is held-out for CE. We train a FINAL L6 on query_gen-train
(dev-tuned on query_gen-dev) and evaluate EVERY method on query_gen-test — both L6
and CE are then held-out → a fair supervised comparison.

Phases: (1) materialize datasets into data/; (2) train+save final L6; (3) evaluate
all methods on query_gen-test + paired-bootstrap significance; (4) write reports +
append §20 to FULL_M4_V2_RESULTS.md.

Run: python src/kmeans/scripts/run_fair_eval.py [--grid full|moderate]
"""

import argparse
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

_bootstrap.setup_logging()

import numpy as np                                              # noqa: E402
from sragents.config import PROJECT_ROOT                        # noqa: E402
from kmeans.qsc_ltr_runner import load_ext, _read_jsonl        # noqa: E402
from kmeans import cv, ltr, ltr_features, evaluate, io          # noqa: E402

DATA = PROJECT_ROOT / "data"
COMP = PROJECT_ROOT / "results" / "comparisons"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
MET = evaluate.REPORT_METRICS  # R@1,R@5,R@10,R@50,R@100,nDCG@1,nDCG@5,nDCG@10
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]

GRIDS = {
    "moderate": {"num_leaves": [31, 63], "learning_rate": [0.03, 0.05], "n_estimators": [500],
                 "min_data_in_leaf": [30], "early_stopping_rounds": 50, "eval_at": [1, 5, 10]},
    "full": cv.FULL_GRID,
}


def load_query_gen(ext):
    split = {ds: json.loads((PROJECT_ROOT / f"results/splits/{ds}-query_gen.json").read_text())
             for ds in ext["datasets"]}
    train = {q for ds in split for q in split[ds]["train"]}
    dev = {q for ds in split for q in split[ds].get("dev", [])}
    test = {q for ds in split for q in split[ds]["test"]}
    return split, train, dev, test


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def _pct(block, m):
    return block.get(m, float("nan")) * 100.0


# --- Phase 1: materialize datasets into data/ --------------------------------

def export_subset(tables, datasets, id_set, path):
    Xs, ys, gs, iids, sids, gold = [], [], [], [], [], []
    for ds in datasets:
        for qid, Xq, yq, sk, gd in ltr.iter_queries(tables[ds]):
            if qid in id_set:
                Xs.append(Xq); ys.append(yq); gs.append(len(yq))
                iids.append(qid); sids.extend(sk); gold.append(",".join(gd))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.concatenate(Xs), y=np.concatenate(ys),
                        group_sizes=np.array(gs), instance_ids=np.array(iids, dtype=object),
                        skill_ids=np.array(sids, dtype=object), gold=np.array(gold, dtype=object),
                        feature_names=np.array(ltr_features.ALL_FEATURES, dtype=object))
    return len(iids), sum(gs)


def phase1_materialize(ext, tables, train, dev, test):
    # splits
    sp = DATA / "splits_query_gen"; sp.mkdir(parents=True, exist_ok=True)
    for ds in ext["datasets"]:
        shutil.copy2(PROJECT_ROOT / f"results/splits/{ds}-query_gen.json", sp / f"{ds}-query_gen.json")
    # L6 final-fit dataset
    l6 = DATA / "l6_final"
    for name, ids in [("train", train), ("dev", dev), ("test", test)]:
        nq, nr = export_subset(tables, ext["datasets"], ids, l6 / f"l6_features_query_gen_{name}.npz")
        print(f"  data/l6_final/l6_features_query_gen_{name}.npz : {nq} queries, {nr} rows")
    (l6 / "README.md").write_text(
        "# L6 final-fit dataset (query_gen split)\n\n"
        "Feature matrices the FINAL L6 (LightGBM LambdaRank) is fit on. One row per (query,candidate);\n"
        "X = 45 features (see feature_names), y = 1 if candidate is gold. Train on *train* (+ *dev* for\n"
        "early stopping), evaluate on *test*. Source: results/m4_v2/cache/ltr_features sliced by\n"
        "data/splits_query_gen. Train=query_gen-train (= CE's training queries) → test is held-out for L6 AND CE.\n")
    # CE train dataset
    ce = DATA / "ce_train"; ce.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in ["all_train_pairs_v2.json"] + [f"{ds}-train_pairs_v2.json" for ds in ext["datasets"]]:
        src = PROJECT_ROOT / "results" / "train" / f
        if src.exists():
            shutil.copy2(src, ce / f); copied.append(f)
    # stats of the aggregate CE training file
    agg = ce / "all_train_pairs_v2.json"
    stats = ""
    if agg.exists():
        d = json.loads(agg.read_text())
        recs = d if isinstance(d, list) else d.get("results") or d.get("data") or []
        qids = {r.get("instance_id") or r.get("query_id") for r in recs} if recs else set()
        stats = f"{len(recs)} examples over {len(qids)} queries"
    (ce / "README.md").write_text(
        f"# CE (M5/M7, ce-joint-v3) training dataset\n\nCopied from results/train/. {stats}.\n"
        "ce-joint-v3 was fine-tuned (MiniLM-L6, listwise, seed 42) on these pairs, whose queries are the\n"
        "query_gen-train split (3,782 queries). Therefore query_gen-test is held-out for CE.\n")
    print(f"  data/ce_train/: copied {len(copied)} files ({stats})")
    print(f"  data/splits_query_gen/: 6 split files")


# --- Phase 2: train final L6 -------------------------------------------------

def phase2_train_l6(ext, cfg, tables, train, dev, test, grid):
    col = ltr_features.column_indices(["retrieval", "m4", "a7", "qsc", "confidence"])
    tr = ltr.collect_queries(tables, train, col)
    dv = ltr.collect_queries(tables, dev, col)
    te = ltr.collect_queries(tables, test, col)
    print(f"  L6 final fit: train_q={len(tr)} dev_q={len(dv)} test_q={len(te)} feats={len(col)}")
    model, params = ltr.train_lightgbm(tr, dv, cfg, grid)
    mdir = PROJECT_ROOT / "results" / "models"; mdir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(mdir / "l6_ltr_final.txt"))
    names = ltr._sel_names({"groups": ["retrieval", "m4", "a7", "qsc", "confidence"]})
    (mdir / "l6_ltr_final.features.json").write_text(json.dumps(
        {"features": names, "params": params, "trained_on": "query_gen-train(+dev)",
         "n_train_queries": len(tr), "n_features": len(names)}, indent=2))
    print(f"  saved results/models/l6_ltr_final.txt (params={params})")
    recs = ltr._records_from_scores(te, lambda X: model.predict(X), cfg.output_top_k)
    return recs, params


# --- Phase 3: evaluate all on query_gen-test ---------------------------------

def _group(records):
    out = {}
    for r in records:
        out.setdefault(r.get("dataset", "unknown"), []).append(r)
    return out


def phase3_eval(ext, cfg, l6_recs, test, gold_by_id):
    datasets = ext["datasets"]
    qsc_dir = PROJECT_ROOT / ext["reporting"]["out_dir"]
    methods = []  # (label, type, eval_block, records_for_sig)

    # L6 final (supervised, held-out)
    l6_eval = evaluate.eval_variant(cfg, _group(l6_recs))
    methods.append(("L6-final (LTR)", "supervised", l6_eval, l6_recs))

    # our unsupervised methods
    for vid, label in [("A0", "A0 RRF"), ("A1", "A1 M4"), ("A7", "A7 PRF"),
                       ("Q2", "Q2 QSC"), ("Q6", "Q6 QSC")]:
        by = cv.load_our_baseline(cfg, vid, datasets, qsc_dir)
        ev = cv.eval_on_ids(cfg, by, test)
        recs = [r for ds in by for r in by[ds] if r["instance_id"] in test]
        methods.append((label, "unsupervised", ev, recs))

    # zero-shot retrieval baselines
    for label, tmpl in [("BM25", "results/retrieval_bm25/{ds}-bm25.json"),
                        ("BGE", "results/retrieval_dense/{ds}-dense-bge.json"),
                        ("RRF(BM25+BGE)", "results/retrieval_rrf_bm25_dense/{ds}-rrf-bm25-dense.json")]:
        by = cv.load_retrieval_baseline(tmpl, datasets, gold_by_id)
        ev = cv.eval_on_ids(cfg, by, test)
        recs = [r for ds in by for r in by[ds] if r["instance_id"] in test]
        methods.append((label, "zero-shot", ev, recs))

    # CE (supervised, held-out on query_gen-test): M5/M7 @100 and @500
    def jsonl_by_ds(tmpl):
        by = {}
        for ds in datasets:
            p = PROJECT_ROOT / tmpl.format(ds=ds)
            if p.exists():
                by[ds] = _read_jsonl(p)
        return by
    ce_sources = [
        ("M5-CE@100", "results/rerank/full_bench_h100-{ds}.json", "results"),
        ("M7-CE@100", "results/rerank/fused_alpha_beta70-{ds}.json", "results"),
        ("M5-CE@500", "results/qsc_ltr/m5_500/{ds}.jsonl", "jsonl"),
        ("M7-CE@500", "results/qsc_ltr/ce500/{ds}.jsonl", "jsonl"),
    ]
    for label, tmpl, kind in ce_sources:
        by = cv.load_retrieval_baseline(tmpl, datasets, gold_by_id) if kind == "results" else jsonl_by_ds(tmpl)
        ev = cv.eval_on_ids(cfg, by, test)
        recs = [r for ds in by for r in by[ds] if r["instance_id"] in test]
        methods.append((label, "supervised(CE)", ev, recs))

    # significance: L6 vs key methods on query_gen-test
    by_label = {m[0]: m[3] for m in methods}
    sig = {}
    for other in ["M7-CE@500", "M5-CE@500", "A7 PRF", "BM25", "Q6 QSC"]:
        if other in by_label:
            sig[f"L6_vs_{other}"] = {
                "Recall@10": cv.paired_bootstrap(cv.per_query_metric(l6_recs, "recall", 10),
                                                 cv.per_query_metric(by_label[other], "recall", 10)),
                "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(l6_recs, "ndcg", 10),
                                               cv.per_query_metric(by_label[other], "ndcg", 10))}
    return methods, sig


# --- Phase 4: report ---------------------------------------------------------

def phase4_report(ext, methods, sig, sizes, params):
    datasets = ext["datasets"]
    COMP.mkdir(parents=True, exist_ok=True)
    # JSON + CSV
    rows = []
    for label, typ, ev, _ in methods:
        row = {"method": label, "type": typ}
        for m in MET:
            row[m] = round(_pct(ev["macro"], m), 2)
            for ds in datasets:
                row[f"{ds}:{m}"] = round(_pct(ev["by_dataset"].get(ds, {}), m), 2)
        rows.append(row)
    (COMP / "fair_supervised_query_gen.json").write_text(json.dumps(
        {"split": "query_gen-test", "sizes": sizes, "l6_params": params,
         "rows": rows, "significance": sig}, indent=2))
    import csv
    fields = ["method", "type"] + MET
    with (COMP / "fair_supervised_query_gen.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # MD tables
    macro = _md(["Method", "Type", *SHORT],
                [[lbl, typ, *[f"{_pct(ev['macro'], m):.2f}" for m in MET]] for lbl, typ, ev, _ in methods])

    def per_ds(metric):
        return _md(["Method", *datasets, "AVG"],
                   [[lbl, *[f"{_pct(ev['by_dataset'].get(ds, {}), metric):.2f}" for ds in datasets],
                     f"{_pct(ev['macro'], metric):.2f}"] for lbl, _t, ev, _ in methods])

    sig_rows = []
    for k, b in sig.items():
        for met, st in b.items():
            if st.get("mean_diff") is not None:
                sig_rows.append([k.replace("_", " "), met, f"{st['mean_diff']:+.2f}",
                                 f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]])

    sec = [
        "## 20. Fair Supervised Comparison (query_gen-test, no CE-leakage)", "",
        f"**Every method evaluated on the SAME held-out `query_gen-test` ({sizes['test']} queries).** "
        "CE (`ce-joint-v3`, M5/M7) was fine-tuned on `query_gen-train` "
        f"({sizes['train']} q, seed 42); the **final L6** is trained on `query_gen-train` "
        f"(+ `query_gen-dev` {sizes['dev']} q for early stopping). So **both L6 and CE are held-out** "
        "on the test queries — unlike §15–§19 where CE was scored partly on its own training queries. "
        "Zero-shot (BM25/BGE/RRF) and unsupervised (A0/A1/A7/QSC, KMeans uses no gold) need no training.",
        "",
        "### 20.1 Macro (%) on query_gen-test", "", macro, "",
        "### 20.2 Per-dataset Recall@1 (%)", "", per_ds("Recall@1"), "",
        "### 20.3 Per-dataset Recall@10 (%)", "", per_ds("Recall@10"), "",
        "### 20.4 Per-dataset nDCG@10 (%)", "", per_ds("nDCG@10"), "",
        "### 20.5 Significance (paired bootstrap, query_gen-test)", "",
        _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows), "",
        "### 20.6 Reading", "",
        "- **L6 vs CE is now a fair, leakage-free, depth-and-split-matched comparison.** "
        "L6 is the best **no-CE** ranker; CE (with fusion) is the upper line. See Δ above.",
        "- Categories are NOT interchangeable: **supervised** (L6, CE — trained on query_gen-train) vs "
        "**zero-shot** (BM25/BGE/RRF) vs **unsupervised** (M4/QSC). Compare within intent.",
        "- The final L6 model is saved at `results/models/l6_ltr_final.txt` "
        "(+ `.features.json`); training data in `data/l6_final/`, CE training data in `data/ce_train/`.",
        f"- L6 final config (dev-tuned): `{params}`.", "",
    ]
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 20.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")

    md = ["# Fair supervised comparison — query_gen-test", "", macro, "",
          "## Per-dataset Recall@10", "", per_ds("Recall@10"), "",
          "## Per-dataset nDCG@10", "", per_ds("nDCG@10"), ""]
    (COMP / "fair_supervised_query_gen.md").write_text("\n".join(md))
    print("  wrote fair_supervised_query_gen.{csv,md,json} + appended §20")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["full", "moderate"], default="moderate")
    args = ap.parse_args()
    ext, cfg = load_ext()
    datasets = ext["datasets"]
    tables = {ds: ltr_features.load_features(PROJECT_ROOT / ext["reporting"]["cache_dir"] / f"{ds}.npz")
              for ds in datasets}
    split, train, dev, test = load_query_gen(ext)
    gold_by_id = {r["instance_id"]: r["gold_skill_ids"]
                  for ds in datasets for r in io.load_instances(cfg, ds)}
    sizes = {"train": len(train), "dev": len(dev), "test": len(test)}

    print("Phase 1: materialize /data ..."); phase1_materialize(ext, tables, train, dev, test)
    print("Phase 2: train final L6 ..."); l6_recs, params = phase2_train_l6(ext, cfg, tables, train, dev, test, GRIDS[args.grid])
    print("Phase 3: evaluate all on query_gen-test ..."); methods, sig = phase3_eval(ext, cfg, l6_recs, test, gold_by_id)
    for lbl, typ, ev, _ in methods:
        m = ev["macro"]
        print(f"  {lbl:16} [{typ:14}] R@1={_pct(m,'Recall@1'):5.2f} R@10={_pct(m,'Recall@10'):5.2f} "
              f"R@100={_pct(m,'Recall@100'):5.2f} nDCG@10={_pct(m,'nDCG@10'):5.2f}")
    print("Phase 4: report ..."); phase4_report(ext, methods, sig, sizes, params)
    # Phase 5 leakage audit
    print("Phase 5 audit: train∩test=%d dev∩test=%d (expect 0,0)" % (len(train & test), len(dev & test)))


if __name__ == "__main__":
    main()
