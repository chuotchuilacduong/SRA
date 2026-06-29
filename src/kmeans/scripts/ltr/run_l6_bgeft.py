"""§23 — bge_ft as an L6 feature: L6-final vs L6-final+bge_ft (query_gen-test).

Trains TWO LightGBM LambdaRank final models on query_gen-train(+dev), evaluated on the
SAME held-out query_gen-test (1,079) as §20/§21/§22:
  * L6-final         : 45 features (baseline groups)         -- reproduces §20 L6
  * L6-final+bge_ft  : 52 features (baseline + bge_ft group) -- NEW separate method
Both read the augmented cache results/m4_v2/cache/ltr_features_bgeft/{ds}.npz (built by
add_bgeft_features.py). Context rows `bge_ft (retriever-only)` and `M7-CE@500` are pulled
from the existing §21/§20 comparison JSONs (identical eval pipeline → directly comparable).

Everything is ADDITIVE (nothing overwritten):
  * results/models/l6_ltr_final_bgeft.txt (+ .features.json with gain importances)
  * results/comparisons/l6_bgeft.{json,md}
  * results/retrieval/{ds}-l6_bgeft.json     (new end-task retrieval source)
  * §23 appended to FULL_M4_V2_RESULTS.md

Run: python src/kmeans/scripts/run_l6_bgeft.py [--grid full|moderate]
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
_bootstrap.setup_logging()

import numpy as np                                       # noqa: E402
from sragents.config import PROJECT_ROOT                 # noqa: E402
from kmeans.qsc_ltr_runner import load_ext               # noqa: E402
from kmeans import cv, ltr, ltr_features, evaluate        # noqa: E402

COMP = PROJECT_ROOT / "results" / "comparisons"
RETR = PROJECT_ROOT / "results" / "retrieval"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
BGEFT_CACHE = PROJECT_ROOT / "results" / "m4_v2" / "cache" / "ltr_features_bgeft"
MET = evaluate.REPORT_METRICS
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]
BASE = ["retrieval", "m4", "a7", "qsc", "confidence"]
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
    return train, dev, test


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def _group(records):
    out = {}
    for r in records:
        out.setdefault(r.get("dataset", "unknown"), []).append(r)
    return out


def train_eval(tables, groups, train, dev, test, cfg, grid):
    col = ltr_features.column_indices(groups)
    tr = ltr.collect_queries(tables, train, col)
    dv = ltr.collect_queries(tables, dev, col)
    te = ltr.collect_queries(tables, test, col)
    print(f"  [{'+'.join(groups)}] feats={len(col)} train_q={len(tr)} dev_q={len(dv)} test_q={len(te)}",
          flush=True)
    model, params = ltr.train_lightgbm(tr, dv, cfg, grid)
    recs = ltr._records_from_scores(te, lambda X: model.predict(X), cfg.output_top_k)
    ev = evaluate.eval_variant(cfg, _group(recs))
    return model, params, recs, ev, col


# --- metric accessors (percent) over either an eval block or a precomputed json row ---

def macro_pct(entry, m):
    kind, data = entry
    return _pct_block(data["macro"], m) if kind == "ev" else float(data.get(m, float("nan")))


def perds_pct(entry, ds, m):
    kind, data = entry
    if kind == "ev":
        return _pct_block(data["by_dataset"].get(ds, {}), m)
    return float(data.get(f"{ds}:{m}", float("nan")))


def _pct_block(block, m):
    return block.get(m, float("nan")) * 100.0


def load_json_row(path, method):
    if not path.exists():
        return None
    for r in json.loads(path.read_text()).get("rows", []):
        if r.get("method") == method:
            return r
    return None


def bge_ft_alone_recs(datasets, test_ids):
    """Per-query records for bge_ft (retriever-only) from its end-task source files."""
    recs = []
    for ds in datasets:
        p = RETR / f"{ds}-bge_ft_retriever.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("results", []):
            if r["instance_id"] in test_ids:
                recs.append(r)
    return recs


def export_source(recs, name):
    RETR.mkdir(parents=True, exist_ok=True)
    by = _group(recs)
    for ds, rs in by.items():
        (RETR / f"{ds}-{name}.json").write_text(json.dumps({"results": rs}))
    print(f"  exported results/retrieval/{{ds}}-{name}.json for {len(by)} datasets", flush=True)


def write_report(methods, sig, imp, datasets):
    """Render §23 (macro + per-dataset R@1/R@10/nDCG@1/nDCG@10 + significance + gain
    importance) and write it to FULL_M4_V2_RESULTS.md + results/comparisons/l6_bgeft.md.
    Reads metrics off `methods` entries, so it works both from freshly-trained eval
    blocks (kind="ev") and from saved JSON rows (kind="row", report-only mode)."""
    macro = _md(["Method", "Type", *SHORT],
                [[lbl, typ, *[f"{macro_pct(e, m):.2f}" for m in MET]] for lbl, typ, e in methods])

    def per_ds(metric):
        return _md(["Method", *datasets, "AVG"],
                   [[lbl, *[f"{perds_pct(e, ds, metric):.2f}" for ds in datasets],
                     f"{macro_pct(e, metric):.2f}"] for lbl, _t, e in methods])

    sig_rows = [[k.replace("_", " "), met, f"{st['mean_diff']:+.2f}",
                 f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]]
                for k, b in sig.items() for met, st in b.items() if st.get("mean_diff") is not None]
    imp_rows = [[i + 1, n, f"{g:.0f}", "← bge_ft" if n in ltr_features.FEATURE_GROUPS["bge_ft"] else ""]
                for i, (n, g) in enumerate(imp[:12])]
    nd = lambda lbl: macro_pct(next(e for l, _t, e in methods if l == lbl), "nDCG@10")

    sec = [
        "## 23. bge_ft as an L6 feature — L6-final vs L6-final+bge_ft (query_gen-test)", "",
        "Feeds the **fine-tuned retriever** `sr-emb-bge-v1` signal into L6 as a new 7-feature group "
        "(`bge_ft`: cosine + within-pool rank/inv-rank/rank-norm/is-top1/margin/z). `L6-final+bge_ft` "
        "(52 feats) is trained the same way as `L6-final` (45 feats) on query_gen-train(+dev) and "
        "evaluated on the SAME held-out query_gen-test (1,079) as §20–§22. `bge_ft (retriever-only)` "
        "and `M7-CE@500` are shown for reference (from §21/§20).", "",
        "_Category nuance: L6 ranks the **M4 Stage-1 pool** (gold-in-pool ≈97.6%); bge_ft-alone "
        "retrieves from the **full 26,262-skill corpus** (R@100 ≈99.3%). So L6+bge_ft uses bge_ft as a "
        "*signal inside the pool* and is bounded by the pool's recall — compare ordering (nDCG), and see "
        "retriever-only R@100 for the ceiling._", "",
        "### 23.1 Macro (%) on query_gen-test", "", macro, "",
        "### 23.2 Per-dataset Recall@1 (%)", "", per_ds("Recall@1"), "",
        "### 23.3 Per-dataset Recall@10 (%)", "", per_ds("Recall@10"), "",
        "### 23.4 Per-dataset nDCG@1 (%)", "", per_ds("nDCG@1"), "",
        "### 23.5 Per-dataset nDCG@10 (%)", "", per_ds("nDCG@10"), "",
        "### 23.6 Significance (paired bootstrap, query_gen-test)", "",
        _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows) if sig_rows else "_(n/a)_", "",
        "### 23.7 LightGBM gain importance (top 12 of L6-final+bge_ft)", "",
        _md(["#", "Feature", "Gain", "Group"], imp_rows), "",
        "### 23.8 Reading", "",
        f"- L6-final+bge_ft macro nDCG@10 = **{nd('L6-final+bge_ft'):.2f}** vs "
        f"L6-final {nd('L6-final'):.2f} (Δ {nd('L6-final+bge_ft') - nd('L6-final'):+.2f} pp; see 23.6 for p).",
        "- If `bgeft_*` features rank high in 23.7, L6 is genuinely using the fine-tuned signal.",
        "- L6+bge_ft ranks the M4 pool, so it is bounded by ~97.6% pool recall; it cannot exceed "
        "bge_ft-alone on recall without widening the pool (plan §6). Model: "
        "`results/models/l6_ltr_final_bgeft.txt`; source: `results/retrieval/{ds}-l6_bgeft.json`.", "",
    ]
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 23.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")

    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "l6_bgeft.md").write_text("\n".join(
        ["# L6 vs L6+bge_ft — query_gen-test", "", macro, "",
         "## Per-dataset Recall@1", "", per_ds("Recall@1"), "",
         "## Per-dataset Recall@10", "", per_ds("Recall@10"), "",
         "## Per-dataset nDCG@1", "", per_ds("nDCG@1"), "",
         "## Per-dataset nDCG@10", "", per_ds("nDCG@10"), ""]))


# Standalone reference methods to show alongside L6/L6+bge_ft in §23. Their metrics are
# pulled from §21 (ceraw) / §20 (fair) comparison JSONs — same eval pipeline, so directly
# comparable. Order = display order after the two L6 rows.
CONTEXT_FROM_FAIR = ["M7-CE@500", "BM25", "BGE", "RRF(BM25+BGE)",
                     "A1 M4", "A7 PRF", "Q6 QSC", "A0 RRF", "Q2 QSC"]


def context_methods(datasets):
    """Reference rows (bge_ft-alone + CE + zero-shot/unsupervised baselines) from §20/§21."""
    out = []
    bg = load_json_row(COMP / "ceraw_query_gen.json", "bge_ft (retriever-only)")
    if bg:
        out.append((bg["method"], bg.get("type", "retriever"), ("row", bg)))
    for name in CONTEXT_FROM_FAIR:
        r = load_json_row(COMP / "fair_supervised_query_gen.json", name)
        if r:
            out.append((r["method"], r.get("type", ""), ("row", r)))
    return out


def save_and_report(computed, sig, imp, datasets, p_l6, p_bg):
    """methods = the trained L6 rows + standalone reference rows; persist l6_bgeft.json
    and render §23. Single assembly path for both full and --report-only runs."""
    methods = list(computed) + context_methods(datasets)
    COMP.mkdir(parents=True, exist_ok=True)
    rows_json = []
    for lbl, typ, e in methods:
        row = {"method": lbl, "type": typ}
        for m in MET:
            row[m] = round(macro_pct(e, m), 2)
            for ds in datasets:
                row[f"{ds}:{m}"] = round(perds_pct(e, ds, m), 2)
        rows_json.append(row)
    (COMP / "l6_bgeft.json").write_text(json.dumps(
        {"split": "query_gen-test", "l6_params": p_l6, "l6_bgeft_params": p_bg,
         "rows": rows_json, "significance": sig, "gain_importance": imp}, indent=2))
    write_report(methods, sig, imp, datasets)
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["full", "moderate"], default="moderate")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild §23 from results/comparisons/l6_bgeft.json without retraining")
    args = ap.parse_args()
    ext, cfg = load_ext()
    datasets = ext["datasets"]

    if args.report_only:
        d = json.loads((COMP / "l6_bgeft.json").read_text())
        computed = [(r["method"], r["type"], ("row", r)) for r in d["rows"]
                    if r["method"] in ("L6-final", "L6-final+bge_ft")]
        save_and_report(computed, d["significance"], d["gain_importance"], datasets,
                        d.get("l6_params"), d.get("l6_bgeft_params"))
        print("rebuilt §23 (report-only) from results/comparisons/l6_bgeft.json", flush=True)
        return

    grid = GRIDS[args.grid]

    tables = {ds: ltr_features.load_features(BGEFT_CACHE / f"{ds}.npz") for ds in datasets}
    ncols = {ds: tables[ds]["X"].shape[1] for ds in datasets}
    assert all(c == len(ltr_features.ALL_FEATURES) for c in ncols.values()), \
        f"bgeft cache must have {len(ltr_features.ALL_FEATURES)} cols; got {ncols}. Run add_bgeft_features.py."
    train, dev, test = load_query_gen(ext)
    print(f"sizes: train={len(train)} dev={len(dev)} test={len(test)}", flush=True)

    # --- train the two final models on the SAME data pipeline ---
    print("Train L6-final (baseline 45) ...", flush=True)
    _, p_l6, recs_l6, ev_l6, _ = train_eval(tables, BASE, train, dev, test, cfg, grid)
    print("Train L6-final+bge_ft (52) ...", flush=True)
    m_bg, p_bg, recs_bg, ev_bg, col_bg = train_eval(tables, BASE + ["bge_ft"], train, dev, test, cfg, grid)

    # save the new model + gain importances (diagnostic: how high do bgeft_* rank?)
    mdir = PROJECT_ROOT / "results" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    m_bg.booster_.save_model(str(mdir / "l6_ltr_final_bgeft.txt"))
    names = [ltr_features.ALL_FEATURES[i] for i in col_bg]
    gain = m_bg.booster_.feature_importance(importance_type="gain")
    imp = sorted(zip(names, [float(g) for g in gain]), key=lambda kv: -kv[1])
    (mdir / "l6_ltr_final_bgeft.features.json").write_text(json.dumps(
        {"features": names, "params": p_bg, "trained_on": "query_gen-train(+dev)",
         "n_features": len(names), "gain_importance": imp}, indent=2))

    computed = [
        ("L6-final", "supervised", ("ev", ev_l6)),
        ("L6-final+bge_ft", "supervised", ("ev", ev_bg)),
    ]

    # --- significance: L6+bge_ft vs L6, and vs bge_ft-alone ---
    sig = {}
    qm = lambda recs, met, k: cv.per_query_metric(recs, met, k)
    sig["L6+bge_ft_vs_L6"] = {
        "Recall@10": cv.paired_bootstrap(qm(recs_bg, "recall", 10), qm(recs_l6, "recall", 10)),
        "nDCG@10": cv.paired_bootstrap(qm(recs_bg, "ndcg", 10), qm(recs_l6, "ndcg", 10))}
    alone = bge_ft_alone_recs(datasets, test)
    if alone:
        sig["L6+bge_ft_vs_bge_ft-alone"] = {
            "Recall@10": cv.paired_bootstrap(qm(recs_bg, "recall", 10), qm(alone, "recall", 10)),
            "nDCG@10": cv.paired_bootstrap(qm(recs_bg, "ndcg", 10), qm(alone, "ndcg", 10))}

    export_source(recs_bg, "l6_bgeft")
    methods = save_and_report(computed, sig, imp, datasets, p_l6, p_bg)

    print("\nSUMMARY (macro nDCG@10):", flush=True)
    for lbl, _t, e in methods:
        print(f"  {lbl:24} {macro_pct(e, 'nDCG@10'):6.2f}", flush=True)
    print("wrote results/comparisons/l6_bgeft.{json,md} + appended §23", flush=True)


if __name__ == "__main__":
    main()
