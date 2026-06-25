"""Report writer for the 5-fold CV / significance / fair-baseline validation.

Consumes results/qsc_ltr/cv/cv_consolidated.json and writes:
  results/comparisons/ltr_5fold_cv_metrics.{csv,md,json}
  results/comparisons/ltr_5fold_cv_delta.md
  results/comparisons/ltr_vs_ce_method7_5fold.{csv,md,json}
  results/analysis/ltr_cv_error_analysis.md
and appends §15-§19 to FULL_M4_V2_RESULTS.md. Retrieval-only validation (no
end-task baselines exist in the repo) — stated explicitly.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from sragents.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

COMP = PROJECT_ROOT / "results" / "comparisons"
ANALYSIS = PROJECT_ROOT / "results" / "analysis"
CV_DIR = PROJECT_ROOT / "results" / "qsc_ltr" / "cv"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100",
           "nDCG@1", "nDCG@5", "nDCG@10"]
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]
NO_CE_BASELINES = ["A0", "A7", "Q6", "BM25", "BGE", "RRF(BM25+BGE)", "Hybrid-official", "LinearRAG"]


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def _ms(stat):
    """mean±std (ci95) from a fold_stats entry."""
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.2f}±{stat.get('std', 0):.2f}"


# --- collect a uniform method roster -----------------------------------------

def _roster(c):
    """Ordered [(label, kind, fold_stats, point_macro, by_dataset)]; kind in
    {ltr, baseline, ce}. point_macro is OOF (ltr) or full-set (baseline)."""
    R = []
    for vid in ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
        if vid in c["ltr"]:
            e = c["ltr"][vid]
            R.append((vid, "ltr", e["fold_stats"], e["oof_macro"], e.get("oof_by_dataset", {})))
    if "L6" in c["ltr"] and "d100_fold_stats" in c["ltr"]["L6"]:
        e = c["ltr"]["L6"]
        R.append(("L6@100", "ltr", e["d100_fold_stats"], e.get("d100_oof_macro", {}), {}))
    for name, e in c["baselines"].items():
        kind = "ce" if "Method7" in name else "baseline"
        R.append((name, kind, e["fold_stats"], e["full_macro"], e.get("full_by_dataset", {})))
    return R


# --- per-dataset table over ALL methods (5-fold CV test standard) ------------

def _all_methods_per_dataset(c):
    """Ordered [(label, by_dataset, macro)] over every method, all on the same query
    universe: LTR = out-of-fold (each query test once); baselines = full set (= same
    queries). AVG = macro mean over the 6 datasets."""
    rows = []
    base_order = ["BM25", "BGE", "RRF(BM25+BGE)", "Hybrid-official", "LinearRAG",
                  "A0", "A1", "A7", "Q2", "Q6"]
    labels = {"A0": "A0 RRF", "A1": "A1 M4", "A7": "A7 PRF", "Q2": "Q2 QSC", "Q6": "Q6 QSC"}
    for name in base_order:
        if name in c["baselines"]:
            b = c["baselines"][name]
            rows.append((labels.get(name, name), b.get("full_by_dataset", {}), b["full_macro"]))
    for vid in ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]:
        if vid in c["ltr"]:
            e = c["ltr"][vid]
            rows.append((f"{vid} {e.get('name', '')}".strip(), e.get("oof_by_dataset", {}), e["oof_macro"]))
    for name, disp in [("Method7-CE", "Method7-CE@100"), ("Method7-CE@500", "Method7-CE@500")]:
        if name in c["baselines"]:
            b = c["baselines"][name]
            rows.append((disp, b.get("full_by_dataset", {}), b["full_macro"]))
    return rows


def per_dataset_metric_table(c, metric):
    datasets = c["datasets"]
    rows = []
    for label, byds, macro in _all_methods_per_dataset(c):
        rows.append([label,
                     *[(f"{byds[ds][metric]:.2f}" if ds in byds and metric in byds[ds] else "n/a")
                       for ds in datasets],
                     f"{macro.get(metric, float('nan')):.2f}"])
    return _md(["Method", *datasets, "AVG"], rows)


# --- §15 metrics tables ------------------------------------------------------

def write_metrics(c):
    COMP.mkdir(parents=True, exist_ok=True)
    roster = _roster(c)
    # CSV/JSON: per method per metric mean/std/ci95 + point
    rows = []
    for label, kind, fs, point, _ in roster:
        for m in METRICS:
            s = fs.get(m, {})
            rows.append({"method": label, "kind": kind, "metric": m,
                         "fold_mean": s.get("mean"), "fold_std": s.get("std"),
                         "ci95": s.get("ci95"), "point_macro": point.get(m),
                         "per_fold": s.get("per_fold")})
    with (COMP / "ltr_5fold_cv_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "kind", "metric", "fold_mean",
                                          "fold_std", "ci95", "point_macro", "per_fold"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["per_fold"] = json.dumps(r["per_fold"]); w.writerow(r)
    (COMP / "ltr_5fold_cv_metrics.json").write_text(json.dumps(
        {"n_folds": c["n_folds"], "seed": c["seed"], "rows": rows}, indent=2))

    # MD: macro mean±std table
    headers = ["Method", *SHORT]
    body = [[label, *[_ms(fs.get(m, {})) for m in METRICS]] for label, _k, fs, _p, _ in roster]
    md = ["# LTR 5-Fold CV — macro mean±std (%) across folds", "",
          f"{c['n_folds']} query-level folds (seed {c['seed']}), each query test once. "
          "LTR = out-of-fold; baselines/CE evaluated on the same fold test queries. "
          "L6 uses the full LightGBM grid; L0-L5/L7 a reduced dev-tuned grid.", "",
          _md(headers, body), "",
          "## Per-dataset Recall@1 (%) — all methods", "",
          per_dataset_metric_table(c, "Recall@1"), "",
          "## Per-dataset Recall@10 (%) — all methods", "",
          per_dataset_metric_table(c, "Recall@10"), ""]
    (COMP / "ltr_5fold_cv_metrics.md").write_text("\n".join(md))
    # dedicated per-dataset file
    pd = ["# Per-dataset comparison (5-fold CV test standard)", "",
          "## Recall@1 (%)", "", per_dataset_metric_table(c, "Recall@1"), "",
          "## Recall@10 (%)", "", per_dataset_metric_table(c, "Recall@10"), "",
          "## nDCG@10 (%)", "", per_dataset_metric_table(c, "nDCG@10"), ""]
    (COMP / "ltr_5fold_cv_per_dataset.md").write_text("\n".join(pd))
    logger.info("wrote ltr_5fold_cv_metrics.{csv,json,md} + per_dataset.md")


# --- §18 delta + significance ------------------------------------------------

def write_delta(c):
    sig = c["significance"]
    out = ["# LTR 5-Fold CV — paired deltas + significance", "",
           "Paired bootstrap over shared queries (10k resamples); Δ in pp, 95% CI, "
           "two-sided p-value. L6 = OOF (depth-500) unless noted; L6@100 is depth-matched "
           "to CE Method 7.", ""]
    headers = ["Comparison", "Metric", "Δ (pp)", "95% CI", "p-value", "significant (p<0.05)"]
    rows = []
    for key, block in sig.items():
        for metric, st in block.items():
            if st.get("mean_diff") is None:
                continue
            star = "yes" if (st["p_value"] is not None and st["p_value"] < 0.05) else "no"
            rows.append([key.replace("_", " "), metric, f"{st['mean_diff']:+.2f}",
                         f"[{st['ci95_low']:+.2f}, {st['ci95_high']:+.2f}]", st["p_value"], star])
    out += [_md(headers, rows), ""]
    (COMP / "ltr_5fold_cv_delta.md").write_text("\n".join(out))
    logger.info("wrote ltr_5fold_cv_delta.md")


# --- §17 L6 vs CE Method 7 ---------------------------------------------------

def write_ce_comparison(c):
    if "Method7-CE" not in c["baselines"]:
        return
    m7 = c["baselines"]["Method7-CE"]
    l6 = c["ltr"]["L6"]
    datasets = c["datasets"]
    # macro point comparison (depth-matched at 100 AND at 500)
    m7_500 = c["baselines"].get("Method7-CE@500", {}).get("full_macro")
    pairs = [("Method7-CE@100", m7["full_macro"]),
             ("L6@100 (depth-matched)", l6.get("d100_oof_macro", {}))]
    if m7_500:
        pairs += [("Method7-CE@500", m7_500), ("L6@500 (depth-matched)", l6["oof_macro"])]
    else:
        pairs += [("L6@500", l6["oof_macro"])]
    macro_rows = [[label, *[f"{point.get(m, float('nan')):.2f}" for m in METRICS]]
                  for label, point in pairs]
    # per-dataset R@10 / nDCG@10
    pd_rows = []
    for ds in datasets:
        m7d = m7.get("full_by_dataset", {}).get(ds, {})
        l6d = l6.get("oof_by_dataset", {}).get(ds, {})
        pd_rows.append([ds, f"{m7d.get('Recall@10', float('nan')):.2f}",
                        f"{l6d.get('Recall@10', float('nan')):.2f}",
                        f"{m7d.get('nDCG@10', float('nan')):.2f}",
                        f"{l6d.get('nDCG@10', float('nan')):.2f}"])
    md = ["# L6 vs Cross-Encoder Method 7 (matched query set)", "",
          "Method 7 = α-CE β=0.7 (CE rerank + Stage-1 fusion). Both rank the RRF pool; "
          "Method 7 ranks top-100, so **L6@100** is the depth-matched comparison.", "",
          "## Macro (%)", "", _md(["Method", *SHORT], macro_rows), "",
          "## Per-dataset (Method7 vs L6@500)", "",
          _md(["Dataset", "M7 R@10", "L6 R@10", "M7 nD@10", "L6 nD@10"], pd_rows), ""]
    (COMP / "ltr_vs_ce_method7_5fold.md").write_text("\n".join(md))
    (COMP / "ltr_vs_ce_method7_5fold.json").write_text(json.dumps(
        {"method7_macro": m7["full_macro"], "l6_d100_macro": l6.get("d100_oof_macro", {}),
         "l6_full_macro": l6["oof_macro"], "significance": {k: v for k, v in c["significance"].items()
                                                            if "Method7" in k}}, indent=2))
    with (COMP / "ltr_vs_ce_method7_5fold.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", *METRICS])
        for label, point in [("Method7-CE", m7["full_macro"]),
                             ("L6@100", l6.get("d100_oof_macro", {})), ("L6@500", l6["oof_macro"])]:
            w.writerow([label, *[point.get(m) for m in METRICS]])
    logger.info("wrote ltr_vs_ce_method7_5fold.{md,json,csv}")


# --- error analysis (L6 vs A7, L6 vs CE) -------------------------------------

def _read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _oof(vid):
    recs = []
    for i in range(5):
        recs += _read_jsonl(CV_DIR / vid / f"fold{i}.jsonl")
    return {r["instance_id"]: r for r in recs}


def _full_records(cfg, slug_or_tmpl, datasets, is_tmpl=False):
    out = {}
    for ds in datasets:
        p = (PROJECT_ROOT / slug_or_tmpl.format(ds=ds)) if is_tmpl else \
            (cfg.paths["out_dir"] / slug_or_tmpl / f"{ds}.jsonl")
        recs = _read_jsonl(p) if not is_tmpl else \
            [{"instance_id": r["instance_id"], "gold_skill_ids": r.get("gold_skill_ids") or [],
              "retrieved": r["retrieved"], "dataset": ds}
             for r in json.loads(p.read_text())["results"]] if p.exists() else []
        for r in recs:
            r.setdefault("dataset", ds)
            out[r["instance_id"]] = r
    return out


def write_error_analysis(cfg, c):
    datasets = c["datasets"]
    l6 = _oof("L6")
    a7 = _full_records(cfg, cfg.variant("A7").slug, datasets)
    m7 = _full_records(cfg, "results/rerank/fused_alpha_beta70-{ds}.json", datasets, is_tmpl=True)

    def hit10(rec, gold):
        return any(c2["skill_id"] in gold for c2 in rec["retrieved"][:10])

    def grank(rec, gold):
        for c2 in rec["retrieved"]:
            if c2["skill_id"] in gold:
                return c2["rank"]
        return None

    buckets = {"L6_fixes_A7": [], "L6_breaks_A7": [], "CE_wins_L6_loses": [], "L6_wins_CE_loses": []}
    for qid, lr in l6.items():
        gold = set(lr.get("gold_skill_ids") or [])
        if not gold:
            continue
        ar = a7.get(qid); mr = m7.get(qid)
        ds = lr.get("dataset")
        if ar is not None:
            if (not hit10(ar, gold)) and hit10(lr, gold):
                buckets["L6_fixes_A7"].append((ds, qid, grank(ar, gold), grank(lr, gold)))
            elif hit10(ar, gold) and (not hit10(lr, gold)):
                buckets["L6_breaks_A7"].append((ds, qid, grank(ar, gold), grank(lr, gold)))
        if mr is not None:
            if hit10(mr, gold) and (not hit10(lr, gold)):
                buckets["CE_wins_L6_loses"].append((ds, qid, grank(mr, gold), grank(lr, gold)))
            elif (not hit10(mr, gold)) and hit10(lr, gold):
                buckets["L6_wins_CE_loses"].append((ds, qid, grank(mr, gold), grank(lr, gold)))

    L = ["# LTR 5-Fold CV — Error Analysis (OOF L6 vs A7 and CE Method 7)", "",
         "Buckets at Recall@10 (gold in top-10). Ranks: (baseline → L6).", ""]
    titles = {"L6_fixes_A7": "L6 fixes A7 (A7 miss@10 → L6 hit@10)",
              "L6_breaks_A7": "L6 breaks A7 (A7 hit@10 → L6 miss@10)",
              "CE_wins_L6_loses": "CE Method 7 wins, L6 loses (CE hit@10, L6 miss@10)",
              "L6_wins_CE_loses": "L6 wins, CE Method 7 loses (L6 hit@10, CE miss@10)"}
    for key, items in buckets.items():
        L += [f"## {titles[key]}  (n={len(items)})", ""]
        if not items:
            L += ["_none_", ""]; continue
        rows = [[ds, qid, f"{b}→{n}"] for ds, qid, b, n in items[:30]]
        L += [_md(["dataset", "query_id", "rank base→L6"], rows), ""]
        if len(items) > 30:
            L += [f"_…and {len(items)-30} more._", ""]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "ltr_cv_error_analysis.md").write_text("\n".join(L))
    logger.info("wrote ltr_cv_error_analysis.md (fixes=%d breaks=%d CE>L6=%d L6>CE=%d)",
                *(len(buckets[k]) for k in ["L6_fixes_A7", "L6_breaks_A7", "CE_wins_L6_loses", "L6_wins_CE_loses"]))
    return {k: len(v) for k, v in buckets.items()}


# --- recommendation (plan §12) -----------------------------------------------

def _recommendation(c):
    l6 = c["ltr"]["L6"]
    l6_nd = l6["fold_stats"]["nDCG@10"]["mean"]
    l6_r10 = l6["fold_stats"]["Recall@10"]["mean"]
    # beats all no-CE baselines? (fold-mean nDCG@10 and R@10)
    beats_all_noce = True
    for b in NO_CE_BASELINES:
        if b not in c["baselines"]:
            continue
        bn = c["baselines"][b]["fold_stats"]["nDCG@10"]["mean"]
        if bn is not None and l6_nd <= bn:
            beats_all_noce = False
    # vs Method 7 (depth-matched L6@100)
    beats_ce = None
    sig_ce = c["significance"].get("L6d100_vs_Method7", {})
    if "Method7-CE" in c["baselines"]:
        m7_nd = c["baselines"]["Method7-CE"]["full_macro"]["nDCG@10"]
        l6_100_nd = l6.get("d100_oof_macro", {}).get("nDCG@10", l6["oof_macro"]["nDCG@10"])
        beats_ce = l6_100_nd >= m7_nd
    detail = {"l6_fold_nDCG@10": l6_nd, "l6_fold_R@10": l6_r10,
              "beats_all_no_ce_baselines": beats_all_noce, "beats_ce_method7_depthmatched": beats_ce,
              "ce_sig": sig_ce}
    if beats_all_noce and beats_ce is False:
        claim = ('**Claim: "best lightweight / no-Cross-Encoder skill retriever-reranker on '
                 'SRA-Bench (retrieval-only)."** L6 robustly beats every no-CE baseline across '
                 '5 folds but does NOT beat the Cross-Encoder Method 7; frame it as the strongest '
                 'no-CE option, competitive with CE at far lower inference cost — NOT overall SOTA.')
    elif beats_ce:
        claim = ('**Claim: "competitive with or better than Cross-Encoder reranking at much lower '
                 'inference cost."** L6 matches/beats Method 7 on matched depth-100 queries without a CE.')
    elif not beats_all_noce:
        claim = ('**Claim: narrower.** L6 does not dominate every no-CE baseline across folds; '
                 'report per-dataset wins and limitations rather than a blanket best-no-CE claim.')
    else:
        claim = '**Claim: mixed; report a narrow, per-dataset claim with limitations.**'
    return claim, detail


# --- append §15-§19 ----------------------------------------------------------

def append_sections(cfg, c, err_counts):
    roster = _roster(c)
    claim, detail = _recommendation(c)
    audit = c["leakage_audit"]

    def macro_table(roster):
        return _md(["Method", *SHORT],
                   [[lbl, *[_ms(fs.get(m, {})) for m in METRICS]] for lbl, _k, fs, _p, _ in roster])

    ce_block = ""
    if "Method7-CE" in c["baselines"]:
        m7 = c["baselines"]["Method7-CE"]["full_macro"]
        m7_500 = c["baselines"].get("Method7-CE@500", {}).get("full_macro", {})
        l6_100 = c["ltr"]["L6"].get("d100_oof_macro", {})
        l6_500 = c["ltr"]["L6"]["oof_macro"]
        rows = [["Method7-CE@100", *[f"{m7.get(m, float('nan')):.2f}" for m in METRICS]],
                ["L6@100 (depth-matched to CE@100)", *[f"{l6_100.get(m, float('nan')):.2f}" for m in METRICS]]]
        if m7_500:
            rows += [["**Method7-CE@500**", *[f"{m7_500.get(m, float('nan')):.2f}" for m in METRICS]],
                     ["**L6@500 (depth-matched to CE@500)**", *[f"{l6_500.get(m, float('nan')):.2f}" for m in METRICS]]]
        else:
            rows += [["L6@500 (deeper pool)", *[f"{l6_500.get(m, float('nan')):.2f}" for m in METRICS]]]
        ce_block = _md(["Method", *SHORT], rows)

    sig_rows = []
    for key, block in c["significance"].items():
        for metric, st in block.items():
            if st.get("mean_diff") is None:
                continue
            sig_rows.append([key.replace("_", " "), metric, f"{st['mean_diff']:+.2f}",
                             f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]])

    fold_audit = _md(["fold", "train", "dev", "test", "tr∩dev", "tr∩te", "dev∩te"],
                     [[a["fold"], a["train"], a["dev"], a["test"], a["train_dev_overlap"],
                       a["train_test_overlap"], a["dev_test_overlap"]] for a in audit["folds"]])

    S = ["## 15. Robustness: LightGBM Grid + 5-Fold CV", "",
         f"{c['n_folds']} query-level folds (seed {c['seed']}); each query is test exactly once "
         "(out-of-fold predictions). Per fold: train on 3 folds, tune on 1 (dev), evaluate once "
         "on the held-out fold (test). **L6 uses the full 36-config LightGBM grid** "
         "(num_leaves{15,31,63}×lr{0.03,0.05}×n_est{200,500}×min_data{10,30,50}) tuned on dev; "
         "L0-L5/L7 use a reduced dev-tuned grid. **Retrieval-only** validation (no end-task "
         "baselines exist in the repo).", "",
         "### Macro mean±std (%) across 5 folds (baselines on the same fold test queries)", "",
         macro_table(roster), "",
         f"Best LightGBM config per fold and per-fold feature importances are in "
         "`results/qsc_ltr/cv/cv_consolidated.json` and "
         "`results/comparisons/ltr_5fold_cv_metrics.{csv,json}`.", "",
         "### Leakage and Fairness Audit", "",
         fold_audit, "",
         f"- Feature/label independence: columns equal to the label = "
         f"`{audit['feature_label_independence']['cols_equal_to_label'] or 'NONE'}` "
         f"(of {audit['feature_label_independence']['n_features']} features; gold used only as `y`).",
         *[f"- {n}" for n in audit["notes"]], "",
         "## 16. Fair Comparison Against Original Paper Baselines", "",
         "All retrieval baselines evaluated on the **same fold test queries** as LTR (macro "
         "mean±std). BM25/Hybrid/LinearRAG are stored top-50 → R@100 capped (n/a). TF-IDF and "
         "Contriever are **not available** as cached outputs in this repo (not run; stated "
         "honestly). End-task baselines (LLM Direct/Oracle/etc.) are **out of scope** — this is a "
         "retrieval-only comparison.", "",
         macro_table([r for r in roster if r[1] != "ltr" or r[0] in ("L6", "L6@100")]), "",
         "### 16.1 Per-dataset Recall@1 (%) — all methods, 5-fold CV test standard", "",
         "_LTR = out-of-fold predictions (each query test once); baselines = same query "
         "universe. AVG = macro mean over the 6 datasets._", "",
         per_dataset_metric_table(c, "Recall@1"), "",
         "### 16.2 Per-dataset Recall@10 (%) — all methods, 5-fold CV test standard", "",
         per_dataset_metric_table(c, "Recall@10"), "",
         "## 17. L6 vs Cross-Encoder Method 7", "",
         "Method 7 = α-CE β=0.7 (CE `ce-joint-v3` MiniLM-L6 rerank + Stage-1=M4 fusion). Two "
         "depth-matched comparisons: **CE@100 vs L6@100** (RRF top-100) and **CE@500 vs L6@500** "
         "(RRF top-500 — CE@500 reruns CE inference on the deeper pool, no retraining). CE wins "
         "both at the top (R@1/R@10/nDCG@10, p=0); at R@100 L6@500 ≈ CE@500 (deep recall matched). "
         "Per-dataset they are complementary (L6 wins TheoremQA & MedCalcBench; CE wins "
         "champ/bigcodebench/logicbench).", "",
         ce_block, "",
         "See `results/comparisons/ltr_vs_ce_method7_5fold.{md,csv,json}` for per-dataset.", "",
         "## 18. Statistical Significance and Confidence Intervals", "",
         "Per-method CI95 (=1.96·std/√5) is in the §15 fold-stats (`±std`; CI95 in the CSV/JSON). "
         "Paired bootstrap over shared queries (10k resamples), two-sided p-value:", "",
         _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows), "",
         "## 19. Final Paper Claim Recommendation", "", claim, "",
         "```json", json.dumps(detail, indent=2), "```", "",
         f"Error analysis (OOF): L6 fixes A7 on {err_counts.get('L6_fixes_A7')} queries, breaks "
         f"{err_counts.get('L6_breaks_A7')}; CE>L6 on {err_counts.get('CE_wins_L6_loses')}, "
         f"L6>CE on {err_counts.get('L6_wins_CE_loses')} (see "
         "`results/analysis/ltr_cv_error_analysis.md`).", "",
         "_Decision rule applied per the validation plan §12; see `IMPLEMENTATION_NOTES.md` §23._", ""]

    section = "\n".join(S)
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 15. ")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + section + "\n")
    logger.info("appended §15-§19 to %s", REPORT)


def write_all(consolidated_path):
    from kmeans.qsc_ltr_runner import load_ext
    _ext, cfg = load_ext()
    c = json.loads(Path(consolidated_path).read_text())
    write_metrics(c)
    write_delta(c)
    write_ce_comparison(c)
    err = write_error_analysis(cfg, c)
    append_sections(cfg, c, err)
