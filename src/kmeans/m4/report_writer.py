"""Report generation (spec §10, §12, §17).

Consumes ``results/m4_v2/ablation/_all_eval.json`` (+ the per-variant JSONL for
error analysis) and writes:
  results/comparisons/m4_v2_ablation_metrics.{csv,json,md}
  results/comparisons/m4_v2_main_comparison.md
  results/comparisons/m4_v2_delta.md
  results/analysis/m4_v2_error_analysis.md
  FULL_M4_V2_RESULTS.md     (repo root)

All numbers come from the evaluation; nothing is guessed (spec §18). Acceptance
logic per §11 / §17.1.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from sragents.config import PROJECT_ROOT

from ..common.config import M4V2Config
from ..common import io
from ..common.evaluate import REPORT_METRICS

logger = logging.getLogger(__name__)

ABLATION_COLS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100", "nDCG@10"]


def _pct(ev_block: dict, metric: str) -> float:
    return ev_block.get(metric, float("nan")) * 100.0


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "n/a"


# --- ablation metrics csv/json/md --------------------------------------------

def write_ablation_metrics(cfg: M4V2Config, all_eval: dict, datasets: list[str]) -> None:
    comp = cfg.paths["comparisons_dir"]
    rows = []
    for vid, blk in all_eval.items():
        ev = blk["eval"]
        for metric in REPORT_METRICS:
            row = {"variant": vid, "variant_name": blk["name"], "metric": metric}
            for ds in datasets:
                row[ds] = round(_pct(ev["by_dataset"].get(ds, {}), metric), 2)
            row["macro_avg"] = round(_pct(ev["macro"], metric), 2)
            row["micro_avg"] = round(_pct(ev["micro"], metric), 2)
            rows.append(row)

    # CSV
    csv_path = comp / "m4_v2_ablation_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "variant_name", "metric", *datasets, "macro_avg", "micro_avg"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # JSON
    io.write_json(comp / "m4_v2_ablation_metrics.json",
                  {"datasets": datasets, "rows": rows})
    # MD (macro, the ablation table per §10.2)
    headers = ["Variant", "Name", "Pool", "PRF", "Soft", "Calib", "Adaptive α", *ABLATION_COLS]
    md_rows = []
    for vid, blk in all_eval.items():
        macro = blk["eval"]["macro"]
        soft = "yes" if blk["affinity"] in ("soft", "soft_calibrated") else "no"
        calib = "yes" if blk["affinity"] in ("hard_calibrated", "soft_calibrated") else "no"
        adapt = "yes" if blk["alpha"] == "adaptive" else ("fixed" if blk["alpha"] == "fixed" else "no")
        md_rows.append([vid, blk["name"], blk["pool"], "yes" if blk["prf"] else "no",
                        soft, calib, adapt, *[_fmt(_pct(macro, m)) for m in ABLATION_COLS]])
    md = ["# M4-v2 Ablation (macro avg over datasets, %)", "",
          "All variants evaluated on the same queries/corpus with the reused "
          "`sragents` metrics. Numbers are macro (equal-weight over datasets).", "",
          _md_table(headers, md_rows), "",
          "### Isolated component effects (macro, pp)", ""]
    md += _component_effects(all_eval)
    (comp / "m4_v2_ablation.md").write_text("\n".join(md))
    logger.info("wrote ablation metrics csv/json/md")


def _delta_row(all_eval, a, b, metric):
    if a not in all_eval or b not in all_eval:
        return None
    return _pct(all_eval[a]["eval"]["macro"], metric) - _pct(all_eval[b]["eval"]["macro"], metric)


def _component_effects(all_eval: dict) -> list[str]:
    pairs = [("A2", "A1", "larger pool"), ("A3", "A2", "reliability calib (hard)"),
             ("A4", "A2", "soft assignment"), ("A5", "A4", "reliability on soft"),
             ("A6", "A5", "adaptive alpha"), ("A8", "A6", "PRF + all")]
    headers = ["Effect", "Comparison", "ΔR@10", "ΔnDCG@10", "ΔR@1", "ΔR@100"]
    rows = []
    for a, b, label in pairs:
        if a in all_eval and b in all_eval:
            rows.append([label, f"{a}−{b}",
                         _fmt(_delta_row(all_eval, a, b, "Recall@10")),
                         _fmt(_delta_row(all_eval, a, b, "nDCG@10")),
                         _fmt(_delta_row(all_eval, a, b, "Recall@1")),
                         _fmt(_delta_row(all_eval, a, b, "Recall@100"))])
    return [_md_table(headers, rows)]


# --- main comparison ----------------------------------------------------------

def write_main_comparison(cfg: M4V2Config, all_eval: dict, datasets: list[str]) -> None:
    """Per-metric tables: rows = key methods, cols = datasets + AVG (§10.1)."""
    key = [("A0", "RRF baseline"), ("A1", "Current M4 (α=0.7)"), ("A8", "Full M4-v2")]
    key = [(vid, name) for vid, name in key if vid in all_eval]
    out = ["# M4-v2 Main Comparison", "",
           "Per-dataset Recall/nDCG (%). RRF and Current M4 are recomputed here "
           "(A0/A1) with the same metrics, so they are directly comparable.", ""]
    for metric in REPORT_METRICS:
        out += [f"## {metric}", ""]
        headers = ["Method", *datasets, "AVG"]
        rows = []
        for vid, name in key:
            ev = all_eval[vid]["eval"]
            rows.append([f"{vid} {name}",
                         *[_fmt(_pct(ev["by_dataset"].get(ds, {}), metric)) for ds in datasets],
                         _fmt(_pct(ev["macro"], metric))])
        out += [_md_table(headers, rows), ""]
    (cfg.paths["comparisons_dir"] / "m4_v2_main_comparison.md").write_text("\n".join(out))
    logger.info("wrote main comparison")


# --- delta --------------------------------------------------------------------

def _load_method7_macro() -> dict | None:
    """Best-effort: pull Method 7 (α-CE β=0.7) macro numbers from the existing
    paper comparison CSV if present (R@1,R@10,nDCG@1,nDCG@10 only)."""
    p = PROJECT_ROOT / "results" / "comparisons" / "paper_compare_full_bench.csv"
    if not p.exists():
        return None
    try:
        out: dict[str, float] = {}
        with p.open() as f:
            for row in csv.DictReader(f):
                meth = (row.get("method") or "").lower()
                if "β=0.7" in meth or "b=0.7" in meth or "method 7" in meth or "α-ce" in meth or "a-ce" in meth:
                    metric = row.get("metric", "")
                    val = row.get("macro_avg") or row.get("AVG") or row.get("avg")
                    if metric and val:
                        try:
                            out[metric] = float(val)
                        except ValueError:
                            pass
        return out or None
    except Exception:
        return None


def write_delta(cfg: M4V2Config, all_eval: dict) -> None:
    out = ["# M4-v2 Delta (macro, percentage points)", "",
           "Δ = metric_new − metric_baseline.", ""]
    for new, base, label in [("A8", "A1", "A8 Full M4-v2 vs A1 Current M4"),
                             ("A8", "A0", "A8 Full M4-v2 vs A0 RRF")]:
        if new not in all_eval or base not in all_eval:
            continue
        out += [f"## {label}", ""]
        headers = ["Metric", f"{new}", f"{base}", "Δ (pp)"]
        rows = []
        for metric in REPORT_METRICS:
            nv = _pct(all_eval[new]["eval"]["macro"], metric)
            bv = _pct(all_eval[base]["eval"]["macro"], metric)
            rows.append([metric, _fmt(nv), _fmt(bv), _fmt(nv - bv)])
        out += [_md_table(headers, rows), ""]

    m7 = _load_method7_macro()
    out += ["## A8 Full M4-v2 vs Method 7 (α-CE β=0.7)", ""]
    if m7:
        headers = ["Metric", "A8", "Method 7", "Δ (pp)"]
        rows = []
        for metric in ["Recall@1", "Recall@10", "nDCG@1", "nDCG@10"]:
            if metric in m7 and "A8" in all_eval:
                nv = _pct(all_eval["A8"]["eval"]["macro"], metric)
                rows.append([metric, _fmt(nv), _fmt(m7[metric]), _fmt(nv - m7[metric])])
        out += [_md_table(headers, rows), "",
                "_Method 7 numbers from `results/comparisons/paper_compare_full_bench.csv`._"]
    else:
        out += ["_Method 7 baseline not found in `results/comparisons/`; skipped. "
                "(Note: Method 7 uses a Cross-Encoder; not a like-for-like comparison.)_"]
    (cfg.paths["comparisons_dir"] / "m4_v2_delta.md").write_text("\n".join(out))
    logger.info("wrote delta")


# --- error analysis -----------------------------------------------------------

def _hit_at(retrieved: list[dict], gold: set[str], k: int) -> bool:
    return any(c["skill_id"] in gold for c in retrieved[:k])


def _gold_rank(retrieved: list[dict], gold: set[str]) -> int | None:
    for c in retrieved:
        if c["skill_id"] in gold:
            return c["rank"]
    return None


def write_error_analysis(cfg: M4V2Config, datasets: list[str],
                         new_id: str = "A8", base_id: str = "A1",
                         max_examples: int = 40) -> None:
    new_v = cfg.variant(new_id)
    base_v = cfg.variant(base_id)
    fixed, broken, top1_reg = [], [], []
    for ds in datasets:
        new_recs = {r["instance_id"]: r for r in
                    _read_jsonl(cfg.paths["out_dir"] / new_v.slug / f"{ds}.jsonl")}
        base_recs = {r["instance_id"]: r for r in
                     _read_jsonl(cfg.paths["out_dir"] / base_v.slug / f"{ds}.jsonl")}
        for qid, nr in new_recs.items():
            br = base_recs.get(qid)
            if br is None:
                continue
            gold = set(nr.get("gold_skill_ids") or [])
            if not gold:
                continue
            nb_hit = _hit_at(br["retrieved"], gold, 10)
            nn_hit = _hit_at(nr["retrieved"], gold, 10)
            base_rank = _gold_rank(br["retrieved"], gold)
            new_rank = _gold_rank(nr["retrieved"], gold)
            ex = {"dataset": ds, "query_id": qid,
                  "query": (nr.get("query") or "")[:160],
                  "old_rank": base_rank, "new_rank": new_rank,
                  "alpha": nr.get("alpha"),
                  "top_query_clusters": nr.get("top_query_clusters")}
            if (not nb_hit) and nn_hit:
                fixed.append(ex)
            elif nb_hit and (not nn_hit):
                broken.append(ex)
            if base_rank == 1 and (new_rank is None or new_rank > 1):
                top1_reg.append(ex)

    def _section(title, items, note=""):
        lines = [f"## {title}  (n={len(items)})", ""]
        if note:
            lines += [note, ""]
        if not items:
            lines += ["_none_", ""]
            return lines
        headers = ["dataset", "query_id", "old→new rank", "α", "query"]
        rows = [[e["dataset"], e["query_id"],
                 f"{e['old_rank']}→{e['new_rank']}", _fmt(e["alpha"]) if e["alpha"] else "n/a",
                 e["query"].replace("|", "\\|")] for e in items[:max_examples]]
        lines += [_md_table(headers, rows), ""]
        if len(items) > max_examples:
            lines += [f"_…and {len(items) - max_examples} more (see JSONL outputs)._", ""]
        return lines

    out = [f"# M4-v2 Error Analysis ({new_id} vs {base_id})", "",
           f"Buckets compare **{new_id} {new_v.name}** against **{base_id} {base_v.name}** "
           "at Recall@10 (gold present in top-10).", ""]
    out += _section(f"Fixed by {new_id} (miss@10 in {base_id} → hit@10 in {new_id})", fixed)
    out += _section(f"Broken by {new_id} (hit@10 in {base_id} → miss@10 in {new_id})", broken,
                    "Investigate: PRF topic drift / soft-cluster overboost / "
                    "calibration too strong / α too low / large-pool noise.")
    out += _section(f"Top-1 regressions (gold was rank 1 in {base_id}, no longer in {new_id})",
                    top1_reg)
    (cfg.paths["analysis_dir"] / "m4_v2_error_analysis.md").write_text("\n".join(out))
    logger.info("wrote error analysis (fixed=%d broken=%d top1_reg=%d)",
                len(fixed), len(broken), len(top1_reg))
    return {"fixed": len(fixed), "broken": len(broken), "top1_reg": len(top1_reg)}


def _read_jsonl(path: Path) -> list[dict]:
    import json
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


# --- full report + recommendation --------------------------------------------

def _recommendation(all_eval: dict) -> tuple[str, dict]:
    """Apply spec §11/§17.1 decision rule using macro metrics."""
    if "A8" not in all_eval or "A1" not in all_eval:
        return "Insufficient data (need A1 and A8).", {}
    a8 = all_eval["A8"]["eval"]["macro"]
    a1 = all_eval["A1"]["eval"]["macro"]
    a0 = all_eval.get("A0", {}).get("eval", {}).get("macro", {})

    d_r10 = _pct(a8, "Recall@10") - _pct(a1, "Recall@10")
    d_ndcg10 = _pct(a8, "nDCG@10") - _pct(a1, "nDCG@10")
    d_r100 = _pct(a8, "Recall@100") - _pct(a1, "Recall@100")
    r1_drop_vs_rrf = (_pct(a0, "Recall@1") - _pct(a8, "Recall@1")) if a0 else float("nan")

    checks = {
        "R@10 improves vs M4": d_r10 > 0,
        "nDCG@10 improves vs M4": d_ndcg10 > 0,
        "R@1 drop vs RRF ≤ 0.5pp": (r1_drop_vs_rrf <= 0.5) if r1_drop_vs_rrf == r1_drop_vs_rrf else None,
        "R@100 improves vs M4": d_r100 > 0,
        "deltas": {"R@10": round(d_r10, 2), "nDCG@10": round(d_ndcg10, 2),
                   "R@100": round(d_r100, 2), "R@1_drop_vs_RRF": round(r1_drop_vs_rrf, 2)
                   if r1_drop_vs_rrf == r1_drop_vs_rrf else None},
    }
    improves = d_r10 > 0 and d_ndcg10 > 0
    safe = (r1_drop_vs_rrf != r1_drop_vs_rrf) or (r1_drop_vs_rrf <= 0.5)
    if improves and safe:
        rec = ("**Recommend M4-v2 as a replacement for current M4.** It improves "
               "R@10 and nDCG@10 without meaningfully hurting R@1.")
    elif improves and not safe:
        rec = ("**Recommend M4-v2 only for recall-oriented candidate generation, "
               "not final top-1 ranking** — it improves R@10 but R@1 drops "
               f"{r1_drop_vs_rrf:.2f}pp below RRF (> 0.5pp threshold).")
    else:
        rec = ("**Keep current M4.** M4-v2 does not improve R@10/nDCG@10 over it. "
               "See the ablation table for which component degraded.")
    return rec, checks


def write_full_report(cfg: M4V2Config, all_eval: dict, datasets: list[str],
                      run_summary: dict, err_counts: dict | None = None) -> None:
    rec, checks = _recommendation(all_eval)
    L = ["# FULL_M4_V2_RESULTS.md", "",
         "## 1. Objective",
         "Improve Method 4 (RRF + KMeans rerank) into **M4-v2**: a larger-pool, "
         "soft-cluster, reliability-calibrated, adaptive-weight RRF reranker, and "
         "decide via ablation whether it should replace current M4.", "",
         "## 2. Existing Baselines",
         "A0 = RRF (BM25+BGE, k=60); A1 = current Method 4 (α=0.7, 0.7·RRF+0.3·aff). "
         "Both recomputed here with the published metrics for a like-for-like comparison.", "",
         "## 3. M4-v2 Method",
         "Steps: RRF top-M pool → optional safe PRF query refinement → soft top-L "
         "cluster distributions for query & skill → cluster-reliability calibration "
         "→ adaptive α per query → blend `α·RRF_norm + (1-α)·Aff_norm` → top-100.", "",
         "## 4. Mathematical Formulation",
         "See `docs/kmeans/implementation_m4_v2 (1).md` §2-§10. Affinity:",
         "`Aff_soft-cal(q,s) = Σ_k p(k|q) p(k|s) rel(k)`, "
         "`rel(k) = clamp(minmax(coh(k)·log(N/(|C_k|+1))), 0.05, 1.0)`.", "",
         "## 5. Implementation Details",
         "Code in `src/kmeans/`; reuses `sragents` metrics/corpus/schema. "
         "Query embeddings encoded with `BAAI/bge-base-en-v1.5` + search prefix and "
         "cached. M=500 pool built by extending the cached BGE depth via the "
         "corpus-embedding matmul + cached BM25 top-50, RRF-fused. See "
         "`src/kmeans/IMPLEMENTATION_NOTES.md` for every decision/deviation.", "",
         "## 6. Ablation Results",
         "See `results/comparisons/m4_v2_ablation.md` (macro table + isolated component effects).", ""]
    # inline the ablation macro table
    headers = ["Variant", "Pool", *ABLATION_COLS]
    rows = []
    for vid, blk in all_eval.items():
        macro = blk["eval"]["macro"]
        rows.append([f"{vid} {blk['name']}", blk["pool"],
                     *[_fmt(_pct(macro, m)) for m in ABLATION_COLS]])
    L += [_md_table(headers, rows), ""]
    L += ["## 7. Main Results",
          "See `results/comparisons/m4_v2_main_comparison.md` (per-dataset, per-metric).", "",
          "## 8. Per-Dataset Analysis",
          "Per-dataset numbers for every variant are in "
          "`results/comparisons/m4_v2_ablation_metrics.csv`.", ""]
    # wins/losses A8 vs A1 per dataset on R@10 (§11.4)
    if "A8" in all_eval and "A1" in all_eval:
        L += ["### A8 vs A1 per-dataset (R@10, pp; win>+0.1 / tie / loss<-0.1)", ""]
        wl_headers = ["Dataset", "A1 R@10", "A8 R@10", "Δ", "verdict"]
        wl_rows = []
        for ds in datasets:
            a1 = _pct(all_eval["A1"]["eval"]["by_dataset"].get(ds, {}), "Recall@10")
            a8 = _pct(all_eval["A8"]["eval"]["by_dataset"].get(ds, {}), "Recall@10")
            d = a8 - a1
            verdict = "win" if d > 0.1 else ("loss" if d < -0.1 else "tie")
            wl_rows.append([ds, _fmt(a1), _fmt(a8), _fmt(d), verdict])
        L += [_md_table(wl_headers, wl_rows), ""]
    L += ["## 9. Error Analysis",
          "See `results/analysis/m4_v2_error_analysis.md`."]
    if err_counts:
        L += [f"Fixed by A8: {err_counts.get('fixed')}, broken by A8: "
              f"{err_counts.get('broken')}, top-1 regressions: {err_counts.get('top1_reg')}."]
    L += ["", "## 10. Latency and Resource Usage",
          f"Total wall time: {run_summary.get('total_wall_sec')}s on macOS/MPS "
          "(query encoding once; everything else is matmuls/sorts on CPU). "
          "Per-run timings in `logs/m4_v2/run_summary.json`.", "",
          "## 11. Final Recommendation", "", rec, "",
          "Acceptance checks (macro):", "",
          _md_table(["Check", "Pass"],
                    [[k, ("✅" if v else "❌") if isinstance(v, bool) else "n/a"]
                     for k, v in checks.items() if k != "deltas"]),
          "", f"Key deltas (pp): {checks.get('deltas')}", "",
          "## 12. Limitations",
          "- Tuned-on-test risk: no separate dev tuning was performed; defaults follow the spec. "
          "Sweep results (if run) must not be reported as unbiased test numbers (spec §8/§18).",
          "- M=500 pool extends BGE depth via the corpus matmul and keeps cached BM25 top-50; "
          "BM25 beyond rank 50 is treated as absent (contribution 0) — a documented approximation.",
          "- Method 7 (Cross-Encoder) is not a like-for-like baseline; any comparison is contextual only.", "",
          "## 13. Files Produced",
          "`results/m4_v2/{variant}/{dataset}.jsonl`, "
          "`results/comparisons/m4_v2_ablation_metrics.{csv,md,json}`, "
          "`results/comparisons/m4_v2_main_comparison.md`, "
          "`results/comparisons/m4_v2_delta.md`, "
          "`results/analysis/m4_v2_error_analysis.md`, "
          "`logs/m4_v2/run_summary.json`, this file.", ""]
    (PROJECT_ROOT / "FULL_M4_V2_RESULTS.md").write_text("\n".join(L))
    logger.info("wrote FULL_M4_V2_RESULTS.md")


def write_all(cfg: M4V2Config) -> None:
    data = io.read_json(cfg.paths["ablation_dir"] / "_all_eval.json")
    datasets = data["datasets"]
    all_eval = data["variants"]
    run_summary = {}
    rs_path = cfg.paths["logs_dir"] / "run_summary.json"
    if rs_path.exists():
        run_summary = io.read_json(rs_path)
    write_ablation_metrics(cfg, all_eval, datasets)
    write_main_comparison(cfg, all_eval, datasets)
    write_delta(cfg, all_eval)
    err = None
    if "A8" in all_eval and "A1" in all_eval:
        err = write_error_analysis(cfg, datasets)
    write_full_report(cfg, all_eval, datasets, run_summary, err)
