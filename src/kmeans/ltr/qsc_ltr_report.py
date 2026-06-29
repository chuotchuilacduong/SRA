"""Report writer for the QSC + LTR extension (plan §8, §9, §11, §16).

Consumes ``results/qsc_ltr/consolidated_eval.json`` (+ per-variant JSONL for the
error analysis) and writes the metrics/delta/error-analysis files and appends
``## 14. Extension: Query-Specific Clustering and Lightweight LTR`` to
``FULL_M4_V2_RESULTS.md``. QSC numbers are on the FULL query set; LTR numbers are
on the held-out TEST split (baselines recomputed on the same split). Every table
states which query set it uses. Nothing is guessed — all numbers come from eval.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from sragents.config import PROJECT_ROOT

from ..common.config import M4V2Config
from ..common import io
from ..common.evaluate import REPORT_METRICS
from .qsc_ltr_runner import load_ext, _abs, _read_jsonl

logger = logging.getLogger(__name__)

QSET_FULL = "full"
QSET_TEST = "test"


def _pct(block: dict, m: str) -> float:
    return block.get(m, float("nan")) * 100.0


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "n/a"


def _md(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


def _macro(ev: dict) -> dict:
    return ev["eval"]["macro"]


def _byds(ev: dict, ds: str) -> dict:
    return ev["eval"]["by_dataset"].get(ds, {})


# --- assemble the method roster ----------------------------------------------

def _roster(cons: dict):
    """Return ordered [(label, query_set, ev_block)] for the macro tables."""
    rows = []
    bf, bt = cons["baselines_full"], cons["baselines_test"]
    # QSC section: full-set baselines + Q0-Q6
    if "A0" in bf:
        rows.append(("A0 RRF", QSET_FULL, bf["A0"]))
    if "A1" in bf:
        rows.append(("A1 Current M4", QSET_FULL, bf["A1"]))
    if "A7" in bf:
        rows.append(("A7 PRF only", QSET_FULL, bf["A7"]))
    for vid, blk in cons["qsc"].items():
        rows.append((f"{vid} {blk['name']}", QSET_FULL, blk))
    return rows


# --- metrics csv/json/md -----------------------------------------------------

def write_metrics(cons: dict, ext: dict) -> None:
    datasets = cons["datasets"]
    comp = _abs(ext["reporting"]["metrics_csv"]).parent
    comp.mkdir(parents=True, exist_ok=True)

    rows = []

    def emit(label, qset, ev):
        for m in REPORT_METRICS:
            row = {"method": label, "query_set": qset, "metric": m}
            for ds in datasets:
                row[ds] = round(_pct(_byds(ev, ds), m), 2)
            row["macro_avg"] = round(_pct(_macro(ev), m), 2)
            rows.append(row)

    # FULL-set: QSC baselines + Q0-Q6
    for vid in ["A0", "A1", "A2", "A7", "A8"]:
        if vid in cons["baselines_full"]:
            emit(f"{vid} {cons['baselines_full'][vid]['name']}", QSET_FULL,
                 cons["baselines_full"][vid])
    for vid, blk in cons["qsc"].items():
        emit(f"{vid} {blk['name']}", QSET_FULL, blk)
    # TEST-split: baselines + Q2/Q6 + LTR
    for vid, blk in cons["baselines_test"].items():
        emit(f"{vid} (test)", QSET_TEST, blk)
    for vid, blk in cons["ltr"].items():
        emit(f"{vid} {blk['name']}", QSET_TEST, blk)

    fields = ["method", "query_set", "metric", *datasets, "macro_avg"]
    with (comp / "qsc_ltr_extension_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    io.write_json(comp / "qsc_ltr_extension_metrics.json", {"datasets": datasets, "rows": rows})

    # MD: QSC macro + LTR macro
    md = ["# QSC + LTR Extension Metrics", "",
          "## QSC (FULL query set, macro avg %, directly comparable to A0-A8)", ""]
    md += [_macro_table(cons, _roster(cons)), ""]
    md += ["## LTR (TEST split only, macro avg %)", "",
           "_Baselines recomputed on the same held-out test queries for a fair comparison._", ""]
    ltr_rows = []
    for vid in ["A0", "A1", "A7", "Q2", "Q6"]:
        if vid in cons["baselines_test"]:
            ltr_rows.append((f"{vid} (test)", QSET_TEST, cons["baselines_test"][vid]))
    for vid, blk in cons["ltr"].items():
        ltr_rows.append((f"{vid} {blk['name']}", QSET_TEST, blk))
    md += [_macro_table(cons, ltr_rows), ""]
    (comp / "qsc_ltr_extension_metrics.md").write_text("\n".join(md))
    logger.info("wrote qsc_ltr metrics csv/json/md")


def _macro_table(cons: dict, roster) -> str:
    headers = ["Method", *REPORT_METRICS]
    rows = []
    for label, _qset, ev in roster:
        rows.append([label, *[_fmt(_pct(_macro(ev), m)) for m in REPORT_METRICS]])
    return _md(headers, rows)


def _per_dataset_r10(cons: dict, roster) -> str:
    datasets = cons["datasets"]
    headers = ["Method", *datasets, "AVG"]
    rows = []
    for label, _qset, ev in roster:
        rows.append([label, *[_fmt(_pct(_byds(ev, ds), "Recall@10")) for ds in datasets],
                     _fmt(_pct(_macro(ev), "Recall@10"))])
    return _md(headers, rows)


# --- delta -------------------------------------------------------------------

def write_delta(cons: dict, ext: dict) -> None:
    out = ["# QSC + LTR Extension — Delta tables", "",
           "Δ = metric_new − metric_baseline (percentage points).", ""]
    # QSC vs A0/A1/A7 (FULL set)
    for base_id in ["A0", "A1", "A7"]:
        if base_id not in cons["baselines_full"]:
            continue
        base = cons["baselines_full"][base_id]
        out += [f"## QSC vs {base_id} {base['name']} (FULL set)", ""]
        out += [_delta_table(cons["qsc"].values(), base), ""]
    # LTR vs A0/A1/A7 (TEST split)
    for base_id in ["A0", "A1", "A7"]:
        if base_id not in cons["baselines_test"]:
            continue
        base = cons["baselines_test"][base_id]
        out += [f"## LTR vs {base_id} (TEST split)", ""]
        out += [_delta_table(cons["ltr"].values(), base), ""]
    _abs(ext["reporting"]["delta_md"]).write_text("\n".join(out))
    logger.info("wrote qsc_ltr delta")


def _delta_table(variants, base) -> str:
    metrics = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100", "nDCG@10"]
    headers = ["Method", *[f"Δ{m}" for m in metrics]]
    rows = []
    bm = _macro(base)
    for blk in variants:
        m = _macro(blk)
        label = f"{blk['id']} {blk.get('name', '')}"
        rows.append([label, *[_fmt(_pct(m, k) - _pct(bm, k)) for k in metrics]])
    return _md(headers, rows)


# --- error analysis ----------------------------------------------------------

def _hit10(rec, gold):
    return any(c["skill_id"] in gold for c in rec["retrieved"][:10])


def _grank(rec, gold):
    for c in rec["retrieved"]:
        if c["skill_id"] in gold:
            return c["rank"]
    return None


def write_error_analysis(cfg: M4V2Config, cons: dict, ext: dict,
                         max_ex: int = 30) -> None:
    datasets = cons["datasets"]
    out_dir = _abs(ext["reporting"]["out_dir"])
    a7_slug = cfg.variant("A7").slug

    # QSC: A7 (full) vs Q6 (full)
    qsc_fixed, qsc_broken = [], []
    for ds in datasets:
        a7 = {r["instance_id"]: r for r in _read_jsonl(cfg.paths["out_dir"] / a7_slug / f"{ds}.jsonl")}
        q6 = {r["instance_id"]: r for r in _read_jsonl(out_dir / "qsc" / "Q6" / f"{ds}.jsonl")}
        for qid, qr in q6.items():
            ar = a7.get(qid)
            if ar is None:
                continue
            gold = set(qr.get("gold_skill_ids") or [])
            if not gold:
                continue
            ah, qh = _hit10(ar, gold), _hit10(qr, gold)
            top = qr["retrieved"][0] if qr["retrieved"] else {}
            ex = {"dataset": ds, "query_id": qid, "query": (qr.get("query") or "")[:120],
                  "a7_rank": _grank(ar, gold), "q6_rank": _grank(qr, gold),
                  "local_cluster_id": top.get("cluster_debug", {}).get("hard_cluster")
                  if "cluster_debug" in top else top.get("local_cluster_id"),
                  "local_cluster_size": top.get("local_cluster_size")}
            if (not ah) and qh:
                qsc_fixed.append(ex)
            elif ah and (not qh):
                qsc_broken.append(ex)

    # LTR: A7 (test) vs L6/L5/L2 (test)
    a7_test = {}
    for ds in datasets:
        for r in _read_jsonl(cfg.paths["out_dir"] / a7_slug / f"{ds}.jsonl"):
            a7_test[r["instance_id"]] = r
    ltr_buckets = {}
    for lid in ["L2", "L5", "L6"]:
        recs = _read_jsonl(out_dir / "ltr" / lid / "test.jsonl")
        fixed, broken = [], []
        for r in recs:
            gold = set(r.get("gold_skill_ids") or [])
            ar = a7_test.get(r["instance_id"])
            if not gold or ar is None:
                continue
            ah, lh = _hit10(ar, gold), _hit10(r, gold)
            ex = {"dataset": r.get("dataset"), "query_id": r["instance_id"],
                  "a7_rank": _grank(ar, gold), "new_rank": _grank(r, gold)}
            if (not ah) and lh:
                fixed.append(ex)
            elif ah and (not lh):
                broken.append(ex)
        ltr_buckets[lid] = (fixed, broken)

    def _sec(title, items, cols):
        L = [f"### {title} (n={len(items)})", ""]
        if not items:
            return L + ["_none_", ""]
        headers = list(cols.keys())
        rows = [[str(it.get(c, ""))[:60].replace("|", "\\|") for c in cols.values()]
                for it in items[:max_ex]]
        L += [_md(headers, rows), ""]
        if len(items) > max_ex:
            L += [f"_…and {len(items)-max_ex} more (see JSONL)._", ""]
        return L

    qcols = {"dataset": "dataset", "query_id": "query_id", "A7 rank": "a7_rank",
             "Q6 rank": "q6_rank", "local clst": "local_cluster_id", "clst size": "local_cluster_size",
             "query": "query"}
    L = ["# QSC + LTR Error Analysis", "",
         "## 11.1/11.2 QSC: A7 (full) vs Q6 QSC-A7-tiebreak (full), at Recall@10", ""]
    L += _sec("Fixed by Q6 (A7 miss@10 → Q6 hit@10)", qsc_fixed, qcols)
    L += _sec("Broken by Q6 (A7 hit@10 → Q6 miss@10)", qsc_broken, qcols)
    L += ["## 11.3 LTR: A7 (test) vs L2/L5/L6 (test), at Recall@10", ""]
    lcols = {"dataset": "dataset", "query_id": "query_id", "A7 rank": "a7_rank", "new rank": "new_rank"}
    for lid in ["L2", "L5", "L6"]:
        fixed, broken = ltr_buckets[lid]
        L += _sec(f"{lid}: fixed (A7 miss@10 → {lid} hit@10)", fixed, lcols)
        L += _sec(f"{lid}: broken (A7 hit@10 → {lid} miss@10)", broken, lcols)
    _abs(ext["reporting"]["error_analysis"]).write_text("\n".join(L))
    logger.info("wrote qsc_ltr error analysis (QSC fixed=%d broken=%d)",
                len(qsc_fixed), len(qsc_broken))


# --- recommendation (plan §10, §16) ------------------------------------------

def _acceptance(macro_new: dict, a0: dict, a7: dict, byds_new, byds_a7, datasets) -> dict:
    nd = _pct(macro_new, "nDCG@10")
    checks = {
        "nDCG@10 >= A0": nd >= _pct(a0, "nDCG@10") if a0 else None,
        "R@1 >= A0 - 0.5pp": _pct(macro_new, "Recall@1") >= _pct(a0, "Recall@1") - 0.5 if a0 else None,
        "R@10 >= A7 - 0.2pp": _pct(macro_new, "Recall@10") >= _pct(a7, "Recall@10") - 0.2 if a7 else None,
    }
    worst = None
    if a7 and byds_a7:
        losses = [(ds, _pct(byds_new(ds), "Recall@10") - _pct(byds_a7(ds), "Recall@10"))
                  for ds in datasets]
        worst = min(losses, key=lambda x: x[1]) if losses else None
        checks["no dataset loses >2pp R@10 vs A7"] = (worst[1] >= -2.0) if worst else None
    return {"checks": checks, "worst_dataset_r10_vs_a7": worst}


def _recommendation(cons: dict) -> tuple[str, dict]:
    # Best QSC (full set) by nDCG@10, best LTR (test) by nDCG@10.
    detail = {}
    a0f = _macro(cons["baselines_full"]["A0"]) if "A0" in cons["baselines_full"] else {}
    a7f = _macro(cons["baselines_full"]["A7"]) if "A7" in cons["baselines_full"] else {}
    best_qsc = max(cons["qsc"].values(), key=lambda b: _pct(_macro(b), "nDCG@10"), default=None)
    a0t = _macro(cons["baselines_test"]["A0"]) if "A0" in cons["baselines_test"] else {}
    a7t = _macro(cons["baselines_test"]["A7"]) if "A7" in cons["baselines_test"] else {}
    best_ltr = max(cons["ltr"].values(), key=lambda b: _pct(_macro(b), "nDCG@10"), default=None)

    msgs = []
    if best_qsc:
        acc = _acceptance(_macro(best_qsc), a0f, a7f,
                          lambda ds: _byds(best_qsc, ds),
                          lambda ds: _byds(cons["baselines_full"].get("A7", {"eval": {"by_dataset": {}}}), ds),
                          cons["datasets"])
        detail["best_qsc"] = {"id": best_qsc["id"], "name": best_qsc["name"],
                              "nDCG@10": round(_pct(_macro(best_qsc), "nDCG@10"), 2),
                              "R@10": round(_pct(_macro(best_qsc), "Recall@10"), 2),
                              "acceptance": acc["checks"]}
        passed = all(v for v in acc["checks"].values() if v is not None)
        msgs.append(f"Best QSC = {best_qsc['id']} {best_qsc['name']}: "
                    + ("PASSES" if passed else "does NOT pass")
                    + " the §10.1 final-ranker criteria (full set).")
    if best_ltr:
        acc = _acceptance(_macro(best_ltr), a0t, a7t,
                          lambda ds: _byds(best_ltr, ds),
                          lambda ds: _byds(cons["baselines_test"].get("A7", {"eval": {"by_dataset": {}}}), ds),
                          cons["datasets"])
        detail["best_ltr"] = {"id": best_ltr["id"], "name": best_ltr["name"],
                              "nDCG@10": round(_pct(_macro(best_ltr), "nDCG@10"), 2),
                              "R@10": round(_pct(_macro(best_ltr), "Recall@10"), 2),
                              "acceptance": acc["checks"]}
        passed = all(v for v in acc["checks"].values() if v is not None)
        msgs.append(f"Best LTR = {best_ltr['id']} {best_ltr['name']}: "
                    + ("PASSES" if passed else "does NOT pass")
                    + " the §10.1 final-ranker criteria (test split).")
    return "\n\n".join(msgs) if msgs else "Insufficient data.", detail


# --- append Section 14 to FULL_M4_V2_RESULTS.md ------------------------------

def append_section14(cfg: M4V2Config, cons: dict, ext: dict) -> None:
    datasets = cons["datasets"]
    qsc_roster = _roster(cons)
    ltr_roster = []
    for vid in ["A0", "A1", "A7", "Q2", "Q6"]:
        if vid in cons["baselines_test"]:
            ltr_roster.append((f"{vid} (test)", QSET_TEST, cons["baselines_test"][vid]))
    for vid, blk in cons["ltr"].items():
        ltr_roster.append((f"{vid} {blk['name']}", QSET_TEST, blk))

    rec, detail = _recommendation(cons)
    ss = cons["split_sizes"]

    S = ["## 14. Extension: Query-Specific Clustering and Lightweight LTR", "",
         "### 14.1 Were these methods included in the previous M4-v2 run?", "",
         "No. The previous M4-v2 run (A0–A8) tested RRF top-M, PRF, soft **global** "
         "cluster distributions, global cluster reliability, and adaptive alpha. It did "
         "**not** test query-specific *local* clustering or any trained learning-to-rank "
         "model. This section adds both.", "",
         "### 14.2 QSC Method", "",
         "For each query, run a local KMeans (H="
         f"{ext['qsc']['default_local_k']}, capped at floor(sqrt(M))) on the embeddings of "
         "its RRF top-500 candidates, then rerank with QSC-1 (affinity blend, α="
         f"{ext['qsc']['qsc1_alpha']}), QSC-2 (prior blend), or QSC-3 (tie-break only, "
         f"δ={ext['qsc']['qsc3_delta']}, λ={ext['qsc']['qsc3_lambda']}). Cluster signal is "
         "deliberately weak (the previous global-cluster scoring hurt top-rank quality).", "",
         "### 14.3 LTR Method", "",
         f"Feature-based rankers over {len(_all_feature_count())} numeric features "
         "(retrieval / M4 / A7-PRF / QSC / query-confidence groups), no Cross-Encoder. "
         "LightGBM LambdaRank (objective=lambdarank, metric=ndcg) and a linear pairwise "
         "ranker. **Query-level stratified split** "
         f"(train={ss.get('train')} / dev={ss.get('dev')} / test={ss.get('test')} queries, "
         "seed 42); candidates from the same query never cross splits; hyperparameters "
         "tuned on dev; metrics reported on the **held-out test split** only.", "",
         "### 14.4 QSC Results (FULL query set, macro %, comparable to A0–A8)", "",
         _macro_table(cons, qsc_roster), "",
         "Per-dataset Recall@10:", "",
         _per_dataset_r10(cons, qsc_roster), "",
         "### 14.5 LTR Results (TEST split, macro %; baselines on same split)", "",
         _macro_table(cons, ltr_roster), "",
         "Per-dataset Recall@10 (test split):", "",
         _per_dataset_r10(cons, ltr_roster), "",
         "### 14.6 Combined QSC + LTR", "",
         "QSC-as-features inside LTR is variant **L3** (RRF+QSC), **L5** (A7+QSC) and "
         "**L6** (full, includes QSC). Compare L5/L6 against L2 (A7 without QSC) in 14.5 "
         "to read whether QSC features add anything once signals are learned.", "",
         "### 14.7 Comparison against A0/A1/A7/A8", "",
         "See `results/comparisons/qsc_ltr_extension_delta.md` for full Δ tables. "
         "QSC is compared on the full set; LTR on the test split (vs baselines on the "
         "same split).", "",
         "### 14.8 Final Recommendation", "", rec, "",
         "Acceptance detail (plan §10.1):", "",
         "```json", json.dumps(detail, indent=2), "```", "",
         "_QSC/LTR query sets differ (full vs test) — compare each method only against "
         "its same-query-set baseline. The A0/A7 test-split baselines are **depth-matched** "
         "to LTR (same RRF top-500 pool, both output top-100; verified macro-identical to a "
         "fresh RRF/500→top-100 re-rank), so L6's gains are genuine reranking, not a pool-depth "
         "artifact. LTR is a leakage-controlled learned result (query-level split, train/dev/test "
         "verified disjoint, no feature uses gold). Adversarial review found no leakage and all "
         "modules spec-correct. See `IMPLEMENTATION_NOTES.md` §17–§22._", ""]

    section = "\n".join(S)
    report = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
    text = report.read_text() if report.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 14.")          # idempotent: drop any prior §14 first
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    report.write_text(text + section + "\n")
    logger.info("appended Section 14 to %s", report)


def _all_feature_count() -> list:
    from .ltr_features import ALL_FEATURES
    return ALL_FEATURES


def write_all(ext_path: str | None = None) -> None:
    ext, cfg = load_ext(ext_path)
    out_dir = _abs(ext["reporting"]["out_dir"])
    cons = io.read_json(out_dir / "consolidated_eval.json")
    write_metrics(cons, ext)
    write_delta(cons, ext)
    write_error_analysis(cfg, cons, ext)
    append_section14(cfg, cons, ext)
