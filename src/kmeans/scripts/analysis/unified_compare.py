"""§22 Unified fair comparison — ALL methods + CE-Raw on query_gen-test.

Merges the two already-computed result files (no GPU, no re-eval) into one fair table:

  * results/comparisons/fair_supervised_query_gen.json  (§20: L6-final, M5/M7-CE@100/@500,
    BM25/BGE/RRF, M4/A7/QSC) — every row evaluated on query_gen-test.
  * results/comparisons/ceraw_query_gen.json            (§21: CE-Raw·bge_base/rrf/bge_ft) —
    same query_gen-test, same metric aggregation (evaluate.eval_variant).

Because BOTH files score the SAME held-out query_gen-test (1,079) with the SAME aggregation
and use final-fit / held-out supervised models, the merged tables are a fair, leakage-free,
same-test comparison across every method. Writes §22 (macro + the four per-dataset tables
the request asked for: Recall@1, Recall@10, nDCG@1, nDCG@10) to FULL_M4_V2_RESULTS.md.

    python src/kmeans/scripts/unified_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from sragents.config import PROJECT_ROOT
from kmeans.qsc_ltr_runner import load_ext

COMP = PROJECT_ROOT / "results" / "comparisons"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
FAIR = COMP / "fair_supervised_query_gen.json"
CERAW = COMP / "ceraw_query_gen.json"

SHORT_METRICS = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "Recall@100",
                 "nDCG@1", "nDCG@5", "nDCG@10"]
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]
# the four per-dataset tables requested
PER_DS = [("Recall@1", "Recall@1"), ("Recall@10", "Recall@10"),
          ("nDCG@1", "nDCG@1"), ("nDCG@10", "nDCG@10")]


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else (str(v) if v is not None else "—")


def main() -> None:
    if not FAIR.exists():
        raise SystemExit(f"missing {FAIR} — run run_fair_eval.py (§20) first")
    if not CERAW.exists():
        raise SystemExit(f"missing {CERAW} — run ceraw_eval.py (§21) first")

    ext, _cfg = load_ext()
    datasets = ext["datasets"]  # theoremqa, logicbench, toolqa, champ, medcalcbench, bigcodebench

    fair = json.loads(FAIR.read_text())
    ceraw = json.loads(CERAW.read_text())
    depth = ceraw.get("rerank_depth", "?")

    # --- assemble the unified row set (all on query_gen-test) ----------------
    # §20 methods, kept in a category order; then the standalone CE-Raw rerankers.
    order = ["BM25", "BGE", "RRF(BM25+BGE)",
             "A1 M4", "A7 PRF", "Q6 QSC", "A0 RRF", "Q2 QSC",
             "L6-final (LTR)",
             "M5-CE@100", "M7-CE@100", "M5-CE@500", "M7-CE@500"]
    fair_by = {r["method"]: r for r in fair["rows"]}
    rows = [fair_by[m] for m in order if m in fair_by]
    # any §20 method not in the explicit order (future-proof)
    rows += [r for r in fair["rows"] if r["method"] not in {x["method"] for x in rows}]
    # CE-Raw standalone rerankers (skip retriever-only rows; those are §21 ceilings)
    ceraw_std = [r for r in ceraw["rows"] if r.get("type") == "standalone-CE"]
    rows += ceraw_std

    n_fair = len(fair["rows"])
    sizes = fair.get("sizes", {})
    print(f"[unified] {len(rows)} methods ({n_fair} from §20 + {len(ceraw_std)} CE-Raw@{depth}) "
          f"| test={sizes.get('test','?')}")

    # --- tables --------------------------------------------------------------
    macro = _md(["Method", "Type", *SHORT],
                [[r["method"], r.get("type", ""), *[_fmt(r.get(m)) for m in SHORT_METRICS]] for r in rows])

    def per_ds(metric):
        return _md(["Method", *datasets, "AVG"],
                   [[r["method"], *[_fmt(r.get(f"{ds}:{metric}")) for ds in datasets], _fmt(r.get(metric))]
                    for r in rows])

    sec = [
        "## 22. Unified Fair Comparison — all methods + CE-Raw (query_gen-test)", "",
        f"Every row is evaluated on the **same held-out `query_gen-test` ({sizes.get('test','1,079')} "
        "queries)** with the same metric aggregation (`evaluate.eval_variant`). Supervised rankers are "
        "final-fit / held-out: **L6-final** is trained on `query_gen-train`(+dev); **CE (M5/M7, "
        "ce-joint-v3)** on `query_gen-train`; **CE-Raw** on full-corpus-mined negatives "
        f"(data/ce_raw/), reranking the full-corpus top-{depth} with no fusion. Zero-shot "
        "(BM25/BGE/RRF) and unsupervised (M4/A7/QSC) need no training. → a fair, leakage-free, "
        "same-test comparison across **all** methods. Sources: §20 "
        "(`fair_supervised_query_gen.json`) + §21 (`ceraw_query_gen.json`).", "",
        "### 22.1 Macro (%) on query_gen-test", "", macro, "",
        "### 22.2 Per-dataset Recall@1 (%)", "", per_ds("Recall@1"), "",
        "### 22.3 Per-dataset Recall@10 (%)", "", per_ds("Recall@10"), "",
        "### 22.4 Per-dataset nDCG@1 (%)", "", per_ds("nDCG@1"), "",
        "### 22.5 Per-dataset nDCG@10 (%)", "", per_ds("nDCG@10"), "",
        "### 22.6 Reading", "",
        "- All methods share the identical test queries and aggregation, so columns are directly "
        "comparable. Categories still differ in supervision (**supervised**: L6-final, CE, CE-Raw; "
        "**zero-shot**: BM25/BGE/RRF; **unsupervised**: M4/A7/QSC) — compare within intent.",
        "- **CE-Raw vs pipeline-CE (M5/M7):** CE-Raw is decoupled from the M4 Stage-1 in BOTH "
        f"training and inference (full-corpus retrieve→rerank top-{depth}); M5/M7 rerank the M4 pool. "
        "The gap isolates the cost of dropping the strong M4 first stage (see §21 retriever ceilings).",
        "- Per-dataset retriever ceilings (R@100) for CE-Raw are in §21.1; CE-Raw cannot recover gold "
        "outside its first-stage shortlist.", "",
    ]

    # --- write report + sidecar files ---------------------------------------
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "unified_query_gen.json").write_text(json.dumps(
        {"split": "query_gen-test", "sizes": sizes, "rerank_depth": depth, "rows": rows}, indent=2))
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 22.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")
    (COMP / "unified_query_gen.md").write_text("\n".join(
        ["# Unified fair comparison — query_gen-test", "", macro, "",
         "## Per-dataset Recall@1", "", per_ds("Recall@1"), "",
         "## Per-dataset Recall@10", "", per_ds("Recall@10"), "",
         "## Per-dataset nDCG@1", "", per_ds("nDCG@1"), "",
         "## Per-dataset nDCG@10", "", per_ds("nDCG@10"), ""]))
    print("wrote results/comparisons/unified_query_gen.{json,md} + appended §22")


if __name__ == "__main__":
    main()
