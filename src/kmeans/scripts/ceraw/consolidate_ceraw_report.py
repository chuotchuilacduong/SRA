"""Merge the per-split CE-Raw eval outputs into ONE complete §21 in
FULL_M4_V2_RESULTS.md.

The §21 eval was run split across two GPU nodes (worker-1: bge_base,rrf;
worker-2: bge_ft) to use both H100s. Each ``ceraw_eval.py`` process writes its
OWN §21 (last-writer-wins), so neither the report nor results/comparisons/
ceraw_query_gen.json ends up with all three retrievers. This script reads both
splits' snapshotted JSONs, merges their rows + significance, and rewrites §21
with the full set — preserving everything up to §20 (no overwrite of prior
sections). Pure post-processing: no GPU, no re-rerank.

Inputs (both must exist):
  results/comparisons/ceraw_query_gen.bge_ft.json        (worker-2 snapshot)
  results/comparisons/ceraw_query_gen.bge_base_rrf.json  (worker-1 snapshot)
Outputs:
  FULL_M4_V2_RESULTS.md  (§21 replaced with the complete table)
  results/comparisons/ceraw_query_gen.{json,md}  (complete, merged)
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kmeans import evaluate                      # noqa: E402
from kmeans.qsc_ltr_runner import load_ext       # noqa: E402

COMP = PROJECT_ROOT / "results" / "comparisons"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
MET = evaluate.REPORT_METRICS
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def _fmt(v):
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "nan"


def main():
    ext, _cfg = load_ext()
    datasets = ext["datasets"]

    br = json.loads((COMP / "ceraw_query_gen.bge_base_rrf.json").read_text())
    ft = json.loads((COMP / "ceraw_query_gen.bge_ft.json").read_text())
    rerank_depth = br.get("rerank_depth") or ft.get("rerank_depth")

    # Merge rows: bge_base(ret), CE-Raw·bge_base, rrf(ret), CE-Raw·rrf, then bge_ft(ret), CE-Raw·bge_ft
    rows_all = list(br["rows"]) + list(ft["rows"])
    # Merge significance (each split contributed its own best-CE-Raw-vs-CE@500 comparison)
    sig = {**br.get("significance", {}), **ft.get("significance", {})}

    # --- 21.1 macro table rows -------------------------------------------------
    md_rows = [[r["method"], r["type"]] + [_fmt(r.get(m)) for m in MET] for r in rows_all]

    # context rows from §20
    ctx_rows = []
    s20 = COMP / "fair_supervised_query_gen.json"
    if s20.exists():
        keep = ("L6-final (LTR)", "M5-CE@500", "M7-CE@500", "RRF(BM25+BGE)", "BGE", "BM25")
        for r in json.loads(s20.read_text())["rows"]:
            if r["method"] in keep:
                ctx_rows.append([r["method"] + " (§20)", r["type"]] + [_fmt(r.get(m)) for m in MET])

    # --- 21.2 per-dataset nDCG@10 for CE-Raw variants --------------------------
    ce_rows = [r for r in rows_all if r["type"] == "standalone-CE"]
    per_ds = _md(["Method", *datasets, "AVG"],
                 [[r["method"], *[_fmt(r.get(f"{ds}:nDCG@10")) for ds in datasets],
                   _fmt(r.get("nDCG@10"))] for r in ce_rows])

    # --- 21.3 significance -----------------------------------------------------
    sig_rows = [[k.replace("_", " "), met, f"{st['mean_diff']:+.2f}",
                 f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]]
                for k, b in sig.items() for met, st in b.items()
                if isinstance(st, dict) and st.get("mean_diff") is not None]

    sec = [
        "## 21. Standalone CE (raw-corpus, SkillRouter-style) — query_gen-test", "",
        "First stage retrieves from the **FULL 26,262-skill corpus** (no M4 @100 pool); CE-Raw "
        f"reranks the top-{rerank_depth} with **pure CE scores (no fusion)**. CE-Raw is trained "
        "on full-corpus-mined negatives (data/ce_raw/), so it is decoupled from Stage-1 in BOTH "
        "training and inference. Same held-out query_gen-test (1,079) as §20.", "",
        "_Eval was split across two H100s (worker-1: bge_base,rrf; worker-2: bge_ft) and merged "
        "here into one complete table — all three retrievers' results retained._", "",
        "### 21.1 Macro (%) — CE-Raw variants (+ §20 context rows)", "",
        _md(["Method", "Type", *SHORT], md_rows + ctx_rows), "",
        "### 21.2 Per-dataset nDCG@10 (%) — CE-Raw variants", "", per_ds, "",
        "### 21.3 Significance (paired bootstrap vs held-out CE)", "",
        _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows)
        if sig_rows else "_(no CE@500 records found)_", "",
        "### 21.4 Reading", "",
        "- CE-Raw is a **standalone retrieve-and-rerank** (SkillRouter recipe) — different category "
        "from pipeline-CE (M5/M7, which rerank the M4 pool). Compare ceilings via retriever-only R@100.",
        "- First-stage recall bounds CE-Raw (reranker can't recover gold outside the shortlist); "
        f"rerank-depth={rerank_depth}.", "",
    ]

    # rewrite §21 (preserve everything up to it — no overwrite of §20 or earlier)
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 21.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")

    # complete merged comparison artifacts
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "ceraw_query_gen.json").write_text(json.dumps(
        {"rerank_depth": rerank_depth, "rows": rows_all, "significance": sig}, indent=2))
    (COMP / "ceraw_query_gen.md").write_text("\n".join(
        ["# CE-Raw — query_gen-test (merged: bge_base, rrf, bge_ft)", "",
         _md(["Method", "Type", *SHORT], md_rows), ""]))

    print(f"merged §21 written: {len(rows_all)} method rows "
          f"({sum(r['type'] == 'standalone-CE' for r in rows_all)} CE-Raw), "
          f"{len(ctx_rows)} §20 ctx rows, {len(sig_rows)} sig rows")
    print("methods:", [r["method"] for r in rows_all])


if __name__ == "__main__":
    main()
