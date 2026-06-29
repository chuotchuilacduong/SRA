"""Aggregate end-task accuracy eval JSONs into the two paper-style tables.

Reads the per-cell eval summaries produced by `sragents evaluate` (via the
`endtask` experiment) at:

    results/eval/{dataset}/{model_short}/{label}.json   # {"metrics":{accuracy,correct,total}}

and pivots them into one table per model:

    Retrieval | Skill-use | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | Average

Cell = task accuracy (%) for that (dataset, label). **Average is instance-weighted**
over all datasets that ran: Average = 100 * Σcorrect / Σtotal  (matches the
paper's "overall over all instances", NOT a macro mean of per-dataset rates).
LLM Direct / Oracle Skill are retrieval-independent (Retrieval = "—").

    python src/kmeans/scripts/aggregate_endtask_tables.py [--models Qwen3-4B Qwen3-32B] [--workspace results]

Outputs: results/comparisons/endtask_{model_short}.{md,csv} + results/comparisons/endtask_tables.md
(Lives under src/kmeans/scripts/ because experiments/ is gitignored.)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from sragents.config import PROJECT_ROOT

DATASETS = ["theoremqa", "logicbench", "toolqa", "champ", "medcalcbench", "bigcodebench"]

# (source logical name, display) — must match _ENDTASK_SOURCES in
# src/sragents/experiments/definitions.py
SOURCES = [
    ("bm25", "BM25"),
    ("l6_final", "L6_final"),
    ("ceraw_bge_base", "CE-Raw·bge_base@1000"),
    ("ceraw_rrf", "CE-Raw·rrf@1000"),
    ("ceraw_bge_ft", "CE-Raw·bge_ft@1000"),
    ("bge_ft_retriever", "bge_ft (retriever-only)"),
]
STRATS = [("fsi", "Full-Skill Injection"), ("sel", "LLM Selection"), ("pd", "Progressive Disclosure")]


def _row_plan() -> list[tuple[str, str, str]]:
    """Ordered (Retrieval display, Skill-use display, label). "—" = retrieval-independent."""
    rows = [("—", "LLM Direct", "llm_direct"), ("—", "Oracle Skill", "oracle_skill")]
    for src, sdisp in SOURCES:
        for strat, stdisp in STRATS:
            rows.append((sdisp, stdisp, f"{strat}__{src}"))
    return rows


def _read(workspace: Path, ds: str, model_short: str, label: str) -> dict | None:
    p = workspace / "eval" / ds / model_short / f"{label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("metrics")


def _fmt(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else (x if x else "n/a")


def _md(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def build_table(workspace: Path, model_short: str):
    headers = ["Retrieval", "Skill-use", *DATASETS, "Average"]
    md_rows, csv_rows = [], []
    for rdisp, sdisp, label in _row_plan():
        cells, tot_c, tot_n, any_found = [], 0, 0, False
        for ds in DATASETS:
            m = _read(workspace, ds, model_short, label)
            if m is None:
                cells.append("n/a")
            else:
                any_found = True
                cells.append(_fmt(100.0 * m["accuracy"]))
                tot_c += m["correct"]
                tot_n += m["total"]
        if not any_found:
            continue  # cell never ran for this model — skip the row entirely
        avg = _fmt(100.0 * tot_c / tot_n) if tot_n else "n/a"
        md_rows.append([rdisp, sdisp, *cells, avg])
        csv_rows.append({"Retrieval": rdisp, "Skill-use": sdisp, "label": label,
                         **{ds: c for ds, c in zip(DATASETS, cells)}, "Average": avg, "n": tot_n})
    return headers, md_rows, csv_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate end-task accuracy tables")
    ap.add_argument("--models", nargs="*", default=["Qwen3-4B", "Qwen3-32B"],
                    help="model_short names (basename of --model used at infer time)")
    ap.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "results")
    args = ap.parse_args()

    comp = PROJECT_ROOT / "results" / "comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    combined = ["# End-task accuracy (query_gen-test) — SRA paper-style tables", "",
                "Cell = task accuracy (%). Average = instance-weighted (Σcorrect/Σtotal) "
                "over datasets that ran. `n/a` = cell not yet evaluated.", ""]

    for model_short in args.models:
        headers, md_rows, csv_rows = build_table(args.workspace, model_short)
        if not md_rows:
            print(f"[skip] {model_short}: no eval JSONs under {args.workspace}/eval/*/{model_short}/")
            continue
        table_md = _md(headers, md_rows)
        (comp / f"endtask_{model_short}.md").write_text(
            f"# End-task accuracy — {model_short} (query_gen-test)\n\n{table_md}\n")
        with (comp / f"endtask_{model_short}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Retrieval", "Skill-use", "label", *DATASETS, "Average", "n"])
            w.writeheader(); w.writerows(csv_rows)
        combined += [f"## End-task accuracy — {model_short}", "", table_md, ""]
        print(f"[ok] {model_short}: {len(md_rows)} rows -> results/comparisons/endtask_{model_short}.{{md,csv}}")

    (comp / "endtask_tables.md").write_text("\n".join(combined) + "\n")
    print("wrote results/comparisons/endtask_tables.md")


if __name__ == "__main__":
    main()
