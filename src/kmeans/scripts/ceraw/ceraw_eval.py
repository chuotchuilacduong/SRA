"""CE-Raw Phase 3+4 — standalone retrieve-and-rerank on query_gen-test (no stage-1, no fusion).

First stage retrieves from the FULL 26,262-skill corpus (BGE base, fine-tuned BGE, and/or RRF),
NOT the M4 @100 pool. CE-Raw then reranks a deep shortlist (``--rerank-depth``, default 1000;
``full`` = whole corpus) with pure CE scores (no Stage-1 fusion). Evaluated on the SAME
query_gen-test (1,079) as §20, then a §21 comparison table is appended to FULL_M4_V2_RESULTS.md.

    python src/kmeans/scripts/ceraw_eval.py \
        --retrievers bge_base,rrf,bge_ft --rerank-depth 1000 \
        --ce-model results/models/ce-raw-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from kmeans.qsc_ltr_runner import load_ext
from kmeans import cv, evaluate, io

COMP = PROJECT_ROOT / "results" / "comparisons"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
MET = evaluate.REPORT_METRICS
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]


def _pct(block, m):
    return block.get(m, float("nan")) * 100.0


def _md(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def _device():
    import os
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("SRA_ALLOW_MPS") == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def export_endtask_sources(methods):
    """Dump per-query ranked lists as retrieval sources for the end-task harness.

    Writes results/retrieval/{ds}-{source}.json (schema {"results":[{instance_id,
    retrieved:[{skill_id},...]},...]}) for the table methods:
      CE-Raw·bge_base@* -> ceraw_bge_base, ·rrf -> ceraw_rrf, ·bge_ft -> ceraw_bge_ft,
      bge_ft (retriever-only) -> bge_ft_retriever. Rerank depth in the label is ignored.
    """
    out = PROJECT_ROOT / "results" / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for label, typ, _ev, recs in methods:
        if typ == "standalone-CE" and "·" in label:
            source = "ceraw_" + label.split("·", 1)[1].split("@", 1)[0]
        elif typ == "retriever" and label.startswith("bge_ft"):
            source = "bge_ft_retriever"
        else:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["dataset"], []).append(r)
        for ds, rs in by.items():
            (out / f"{ds}-{source}.json").write_text(json.dumps({"results": rs}))
        written.append(source)
    print(f"  exported end-task retrieval sources -> results/retrieval/: {written}", flush=True)


def load_corpus_matrix():
    emb = np.load(PROJECT_ROOT / "results/bge/corpus_emb.npy").astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    ids = json.loads((PROJECT_ROOT / "results/bge/corpus_ids.json").read_text())
    return emb, ids


def dense_topn(qvec: np.ndarray, corpus_emb: np.ndarray, corpus_ids: list, n: int) -> list:
    qv = qvec / (np.linalg.norm(qvec) + 1e-9)
    scores = corpus_emb @ qv
    n = min(n, len(corpus_ids))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [corpus_ids[i] for i in idx]


def first_stage(name, datasets, test_by_ds, corpus_emb, corpus_ids, depth, ce_model_dir):
    """Return {instance_id: [skill_id ranked]} of length>=depth for the chosen retriever."""
    out = {}
    if name in ("bge_base", "rrf"):
        bge = {}
        for ds in datasets:
            qe = np.load(PROJECT_ROOT / f"results/m4_v2/cache/query_emb/{ds}.npy").astype(np.float32)
            qids = json.loads((PROJECT_ROOT / f"results/m4_v2/cache/query_emb/{ds}_ids.json").read_text())
            qrow = {q: i for i, q in enumerate(qids)}
            for iid in test_by_ds[ds]:
                if iid in qrow:
                    bge[iid] = dense_topn(qe[qrow[iid]], corpus_emb, corpus_ids, max(depth, 2000))
        if name == "bge_base":
            return {k: v for k, v in bge.items()}
        # RRF(bm25 top-50, bge full)
        for ds in datasets:
            bm = json.loads((PROJECT_ROOT / f"results/retrieval_bm25/{ds}-bm25.json").read_text())
            bmr = {r["instance_id"]: [c["skill_id"] for c in r["retrieved"]] for r in bm["results"]}
            for iid in test_by_ds[ds]:
                rrf = {}
                for lst in (bmr.get(iid, []), bge.get(iid, [])):
                    for rank, sid in enumerate(lst):
                        rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (60 + rank + 1)
                out[iid] = [s for s, _ in sorted(rrf.items(), key=lambda kv: -kv[1])]
        return out
    if name == "bge_ft":
        from sentence_transformers import SentenceTransformer
        mdir = Path(ce_model_dir).parent / "sr-emb-bge-v1"
        if not mdir.exists():
            print(f"[ceraw] skip bge_ft (no model at {mdir})")
            return {}
        st = SentenceTransformer(str(mdir), device=_device())
        cache = mdir / "corpus_emb.npy"
        if cache.exists():
            ce_emb = np.load(cache)
        else:
            corpus = load_corpus_dict()
            texts = [f"{corpus[s].get('name','')} | {corpus[s].get('description','')} | "
                     f"{str(corpus[s].get('content',''))[:2500]}" for s in corpus_ids]
            ce_emb = st.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
            np.save(cache, ce_emb)
        instances = {r["instance_id"]: r["query"] for ds in datasets for r in io.load_instances(load_ext()[1], ds)}
        for ds in datasets:
            for iid in test_by_ds[ds]:
                q = instances.get(iid)
                if q is None:
                    continue
                qv = st.encode([q], normalize_embeddings=True)[0]
                out[iid] = dense_topn(qv, ce_emb.astype(np.float32), corpus_ids, max(depth, 2000))
        return out
    raise SystemExit(f"unknown retriever {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CE-Raw standalone eval on query_gen-test")
    ap.add_argument("--retrievers", default="bge_base,rrf", help="comma list: bge_base,rrf,bge_ft")
    ap.add_argument("--rerank-depth", default="1000", help="int or 'full'")
    ap.add_argument("--ce-model", type=Path, default=PROJECT_ROOT / "results/models/ce-raw-v1")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ext, cfg = load_ext()
    datasets = ext["datasets"]
    splits = {ds: json.loads((PROJECT_ROOT / f"results/splits/{ds}-query_gen.json").read_text()) for ds in datasets}
    test_by_ds = {ds: set(splits[ds]["test"]) for ds in datasets}
    test_ids = {q for ds in datasets for q in test_by_ds[ds]}
    qtext = {r["instance_id"]: r["query"] for ds in datasets for r in io.load_instances(cfg, ds)}
    gold_by_id = {r["instance_id"]: (r.get("gold_skill_ids") or []) for ds in datasets for r in io.load_instances(cfg, ds)}
    corpus = load_corpus_dict()
    corpus_emb, corpus_ids = load_corpus_matrix()
    depth = len(corpus_ids) if args.rerank_depth == "full" else int(args.rerank_depth)

    from sragents.retrieve.skill_packer import SkillPacker
    from sragents.retrieve.cross_rerank import CrossEncoderReranker
    ce = CrossEncoderReranker(model_name=str(args.ce_model), device=_device(), max_length=256,
                              packer=SkillPacker(mode="field_tagged", max_content_chars=1800), corpus=corpus)

    methods = []  # (label, type, eval_block, records)

    def ds_of(iid):
        return iid.rsplit("_", 1)[0] if iid.rsplit("_", 1)[0] in datasets else next(d for d in datasets if iid in test_by_ds[d])

    for rname in [r.strip() for r in args.retrievers.split(",") if r.strip()]:
        rank = first_stage(rname, datasets, test_by_ds, corpus_emb, corpus_ids, depth, args.ce_model)
        if not rank:
            continue
        # retriever-only row
        ronly = {ds: [] for ds in datasets}
        for ds in datasets:
            for iid in test_by_ds[ds]:
                ronly[ds].append({"instance_id": iid, "dataset": ds, "gold_skill_ids": gold_by_id.get(iid, []),
                                  "retrieved": [{"skill_id": s} for s in rank.get(iid, [])[:100]]})
        ev = evaluate.eval_variant(cfg, ronly)
        methods.append((f"{rname} (retriever-only)", "retriever", ev,
                        [r for ds in datasets for r in ronly[ds]]))
        # CE-Raw rerank
        reranked = {ds: [] for ds in datasets}
        for ds in datasets:
            for iid in test_by_ds[ds]:
                cand = [{"skill_id": s} for s in rank.get(iid, [])[:depth]]
                if not cand:
                    continue
                out = ce.rerank(qtext[iid], cand, top_k=cfg.output_top_k, batch_size=args.batch_size)
                reranked[ds].append({"instance_id": iid, "dataset": ds, "gold_skill_ids": gold_by_id.get(iid, []),
                                     "retrieved": [{"skill_id": c["skill_id"]} for c in out]})
        ev = evaluate.eval_variant(cfg, reranked)
        methods.append((f"CE-Raw·{rname}@{args.rerank_depth}", "standalone-CE", ev,
                        [r for ds in datasets for r in reranked[ds]]))
        print(f"  {rname}: retriever R@100={_pct(methods[-2][2]['macro'],'Recall@100'):.2f} "
              f"-> CE-Raw nDCG@10={_pct(methods[-1][2]['macro'],'nDCG@10'):.2f}", flush=True)

    # export ranked lists as retrieval sources for the end-task eval (results/retrieval/)
    export_endtask_sources(methods)

    # significance: best CE-Raw vs M7@500 / M5@500 (held-out CE) on query_gen-test
    sig = {}
    ce_raw_rows = [m for m in methods if m[1] == "standalone-CE"]
    if ce_raw_rows:
        best = max(ce_raw_rows, key=lambda m: _pct(m[2]["macro"], "nDCG@10"))
        for label, tmpl in [("M7-CE@500", "results/qsc_ltr/ce500/{ds}.jsonl"),
                            ("M5-CE@500", "results/qsc_ltr/m5_500/{ds}.jsonl")]:
            recs = []
            for ds in datasets:
                p = PROJECT_ROOT / tmpl.format(ds=ds)
                if p.exists():
                    recs += [r for r in cv._read_jsonl(p) if r["instance_id"] in test_ids]
            if recs:
                sig[f"{best[0]}_vs_{label}"] = {
                    "Recall@10": cv.paired_bootstrap(cv.per_query_metric(best[3], "recall", 10),
                                                     cv.per_query_metric(recs, "recall", 10)),
                    "nDCG@10": cv.paired_bootstrap(cv.per_query_metric(best[3], "ndcg", 10),
                                                   cv.per_query_metric(recs, "ndcg", 10))}

    # context rows from §20 (if present)
    ctx_rows = []
    s20 = COMP / "fair_supervised_query_gen.json"
    if s20.exists():
        for r in json.loads(s20.read_text())["rows"]:
            if r["method"] in ("L6-final (LTR)", "M5-CE@500", "M7-CE@500", "RRF(BM25+BGE)", "BGE", "BM25"):
                ctx_rows.append([r["method"] + " (§20)", r["type"]] + [f"{r.get(m, float('nan')):.2f}" for m in MET])

    rows = [[lbl, typ] + [f"{_pct(ev['macro'], m):.2f}" for m in MET] for lbl, typ, ev, _ in methods]
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / "ceraw_query_gen.json").write_text(json.dumps(
        {"rerank_depth": args.rerank_depth,
         "rows": [{"method": l, "type": t, **{m: round(_pct(e["macro"], m), 2) for m in MET},
                   **{f"{ds}:{m}": round(_pct(e["by_dataset"].get(ds, {}), m), 2) for ds in datasets for m in MET}}
                  for l, t, e, _ in methods],
         "significance": sig}, indent=2))

    def per_ds(metric):
        return _md(["Method", *datasets, "AVG"],
                   [[l, *[f"{_pct(e['by_dataset'].get(ds, {}), metric):.2f}" for ds in datasets],
                     f"{_pct(e['macro'], metric):.2f}"] for l, _t, e, _ in methods if _t == "standalone-CE"])
    sig_rows = [[k.replace("_", " "), met, f"{st['mean_diff']:+.2f}",
                 f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]]
                for k, b in sig.items() for met, st in b.items() if st.get("mean_diff") is not None]
    sec = [
        "## 21. Standalone CE (raw-corpus, SkillRouter-style) — query_gen-test", "",
        f"First stage retrieves from the **FULL 26,262-skill corpus** (no M4 @100 pool); CE-Raw "
        f"reranks the top-{args.rerank_depth} with **pure CE scores (no fusion)**. CE-Raw is trained "
        "on full-corpus-mined negatives (data/ce_raw/), so it is decoupled from Stage-1 in BOTH "
        "training and inference. Same held-out query_gen-test (1,079) as §20.", "",
        "### 21.1 Macro (%) — CE-Raw variants (+ §20 context rows)", "",
        _md(["Method", "Type", *SHORT], rows + ctx_rows), "",
        "### 21.2 Per-dataset nDCG@10 (%) — CE-Raw variants", "", per_ds("nDCG@10"), "",
        "### 21.3 Significance (paired bootstrap vs held-out CE)", "",
        _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows) if sig_rows else "_(no CE@500 records found)_", "",
        "### 21.4 Reading", "",
        "- CE-Raw is a **standalone retrieve-and-rerank** (SkillRouter recipe) — different category "
        "from pipeline-CE (M5/M7, which rerank the M4 pool). Compare ceilings via retriever-only R@100.",
        "- First-stage recall bounds CE-Raw (reranker can't recover gold outside the shortlist); "
        f"rerank-depth={args.rerank_depth}.", "",
    ]
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 21.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")
    (COMP / "ceraw_query_gen.md").write_text("\n".join(["# CE-Raw — query_gen-test", "",
                                                        _md(["Method", "Type", *SHORT], rows), ""]))
    print("wrote results/comparisons/ceraw_query_gen.{json,md} + appended §21")


if __name__ == "__main__":
    main()
