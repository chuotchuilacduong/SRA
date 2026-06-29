"""§24 — Experiment 1: pool union with bge_ft top-K for L6-final+bge_ft.

Lifts the M4 Stage-1 pool ceiling by ranking over P_union(q) = P_M4(q) ∪ TopK_bge_ft(q),
K ∈ {50,100,200,500}. Same query_gen train/dev/test split + eval protocol as §23. NO CE.

Per-K (`--K`): build union pools, recompute LTR features over the union, train
`L6-union-bgeft@K` (LightGBM LambdaRank, 59 feats), evaluate on query_gen-test. Writes
results/comparisons/_union_k{K}.json + results/retrieval/{ds}-l6_union_bgeft_k{K}.json.

Aggregate (`--report`): combine all K + the §23 reference rows (L6-final, L6-final+bge_ft,
bge_ft retriever-only, M7-CE@500 from l6_bgeft.json), paired-bootstrap, and emit
l6_bgeft_union_metrics.{csv,md,json}, l6_bgeft_union_delta.md, pool_coverage.md,
error_analysis.md, and append §24 to FULL_M4_V2_RESULTS.md.

Feature design for the union (honest missing flags — §2 of the spec):
  - Candidates in the M4 pool keep their real 52 features (retrieval/m4/a7/qsc/confidence/
    bge_ft from the ltr_features_bgeft cache).
  - Candidates added from bge_ft top-K (outside M4) get: query-level features broadcast
    (a7-query + confidence, identical per query), bge_ft within-pool features recomputed
    over the union, and ALL per-candidate retrieval/m4/a7/qsc features = 0 WITH a
    `missing_pool_features` flag (never silently filled).
  - New `union` group (7): in_m4_pool, in_bgeft_topk, in_both, missing_pool_features,
    bgeft_full_cosine, bgeft_full_rank, bgeft_full_inv_rank (full = over the 26,262 corpus).

Run: python src/kmeans/scripts/run_l6_union_bgeft.py --K 100 [--n-jobs 14]
     python src/kmeans/scripts/run_l6_union_bgeft.py --report
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "14")  # before lightgbm/ltr import (ltr pins to 1)

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
_bootstrap.setup_logging()

import numpy as np                                       # noqa: E402
from sragents.config import PROJECT_ROOT                 # noqa: E402
from kmeans.qsc_ltr_runner import load_ext               # noqa: E402
from kmeans import cv, ltr, ltr_features as lf, evaluate  # noqa: E402

COMP = PROJECT_ROOT / "results" / "comparisons"
RETR = PROJECT_ROOT / "results" / "retrieval"
ANALYSIS = PROJECT_ROOT / "results" / "analysis"
REPORT = PROJECT_ROOT / "FULL_M4_V2_RESULTS.md"
BGEFT_CACHE = PROJECT_ROOT / "results" / "m4_v2" / "cache" / "ltr_features_bgeft"
CORPUS_EMB = PROJECT_ROOT / "results" / "models" / "sr-emb-bge-v1" / "corpus_emb.npy"
CORPUS_IDS = PROJECT_ROOT / "results" / "bge" / "corpus_ids.json"
QEMB = PROJECT_ROOT / "results" / "m4_v2" / "cache" / "query_emb_ft"

MET = evaluate.REPORT_METRICS
SHORT = ["R@1", "R@5", "R@10", "R@50", "R@100", "nD@1", "nD@5", "nD@10"]
KS = [50, 100, 200, 500]
# standalone reference methods (zero-shot + unsupervised) to show alongside, from §20/§23
BASELINES = ["BM25", "BGE", "RRF(BM25+BGE)", "A1 M4", "A7 PRF", "Q6 QSC", "A0 RRF", "Q2 QSC"]
QIDX = np.array([lf.ALL_FEATURES.index(n) for n in lf._QUERY_FEATS])      # 10 query-level cols
BGEFT_IDX = [lf.ALL_FEATURES.index(n) for n in lf.FEATURE_GROUPS["bge_ft"]]  # 7 bge_ft cols
UNION_FEATS = ["in_m4_pool", "in_bgeft_topk", "in_both", "missing_pool_features",
               "bgeft_full_cosine", "bgeft_full_rank", "bgeft_full_inv_rank"]
UNION_FEATURE_NAMES = list(lf.ALL_FEATURES) + UNION_FEATS  # 52 + 7 = 59
GRID = {"num_leaves": [31, 63], "learning_rate": [0.03, 0.05], "n_estimators": [500],
        "min_data_in_leaf": [30], "early_stopping_rounds": 50, "eval_at": [1, 5, 10]}


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


def load_query_gen(ext):
    sp = {ds: json.loads((PROJECT_ROOT / f"results/splits/{ds}-query_gen.json").read_text())
          for ds in ext["datasets"]}
    return ({q for ds in sp for q in sp[ds]["train"]},
            {q for ds in sp for q in sp[ds].get("dev", [])},
            {q for ds in sp for q in sp[ds]["test"]})


# --- per-K build + train + eval ----------------------------------------------

def build_union(ds, K, corpus_emb, corpus_ids, cidx, qemb):
    """Per-query union pool + 59-col feature matrix; returns (queries, coverage)."""
    t = lf.load_features(BGEFT_CACHE / f"{ds}.npz")
    iids, sizes, sids_all, X_all, gold_all = (t["instance_ids"], t["group_sizes"],
                                              t["skill_ids"], t["X"], t["gold"])
    U = np.stack([qemb[i] for i in iids]).astype(np.float32)
    S = U @ corpus_emb.T                                   # (nq, N) cosine (both normalized)
    N = corpus_emb.shape[0]
    queries, cov, off = [], [], 0
    for qi, iid in enumerate(iids):
        n_m4 = int(sizes[qi]); sl = slice(off, off + n_m4); off += n_m4
        m4_sids = list(sids_all[sl]); m4_set = set(m4_sids)
        Xm4 = X_all[sl]
        gold = set(gold_all[qi])
        s = S[qi]
        order = np.argsort(-s, kind="stable")
        topk_sids = [corpus_ids[j] for j in order[:K]]; topk_set = set(topk_sids)
        added = [sd for sd in topk_sids if sd not in m4_set]
        union_sids = m4_sids + added
        n = len(union_sids)
        rank_lookup = np.empty(N, dtype=np.float32)
        rank_lookup[order] = np.arange(1, N + 1, dtype=np.float32)

        base = np.zeros((n, 52), dtype=np.float32)
        base[:n_m4] = Xm4
        if added and n_m4 > 0:
            base[n_m4:, QIDX] = Xm4[0, QIDX]               # broadcast query-level to added

        ci = np.array([cidx.get(sd, -1) for sd in union_sids])
        valid = ci >= 0
        cos = np.where(valid, s[ci.clip(min=0)], 0.0).astype(np.float32)
        frank = np.where(valid, rank_lookup[ci.clip(min=0)], float(N + 1)).astype(np.float32)
        # recompute bge_ft within-union features for ALL union candidates
        o = np.argsort(-cos, kind="stable")
        rnk = np.empty(n, dtype=np.float32); rnk[o] = np.arange(1, n + 1, dtype=np.float32)
        topcos = float(cos[o[0]]) if n else 0.0
        std = float(cos.std())
        z = ((cos - float(cos.mean())) / std).astype(np.float32) if std > 1e-12 \
            else np.zeros(n, dtype=np.float32)
        base[:, BGEFT_IDX] = np.column_stack(
            [cos, rnk, 1.0 / rnk, rnk / float(n), (rnk == 1).astype(np.float32), cos - topcos, z])

        in_m4 = np.array([1.0 if sd in m4_set else 0.0 for sd in union_sids], dtype=np.float32)
        in_tk = np.array([1.0 if sd in topk_set else 0.0 for sd in union_sids], dtype=np.float32)
        union7 = np.column_stack([in_m4, in_tk, in_m4 * in_tk, 1.0 - in_m4,
                                  cos, frank, 1.0 / frank]).astype(np.float32)
        X59 = np.concatenate([base, union7], axis=1).astype(np.float32)
        y = np.array([1 if sd in gold else 0 for sd in union_sids], dtype=np.int32)
        queries.append((iid, ds, X59, y, union_sids, list(gold_all[qi])))

        gi_m4 = sum(1 for g in gold if g in m4_set)
        gi_un = sum(1 for g in gold if g in set(union_sids))
        ng = max(len(gold), 1)
        cov.append({"instance_id": iid, "dataset": ds,
                    "gold_in_m4": int(gi_m4 > 0), "gold_in_union": int(gi_un > 0),
                    "gold_recall_m4": gi_m4 / ng, "gold_recall_union": gi_un / ng,
                    "n_m4": n_m4, "n_added": len(added)})
    return queries, cov


def train_lgbm(tr, dev, cfg, n_jobs):
    import lightgbm as lgb
    Xtr = np.vstack([q[2] for q in tr]); ytr = np.concatenate([q[3] for q in tr])
    gtr = [len(q[3]) for q in tr]
    Xdev = np.vstack([q[2] for q in dev]); ydev = np.concatenate([q[3] for q in dev])
    gdev = [len(q[3]) for q in dev]
    best = None
    for nl in GRID["num_leaves"]:
        for lr in GRID["learning_rate"]:
            for ne in GRID["n_estimators"]:
                for ml in GRID["min_data_in_leaf"]:
                    m = lgb.LGBMRanker(objective="lambdarank", metric="ndcg",
                                       n_estimators=int(ne), num_leaves=int(nl),
                                       learning_rate=float(lr), min_child_samples=int(ml),
                                       random_state=42, n_jobs=int(n_jobs), verbose=-1)
                    m.fit(Xtr, ytr, group=gtr, eval_set=[(Xdev, ydev)], eval_group=[gdev],
                          eval_at=tuple(GRID["eval_at"]),
                          callbacks=[lgb.early_stopping(int(GRID["early_stopping_rounds"]), verbose=False)])
                    recs = ltr._records_from_scores(dev, lambda X: m.predict(X), cfg.eval_top_k)
                    nd = evaluate.eval_records(cfg, recs).get("nDCG@10", 0.0)
                    p = {"num_leaves": nl, "learning_rate": lr, "n_estimators": ne,
                         "min_data_in_leaf": ml, "best_iteration": getattr(m, "best_iteration_", None)}
                    if best is None or nd > best[0]:
                        best = (nd, m, p)
    return best[1], best[2]


def _cov_summary(cov, datasets):
    def agg(rows):
        if not rows:
            return {}
        return {"gold_in_m4_rate": float(np.mean([r["gold_in_m4"] for r in rows])) * 100,
                "gold_in_union_rate": float(np.mean([r["gold_in_union"] for r in rows])) * 100,
                "gold_recall_m4": float(np.mean([r["gold_recall_m4"] for r in rows])) * 100,
                "gold_recall_union": float(np.mean([r["gold_recall_union"] for r in rows])) * 100,
                "mean_pool_size": float(np.mean([r["n_m4"] + r["n_added"] for r in rows])),
                "mean_added": float(np.mean([r["n_added"] for r in rows])), "n": len(rows)}
    return {"macro_by_dataset": {ds: agg([r for r in cov if r["dataset"] == ds]) for ds in datasets},
            "pooled": agg(cov)}


def run_k(K, n_jobs):
    ext, cfg = load_ext()
    datasets = ext["datasets"]
    train, dev, test = load_query_gen(ext)
    corpus_emb = np.load(CORPUS_EMB).astype(np.float32)
    corpus_ids = json.loads(CORPUS_IDS.read_text())
    cidx = {s: i for i, s in enumerate(corpus_ids)}

    tr, dv, te, cov_test = [], [], [], []
    print(f"[K={K}] building union pools ...", flush=True)
    for ds in datasets:
        qemb = {i: e for i, e in zip(json.loads((QEMB / f"{ds}_ids.json").read_text()),
                                     np.load(QEMB / f"{ds}.npy").astype(np.float32))}
        queries, cov = build_union(ds, K, corpus_emb, corpus_ids, cidx, qemb)
        cset = {c["instance_id"]: c for c in cov}
        for q in queries:
            (tr if q[0] in train else dv if q[0] in dev else te).append(q)
        cov_test += [cset[q[0]] for q in queries if q[0] in test]
        print(f"  {ds:14} q={len(queries)} mean_added={np.mean([c['n_added'] for c in cov]):.0f}", flush=True)

    print(f"[K={K}] train (n_jobs={n_jobs}) train_q={len(tr)} dev_q={len(dv)} test_q={len(te)} ...", flush=True)
    model, params = train_lgbm(tr, dv, cfg, n_jobs)
    recs = ltr._records_from_scores(te, lambda X: model.predict(X), cfg.output_top_k)
    ev = evaluate.eval_variant(cfg, _group(recs))

    # save retrieval source (test) + per-K metrics json
    RETR.mkdir(parents=True, exist_ok=True)
    for ds, rs in _group(recs).items():
        (RETR / f"{ds}-l6_union_bgeft_k{K}.json").write_text(json.dumps({"results": rs}))
    macro = {m: round(ev["macro"].get(m, float("nan")) * 100, 2) for m in MET}
    byds = {ds: {m: round(ev["by_dataset"].get(ds, {}).get(m, float("nan")) * 100, 2) for m in MET}
            for ds in datasets}
    COMP.mkdir(parents=True, exist_ok=True)
    (COMP / f"_union_k{K}.json").write_text(json.dumps(
        {"K": K, "method": f"L6-union-bgeft@{K}", "type": "supervised", "params": params,
         "macro": macro, "by_dataset": byds, "coverage": _cov_summary(cov_test, datasets),
         "coverage_per_query": cov_test, "n_test": len(te)}, indent=2))
    print(f"[K={K}] DONE macro nDCG@10={macro['nDCG@10']} R@100={macro['Recall@100']} "
          f"gold_in_union={_cov_summary(cov_test, datasets)['pooled']['gold_in_union_rate']:.2f}%", flush=True)


# --- aggregate report ---------------------------------------------------------

def load_recs(tmpl, test_ids):
    out = []
    for ds in load_ext()[0]["datasets"]:
        p = RETR / tmpl.format(ds=ds)
        if p.exists():
            out += [r for r in json.loads(p.read_text()).get("results", []) if r["instance_id"] in test_ids]
    return out


def _row_get(row, m, ds=None):
    return float(row.get(f"{ds}:{m}" if ds else m, float("nan")))


def report(n_jobs):
    ext, cfg = load_ext()
    datasets = ext["datasets"]
    _, _, test = load_query_gen(ext)

    # reference rows from §23 l6_bgeft.json
    lb = {r["method"]: r for r in json.loads((COMP / "l6_bgeft.json").read_text())["rows"]}
    refs = [lb[k] for k in ["L6-final", "L6-final+bge_ft", "bge_ft (retriever-only)", "M7-CE@500"] + BASELINES
            if k in lb]
    ks = [K for K in KS if (COMP / f"_union_k{K}.json").exists()]
    unions = [json.loads((COMP / f"_union_k{K}.json").read_text()) for K in ks]

    # unified rows: method -> {macro m, per-ds f"{ds}:{m}", type}
    def ref_to_row(r):
        return r  # already {method,type,m,ds:m}

    def union_to_row(u):
        row = {"method": u["method"], "type": "supervised"}
        for m in MET:
            row[m] = u["macro"][m]
            for ds in datasets:
                row[f"{ds}:{m}"] = u["by_dataset"][ds][m]
        return row

    order = ["L6-final", "L6-final+bge_ft"] + [u["method"] for u in unions] + \
            ["bge_ft (retriever-only)", "M7-CE@500"] + BASELINES
    rowmap = {r["method"]: ref_to_row(r) for r in refs}
    rowmap.update({u["method"]: union_to_row(u) for u in unions})
    rows = [rowmap[m] for m in order if m in rowmap]

    # ---- significance: each union@K vs L6-final+bge_ft and vs bge_ft-alone ----
    l6bg = load_recs("{ds}-l6_bgeft.json", test)
    alone = load_recs("{ds}-bge_ft_retriever.json", test)
    qm = lambda recs, met, k: cv.per_query_metric(recs, met, k)
    sig = {}
    for K in ks:
        ur = load_recs(f"{{ds}}-l6_union_bgeft_k{K}.json", test)
        sig[f"L6-union@{K}_vs_L6+bge_ft"] = {
            "Recall@10": cv.paired_bootstrap(qm(ur, "recall", 10), qm(l6bg, "recall", 10)),
            "nDCG@10": cv.paired_bootstrap(qm(ur, "ndcg", 10), qm(l6bg, "ndcg", 10))}
        sig[f"L6-union@{K}_vs_bge_ft-alone"] = {
            "Recall@10": cv.paired_bootstrap(qm(ur, "recall", 10), qm(alone, "recall", 10)),
            "nDCG@10": cv.paired_bootstrap(qm(ur, "ndcg", 10), qm(alone, "ndcg", 10))}

    # ---- metrics tables ----
    macro_md = _md(["Method", "Type", *SHORT],
                   [[r["method"], r.get("type", ""), *[f"{_row_get(r, m):.2f}" for m in MET]] for r in rows])

    def per_ds(metric):
        return _md(["Method", *datasets, "AVG"],
                   [[r["method"], *[f"{_row_get(r, metric, ds):.2f}" for ds in datasets],
                     f"{_row_get(r, metric):.2f}"] for r in rows])

    COMP.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (COMP / "l6_bgeft_union_metrics.json").write_text(json.dumps(
        {"split": "query_gen-test", "rows": rows, "significance": sig,
         "coverage": {u["method"]: u["coverage"]["pooled"] for u in unions}}, indent=2))
    import csv
    with (COMP / "l6_bgeft_union_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "type"] + MET, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    (COMP / "l6_bgeft_union_metrics.md").write_text("\n".join(
        ["# Pool union with bge_ft top-K — query_gen-test", "", macro_md, "",
         "## Per-dataset nDCG@10", "", per_ds("nDCG@10"), "",
         "## Per-dataset Recall@10", "", per_ds("Recall@10"), "",
         "## Per-dataset Recall@100", "", per_ds("Recall@100"), ""]))

    # ---- delta vs L6-final+bge_ft ----
    base = rowmap["L6-final+bge_ft"]
    delta_macro = _md(["Method", *SHORT],
                      [[u["method"], *[f"{_row_get(rowmap[u['method']], m) - _row_get(base, m):+.2f}" for m in MET]]
                       for u in unions])
    delta_ndcg = _md(["Method", *datasets, "AVG"],
                     [[u["method"], *[f"{_row_get(rowmap[u['method']], 'nDCG@10', ds) - _row_get(base, 'nDCG@10', ds):+.2f}"
                                      for ds in datasets],
                       f"{_row_get(rowmap[u['method']], 'nDCG@10') - _row_get(base, 'nDCG@10'):+.2f}"]
                      for u in unions])
    (COMP / "l6_bgeft_union_delta.md").write_text("\n".join(
        ["# L6-union-bgeft@K − L6-final+bge_ft (pp), query_gen-test", "",
         "## Macro Δ", "", delta_macro, "", "## Per-dataset nDCG@10 Δ", "", delta_ndcg, ""]))

    # ---- pool coverage ----
    cov_rows = [["M4 pool (L6-final+bge_ft)",
                 f"{unions[0]['coverage']['pooled']['gold_in_m4_rate']:.2f}",
                 f"{unions[0]['coverage']['pooled']['gold_recall_m4']:.2f}",
                 f"{_row_get(base, 'Recall@100'):.2f}", "—"]]
    for u in unions:
        p = u["coverage"]["pooled"]
        cov_rows.append([u["method"], f"{p['gold_in_union_rate']:.2f}", f"{p['gold_recall_union']:.2f}",
                         f"{_row_get(rowmap[u['method']], 'Recall@100'):.2f}", f"{p['mean_added']:.0f}"])
    cov_ds = _md(["Method", *datasets, "AVG"],
                 [["M4 pool", *[f"{unions[0]['coverage']['macro_by_dataset'][ds]['gold_in_m4_rate']:.1f}"
                                for ds in datasets],
                   f"{unions[0]['coverage']['pooled']['gold_in_m4_rate']:.1f}"]] +
                 [[u["method"], *[f"{u['coverage']['macro_by_dataset'][ds]['gold_in_union_rate']:.1f}"
                                  for ds in datasets],
                   f"{u['coverage']['pooled']['gold_in_union_rate']:.1f}"] for u in unions])
    (ANALYSIS / "l6_bgeft_union_pool_coverage.md").write_text("\n".join(
        ["# Pool coverage — M4 vs union@K (query_gen-test)", "",
         "Gold-in-pool rate = % of queries with ≥1 gold skill in the pool (the recall ceiling). "
         "Model R@100 ≤ gold-in-pool.", "",
         _md(["Pool", "gold-in-pool %", "gold-recall %", "model R@100", "mean added"], cov_rows), "",
         "## Per-dataset gold-in-pool % (query-level)", "", cov_ds, ""]))

    # ---- error analysis / attribution (best K by macro nDCG@10) ----
    best = max(unions, key=lambda u: u["macro"]["nDCG@10"])
    bk = best["K"]
    cov_by_id = {c["instance_id"]: c for c in best["coverage_per_query"]}
    ur = {r["instance_id"]: r for r in load_recs(f"{{ds}}-l6_union_bgeft_k{bk}.json", test)}
    recovered = [i for i, c in cov_by_id.items() if c["gold_in_union"] and not c["gold_in_m4"]]
    rec_top10 = 0
    for i in recovered:
        r = ur.get(i)
        if r:
            gold = set(r.get("gold_skill_ids") or [])
            if gold & {x["skill_id"] for x in r.get("retrieved", [])[:10]}:
                rec_top10 += 1
    # ordering attribution: nDCG@10 on queries whose gold was already in the M4 pool
    in_m4_ids = {i for i, c in cov_by_id.items() if c["gold_in_m4"]}
    nd_u = cv.per_query_metric(list(ur.values()), "ndcg", 10)
    nd_b = cv.per_query_metric(l6bg, "ndcg", 10)
    sub = [i for i in in_m4_ids if i in nd_u and i in nd_b]
    ord_u = float(np.mean([nd_u[i] for i in sub])) * 100 if sub else float("nan")
    ord_b = float(np.mean([nd_b[i] for i in sub])) * 100 if sub else float("nan")
    # per-dataset nDCG@10 regressions vs L6-final+bge_ft
    regress = [(ds, _row_get(rowmap[best['method']], 'nDCG@10', ds) - _row_get(base, 'nDCG@10', ds))
               for ds in datasets]
    losers = [f"{ds} ({d:+.2f}pp)" for ds, d in regress if d < -1.0]

    crit = {
        f"@100 nDCG@10 > 80.95 (p<.05)": (
            "L6-union-bgeft@100" in rowmap and _row_get(rowmap.get("L6-union-bgeft@100", {}), "nDCG@10") > 80.95
            and sig.get("L6-union@100_vs_L6+bge_ft", {}).get("nDCG@10", {}).get("p_value", 1) is not None
            and (sig.get("L6-union@100_vs_L6+bge_ft", {}).get("nDCG@10", {}).get("p_value") or 1) < 0.05),
        "R@100 >= 99.0 (best K)": _row_get(rowmap[best["method"]], "Recall@100") >= 99.0,
        "LogicBench & ToolQA Recall@10 improve (best K)": all(
            _row_get(rowmap[best["method"]], "Recall@10", ds) > _row_get(base, "Recall@10", ds)
            for ds in ["logicbench", "toolqa"]),
        "no dataset loses > 1.0pp nDCG@10 (best K)": len(losers) == 0,
    }
    (ANALYSIS / "l6_bgeft_union_error_analysis.md").write_text("\n".join(
        [f"# Error analysis — union pool (best K = {bk})", "",
         f"- Recovered queries (gold outside M4 pool but inside union@{bk}): **{len(recovered)}**; "
         f"of these the union model placed a gold in top-10 for **{rec_top10}** "
         f"({(100*rec_top10/max(len(recovered),1)):.1f}%). → realized recall gain.",
         f"- Ordering attribution: on the {len(sub)} test queries whose gold was ALREADY in the M4 pool, "
         f"mean nDCG@10 = **{ord_u:.2f}** (union@{bk}) vs **{ord_b:.2f}** (L6-final+bge_ft) "
         f"→ Δ {ord_u - ord_b:+.2f}pp (pure ordering effect, recall held constant).",
         f"- Total macro nDCG@10 gain @{bk} vs L6-final+bge_ft: "
         f"{_row_get(rowmap[best['method']], 'nDCG@10') - _row_get(base, 'nDCG@10'):+.2f}pp.",
         f"- Per-dataset nDCG@10 regressions > 1pp vs L6-final+bge_ft: "
         f"{', '.join(losers) if losers else 'none'}.", "",
         "## Success criteria", "",
         _md(["Criterion", "Pass"], [[k, "✅" if v else "❌"] for k, v in crit.items()]), ""]))

    # ---- §24 in the report ----
    sig_rows = [[k.replace("_", " "), met, f"{st['mean_diff']:+.2f}",
                 f"[{st['ci95_low']:+.2f},{st['ci95_high']:+.2f}]", st["p_value"]]
                for k, b in sig.items() for met, st in b.items() if st.get("mean_diff") is not None]
    best_lbl = best["method"]
    sec = [
        "## 24. Pool Union with bge_ft Top-K", "",
        "Experiment 1 (plan §6): lift the M4 Stage-1 pool ceiling by ranking over "
        "P_union(q) = P_M4(q) ∪ TopK_bge_ft(q), K ∈ {50,100,200,500}, with L6 (LightGBM "
        "LambdaRank, **no CE**). Candidates added from bge_ft top-K (outside the M4 pool) carry "
        "their bge_ft signal + a `missing_pool_features` flag; per-candidate retrieval/m4/a7/qsc "
        "features are never silently filled. Same held-out query_gen-test (1,079) as §20–§23.", "",
        "### 24.1 Macro (%) on query_gen-test", "", macro_md, "",
        "### 24.2 Per-dataset nDCG@10 (%)", "", per_ds("nDCG@10"), "",
        "### 24.3 Per-dataset Recall@10 (%)", "", per_ds("Recall@10"), "",
        "### 24.4 Per-dataset Recall@100 (%)", "", per_ds("Recall@100"), "",
        "### 24.5 Pool coverage (gold-in-pool %, the recall ceiling)", "",
        _md(["Pool", "gold-in-pool %", "gold-recall %", "model R@100", "mean added"], cov_rows), "",
        "### 24.6 Δ vs L6-final+bge_ft (macro, pp)", "", delta_macro, "",
        "### 24.7 Significance (paired bootstrap)", "",
        _md(["Comparison", "Metric", "Δ (pp)", "95% CI", "p"], sig_rows) if sig_rows else "_(n/a)_", "",
        "### 24.8 Reading", "",
        f"- Best K = **{bk}** (macro nDCG@10 {best['macro']['nDCG@10']:.2f}). "
        f"Union pool lifts gold-in-pool {unions[0]['coverage']['pooled']['gold_in_m4_rate']:.2f}% → "
        f"{best['coverage']['pooled']['gold_in_union_rate']:.2f}% and model R@100 "
        f"{_row_get(base, 'Recall@100'):.2f} → {_row_get(rowmap[best_lbl], 'Recall@100'):.2f}.",
        f"- {best_lbl} vs L6-final+bge_ft (nDCG@10): "
        f"{_row_get(rowmap[best_lbl], 'nDCG@10') - _row_get(base, 'nDCG@10'):+.2f}pp; "
        f"vs bge_ft-alone: {_row_get(rowmap[best_lbl], 'nDCG@10') - _row_get(lb['bge_ft (retriever-only)'], 'nDCG@10'):+.2f}pp. "
        "See §24.7 for p-values and results/analysis/l6_bgeft_union_error_analysis.md for the "
        "recall-vs-ordering attribution.",
        f"- Per-dataset nDCG@10 regressions > 1pp vs L6-final+bge_ft: "
        f"{', '.join(losers) if losers else 'none'}. Outputs: "
        "`results/comparisons/l6_bgeft_union_*`, `results/analysis/l6_bgeft_union_*`.", "",
    ]
    text = REPORT.read_text() if REPORT.exists() else "# FULL_M4_V2_RESULTS.md\n"
    cut = text.find("\n## 24.")
    text = (text[:cut] if cut != -1 else text).rstrip() + "\n\n"
    REPORT.write_text(text + "\n".join(sec) + "\n")

    print("\n=== SUMMARY ===", flush=True)
    for r in rows:
        print(f"  {r['method']:22} nDCG@10={_row_get(r,'nDCG@10'):6.2f}  R@100={_row_get(r,'Recall@100'):6.2f}", flush=True)
    print(f"best K={bk}; wrote l6_bgeft_union_metrics/delta/coverage/error_analysis + §24", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()
    if args.report:
        report(args.n_jobs)
    elif args.K is not None:
        run_k(args.K, args.n_jobs)
    else:
        ap.error("pass --K <int> or --report")


if __name__ == "__main__":
    main()
