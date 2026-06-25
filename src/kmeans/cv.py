"""5-fold cross-validation + statistical robustness for the LTR rankers.

Query-level folds (each query is test exactly once → out-of-fold (OOF) predictions
cover all queries with no leakage). For each fold: train on 3 folds, tune on 1
fold (dev), evaluate once on the held-out fold (test). L6 uses the FULL LightGBM
grid; L0-L5/L7 use a reduced dev-tuned grid (ablation). Also emits a depth-matched
**L6@100** (rerank only the RRF top-100, pool-matched to the CE Method-7 baseline).

Workers are single-threaded (OMP_NUM_THREADS=1) and run in separate processes to
sidestep the macOS LightGBM OpenMP crash while using many cores.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import warnings
from pathlib import Path

# LightGBM's sklearn wrapper warns on every ndarray predict ("X does not have
# valid feature names"); harmless and floods CV logs. Silence it.
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

logger = logging.getLogger(__name__)

N_FOLDS = 5
FOLD_SEED = 1234

# Full LightGBM grid (spec §13). Reduced grid for ablation variants.
FULL_GRID = {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05],
             "n_estimators": [200, 500], "min_data_in_leaf": [10, 30, 50],
             "early_stopping_rounds": 50, "eval_at": [1, 5, 10]}
REDUCED_GRID = {"num_leaves": [31, 63], "learning_rate": [0.05],
                "n_estimators": [500], "min_data_in_leaf": [30],
                "early_stopping_rounds": 50, "eval_at": [1, 5, 10]}
FULL_GRID_VARIANTS = {"L6"}


# --- folds -------------------------------------------------------------------

def make_folds(tables: dict[str, dict], n_folds: int = N_FOLDS,
               seed: int = FOLD_SEED) -> dict[str, int]:
    """Assign each query to a fold, balanced within each dataset (stratified)."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for ds in sorted(tables):
        qids = list(tables[ds]["instance_ids"])
        perm = rng.permutation(len(qids))
        for pos, idx in enumerate(perm):
            fold_of[qids[idx]] = pos % n_folds   # round-robin after shuffle
    return fold_of


def fold_splits(fold_of: dict[str, int], i: int, n_folds: int = N_FOLDS):
    test = {q for q, f in fold_of.items() if f == i}
    dev = {q for q, f in fold_of.items() if f == (i + 1) % n_folds}
    train = {q for q, f in fold_of.items() if f not in (i, (i + 1) % n_folds)}
    return train, dev, test


# --- per-query metrics (for significance) ------------------------------------

_LOG2 = 1.0 / np.log2(np.arange(2, 402))


def recall_at_k(retrieved: list[str], gold: set, k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def ndcg_at_k(retrieved: list[str], gold: set, k: int) -> float:
    if not gold:
        return 0.0
    rels = np.array([1.0 if s in gold else 0.0 for s in retrieved[:k]])
    dcg = float(rels @ _LOG2[:len(rels)])
    idcg = float(_LOG2[:min(len(gold), k)].sum())
    return dcg / idcg if idcg > 0 else 0.0


def per_query_metric(records: list[dict], metric: str, k: int) -> dict[str, float]:
    """instance_id -> metric value, for paired tests."""
    out = {}
    fn = recall_at_k if metric == "recall" else ndcg_at_k
    for r in records:
        gold = set(r.get("gold_skill_ids") or [])
        if not gold:
            continue
        ids = [c["skill_id"] for c in r.get("retrieved", [])]
        out[r["instance_id"]] = fn(ids, gold, k)
    return out


# --- paired bootstrap significance -------------------------------------------

def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     n_boot: int = 10000, seed: int = 7) -> dict:
    """Paired bootstrap over the shared queries: tests mean(a) - mean(b) > 0.

    Returns mean diff, 95% CI of the diff, and a two-sided p-value.
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "mean_diff": None, "p_value": None}
    da = np.array([a[k] for k in keys])
    db = np.array([b[k] for k in keys])
    diff = da - db
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided p-value: fraction of bootstrap means on the opposite side of 0
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": n, "mean_diff": round(obs * 100, 3),
            "ci95_low": round(float(lo) * 100, 3), "ci95_high": round(float(hi) * 100, 3),
            "p_value": round(float(min(p, 1.0)), 5)}


# --- fold-aggregate stats ----------------------------------------------------

def fold_mean_std(per_fold: list[dict], metric: str) -> dict:
    """mean ± std ± CI95 across folds for one metric key (already %)."""
    vals = np.array([f[metric] for f in per_fold if metric in f], dtype=float)
    if len(vals) == 0:
        return {"mean": None}
    mean = float(vals.mean()); std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(vals))
    return {"mean": round(mean, 2), "std": round(std, 2), "ci95": round(ci, 2),
            "n_folds": len(vals), "per_fold": [round(v, 2) for v in vals]}


# --- worker (top-level for multiprocessing) ----------------------------------

def run_unit(args: dict) -> dict:
    """Train one (variant, fold) and write test predictions. Returns metrics+meta.

    args: {variant, fold_i, ext_path, cache_dir, out_dir, n_folds, seed}
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    from kmeans.qsc_ltr_runner import load_ext
    from kmeans import ltr, ltr_features, evaluate

    variant = args["variant"]
    fold_i = args["fold_i"]
    ext, cfg = load_ext(args["ext_path"])
    cache_dir = Path(args["cache_dir"])
    out_dir = Path(args["out_dir"])

    tables = {ds: ltr_features.load_features(cache_dir / f"{ds}.npz")
              for ds in ext["datasets"]}
    fold_of = make_folds(tables, args["n_folds"], args["seed"])
    train_ids, dev_ids, test_ids = fold_splits(fold_of, fold_i, args["n_folds"])

    col = ltr_features.column_indices(variant["groups"])
    tr = ltr.collect_queries(tables, train_ids, col)
    dev = ltr.collect_queries(tables, dev_ids, col)
    test = ltr.collect_queries(tables, test_ids, col)

    grid = FULL_GRID if variant["id"] in FULL_GRID_VARIANTS else REDUCED_GRID
    meta = {}
    if variant["model"] == "lightgbm":
        model, params = ltr.train_lightgbm(tr, dev, cfg, grid)
        score_fn = lambda X: model.predict(X)
        names = ltr._sel_names(variant)
        fi = getattr(model, "feature_importances_", None)
        meta = {"params": params,
                "feature_importances": dict(zip(names, [int(x) for x in fi])) if fi is not None else {}}
    else:
        (scaler, coef), params = ltr.train_linear(tr, dev, cfg, ext["ltr"]["linear_pairwise"])
        score_fn = lambda X: scaler.transform(X) @ coef
        meta = {"params": {"C": params["C"]}}

    recs = ltr._records_from_scores(test, score_fn, cfg.output_top_k)
    out_path = out_dir / variant["id"] / f"fold{fold_i}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in recs))

    # depth-matched @100 for L6 (rerank only RRF top-100)
    extra = {}
    if variant["id"] in FULL_GRID_VARIANTS:
        ir = ltr_features.ALL_FEATURES.index("rrf_rank")
        recs100 = []
        for qid, ds, X, y, sids, gold in test:
            keep = np.where(X[:, ir] <= 100)[0]
            s = score_fn(X[keep])
            order = keep[np.argsort(-s, kind="stable")][:cfg.output_top_k]
            recs100.append({"instance_id": qid, "dataset": ds, "gold_skill_ids": list(gold),
                            "retrieved": [{"skill_id": sids[i]} for i in order]})
        p100 = out_dir / (variant["id"] + "_d100") / f"fold{fold_i}.jsonl"
        p100.parent.mkdir(parents=True, exist_ok=True)
        p100.write_text("\n".join(json.dumps(r) for r in recs100))
        extra["d100_metrics"] = evaluate.eval_variant(cfg, _group(recs100))["macro"]

    m = evaluate.eval_variant(cfg, _group(recs))
    return {"variant": variant["id"], "fold": fold_i,
            "macro": {k: round(v * 100, 4) for k, v in m["macro"].items()},
            "by_dataset": {ds: {k: round(v * 100, 4) for k, v in mm.items()}
                           for ds, mm in m["by_dataset"].items()},
            "meta": meta, **({"d100_macro": {k: round(v * 100, 4)
                                             for k, v in extra["d100_metrics"].items()}}
                             if extra else {})}


def _group(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r.get("dataset", "unknown"), []).append(r)
    return out


# --- OOF assembly + baseline loaders (used by the orchestrator) --------------

def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def assemble_oof(out_dir: Path, variant_id: str, n_folds: int = N_FOLDS) -> list[dict]:
    """Concatenate a variant's per-fold test predictions → one record per query."""
    recs = []
    for i in range(n_folds):
        recs += _read_jsonl(Path(out_dir) / variant_id / f"fold{i}.jsonl")
    return recs


def load_our_baseline(cfg, vid: str, datasets: list[str], qsc_dir: Path) -> dict[str, list[dict]]:
    """Full-set per-query records for A0/A1/A2/A7 (M4-v2) or Q2/Q6 (QSC), by dataset."""
    by_ds = {}
    if vid.startswith("A"):
        slug = cfg.variant(vid).slug
        for ds in datasets:
            recs = _read_jsonl(cfg.paths["out_dir"] / slug / f"{ds}.jsonl")
            if recs:
                by_ds[ds] = recs
    else:  # Q2/Q6
        for ds in datasets:
            recs = _read_jsonl(Path(qsc_dir) / "qsc" / vid / f"{ds}.jsonl")
            if recs:
                by_ds[ds] = recs
    return by_ds


def load_retrieval_baseline(tmpl: str, datasets: list[str],
                            gold_by_id: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Load a per-dataset retrieval results file into by-dataset records. Gold is
    taken from ``gold_by_id`` (authoritative, from the instances) since some
    baseline files (e.g. LinearRAG) omit ``gold_skill_ids``."""
    from sragents.config import PROJECT_ROOT
    by_ds = {}
    for ds in datasets:
        p = PROJECT_ROOT / tmpl.format(ds=ds)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        recs = [{"instance_id": r["instance_id"], "dataset": ds,
                 "gold_skill_ids": gold_by_id.get(r["instance_id"], r.get("gold_skill_ids") or []),
                 "retrieved": r["retrieved"]} for r in data["results"]]
        by_ds[ds] = recs
    return by_ds


def eval_on_ids(cfg, by_ds: dict[str, list[dict]], ids: set | None):
    """Macro metrics over a query-id subset (or all if ids is None)."""
    from kmeans import evaluate
    filt = {}
    for ds, recs in by_ds.items():
        rs = [r for r in recs if (ids is None or r["instance_id"] in ids)]
        if rs:
            filt[ds] = rs
    return evaluate.eval_variant(cfg, filt)
