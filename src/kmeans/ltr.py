"""Family LTR — lightweight learning-to-rank (plan §5), no Cross-Encoder.

Query-level stratified split (anti-leakage, plan §5.2), LightGBM LambdaRank
(LTR-B) and a linear pairwise ranker (LTR-A), trained on numeric features only.
Hyperparameters are chosen on dev; metrics are reported on the held-out test
split. Feature subsets define variants L0-L7.
"""

from __future__ import annotations

import logging
import os

# LightGBM's OpenMP backend segfaults with >1 thread on the local macOS/libomp
# build (verified: n_jobs>1 -> SIGSEGV/exit 139). Pin single-threaded BEFORE any
# OpenMP-linked import. Safe and deterministic; on Linux/HPC bump n_jobs back up.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from .config import M4V2Config
from .evaluate import eval_records
from .ltr_features import column_indices

logger = logging.getLogger(__name__)


# --- query iteration over a cached feature table -----------------------------

def iter_queries(table: dict):
    """Yield (instance_id, X_q, y_q, skill_ids_q, gold_q) per query of a table."""
    off = 0
    sizes = table["group_sizes"]
    for i, qid in enumerate(table["instance_ids"]):
        n = int(sizes[i])
        sl = slice(off, off + n)
        yield (qid, table["X"][sl], table["y"][sl],
               table["skill_ids"][sl], table["gold"][i])
        off += n


# --- split (stratified by dataset, query-level) ------------------------------

def make_split(tables: dict[str, dict], split_cfg: dict) -> dict[str, set]:
    seed = int(split_cfg["seed"])
    tr_f, dev_f = float(split_cfg["train"]), float(split_cfg["dev"])
    train, dev, test = set(), set(), set()
    rng = np.random.default_rng(seed)
    for ds in sorted(tables):
        qids = list(tables[ds]["instance_ids"])
        perm = rng.permutation(len(qids))
        n = len(qids)
        n_tr, n_dev = int(round(n * tr_f)), int(round(n * dev_f))
        for j, idx in enumerate(perm):
            if j < n_tr:
                train.add(qids[idx])
            elif j < n_tr + n_dev:
                dev.add(qids[idx])
            else:
                test.add(qids[idx])
    return {"train": train, "dev": dev, "test": test}


def collect_queries(tables: dict[str, dict], id_set: set, col_idx: list[int]):
    """Return list of (instance_id, dataset, X_q[:,cols], y_q, skill_ids, gold)."""
    out = []
    for ds in sorted(tables):
        for qid, Xq, yq, sids, gold in iter_queries(tables[ds]):
            if qid in id_set:
                X = np.nan_to_num(Xq[:, col_idx].astype(np.float32),
                                  nan=0.0, posinf=1e6, neginf=-1e6)
                out.append((qid, ds, X, yq, sids, gold))
    return out


# --- prediction helpers ------------------------------------------------------

def _records_from_scores(queries, score_fn, top_k: int) -> list[dict]:
    recs = []
    for qid, ds, X, y, sids, gold in queries:
        scores = score_fn(X)
        order = np.argsort(-scores, kind="stable")[:top_k]
        ranked = [{"skill_id": sids[i], "rank": r + 1, "score": float(scores[i]),
                   "is_gold": sids[i] in set(gold)}
                  for r, i in enumerate(order)]
        recs.append({"instance_id": qid, "query_id": qid, "dataset": ds,
                     "gold_skill_ids": list(gold), "retrieved": ranked})
    return recs


def _dev_ndcg(queries, score_fn, cfg: M4V2Config) -> float:
    recs = _records_from_scores(queries, score_fn, cfg.eval_top_k)
    return eval_records(cfg, recs).get("nDCG@10", 0.0)


# --- LightGBM LambdaRank -----------------------------------------------------

def train_lightgbm(tr, dev, cfg: M4V2Config, lgb_cfg: dict):
    import lightgbm as lgb
    Xtr = np.vstack([q[2] for q in tr]); ytr = np.concatenate([q[3] for q in tr])
    gtr = [len(q[3]) for q in tr]
    Xdev = np.vstack([q[2] for q in dev]); ydev = np.concatenate([q[3] for q in dev])
    gdev = [len(q[3]) for q in dev]

    best = None
    for num_leaves in lgb_cfg["num_leaves"]:
        for lr in lgb_cfg["learning_rate"]:
            for n_est in lgb_cfg["n_estimators"]:
                for min_leaf in lgb_cfg["min_data_in_leaf"]:
                    model = lgb.LGBMRanker(
                        objective="lambdarank", metric="ndcg",
                        n_estimators=int(n_est), num_leaves=int(num_leaves),
                        learning_rate=float(lr), min_child_samples=int(min_leaf),
                        random_state=42, n_jobs=1, verbose=-1)
                    model.fit(Xtr, ytr, group=gtr, eval_set=[(Xdev, ydev)],
                              eval_group=[gdev], eval_at=tuple(lgb_cfg["eval_at"]),
                              callbacks=[lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]),
                                                            verbose=False)])
                    ndcg = _dev_ndcg(dev, lambda X: model.predict(X), cfg)
                    params = {"num_leaves": num_leaves, "learning_rate": lr,
                              "n_estimators": n_est, "min_data_in_leaf": min_leaf,
                              "best_iteration": getattr(model, "best_iteration_", None)}
                    if best is None or ndcg > best[0]:
                        best = (ndcg, model, params)
    logger.info("  lightgbm best dev nDCG@10=%.4f params=%s", best[0], best[2])
    return best[1], best[2]


# --- linear pairwise ranker (RankNet-style logistic on feature diffs) --------

def train_linear(tr, dev, cfg: M4V2Config, lin_cfg: dict):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler().fit(np.vstack([q[2] for q in tr]))
    rng = np.random.default_rng(int(lin_cfg["seed"]))
    npp = int(lin_cfg["negative_per_positive"])
    max_pairs = int(lin_cfg["max_pairs_per_query"])

    PX, PY = [], []
    for qid, ds, X, y, sids, gold in tr:
        Xs = scaler.transform(X)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        cnt = 0
        for p in pos:
            if cnt >= max_pairs:
                break
            chosen = rng.choice(neg, size=min(npp, len(neg)), replace=False)
            for nidx in chosen:
                if cnt >= max_pairs:
                    break
                diff = Xs[p] - Xs[nidx]
                PX.append(diff); PY.append(1)
                PX.append(-diff); PY.append(0)
                cnt += 1
    PX = np.asarray(PX, dtype=np.float64); PY = np.asarray(PY, dtype=np.int32)

    best = None
    for C in lin_cfg["C_values"]:
        lr = LogisticRegression(C=float(C), fit_intercept=False, max_iter=2000)
        lr.fit(PX, PY)
        coef = lr.coef_[0]
        ndcg = _dev_ndcg(dev, lambda X: scaler.transform(X) @ coef, cfg)
        if best is None or ndcg > best[0]:
            best = (ndcg, coef, float(C))
    logger.info("  linear best dev nDCG@10=%.4f C=%s", best[0], best[2])
    coef = best[1]
    return (scaler, coef), {"C": best[2], "feature_weights": coef.tolist()}


# --- variant driver ----------------------------------------------------------

def run_variant(cfg: M4V2Config, tables: dict[str, dict], split: dict[str, set],
                variant: dict, lgb_cfg: dict, lin_cfg: dict) -> tuple[list[dict], dict]:
    """Train one LTR variant and predict on the TEST split. Returns (records, meta)."""
    col_idx = column_indices(variant["groups"])
    tr = collect_queries(tables, split["train"], col_idx)
    dev = collect_queries(tables, split["dev"], col_idx)
    test = collect_queries(tables, split["test"], col_idx)
    logger.info("[%s] %s | feats=%d train_q=%d dev_q=%d test_q=%d",
                variant["id"], variant["name"], len(col_idx), len(tr), len(dev), len(test))

    if variant["model"] == "lightgbm":
        model, params = train_lightgbm(tr, dev, cfg, lgb_cfg)
        score_fn = lambda X: model.predict(X)
        names = _sel_names(variant)
        fi = getattr(model, "feature_importances_", None)
        imp = dict(zip(names, [int(x) for x in fi])) if fi is not None else {}
        meta = {"model": "lightgbm", "params": params, "feature_importances": imp,
                "n_features": len(col_idx)}
    else:
        (scaler, coef), params = train_linear(tr, dev, cfg, lin_cfg)
        score_fn = lambda X: scaler.transform(X) @ coef
        meta = {"model": "linear", "params": {"C": params["C"]},
                "feature_weights": dict(zip(_sel_names(variant), params["feature_weights"])),
                "n_features": len(col_idx)}

    records = _records_from_scores(test, score_fn, cfg.output_top_k)
    return records, meta


def _sel_names(variant: dict) -> list[str]:
    from .ltr_features import FEATURE_GROUPS, ALL_FEATURES
    keep = {f for g in variant["groups"] for f in FEATURE_GROUPS[g]}
    return [f for f in ALL_FEATURES if f in keep]
