"""Analysis + visualization of skill-probe outputs (CPU, no model).

Reads the per-(query,method) JSONs and pooled hidden-state .npy files written by
run_probe.py and produces:
  (a) M4 vs CE-HYRR retrieval comparison (gold-rank, %top1-positive-utility, hit@k)
  (b) utility vs candidate rank (+ Spearman)
  (c) layer-wise linear-probe separability (predict correctness v from each layer)
  (d) 2D PCA / t-SNE of pooled vectors, colored by v / method / condition
  (e) cosine-similarity summary among no_skill / gold / candidate vectors
  (f) per-layer centroid distance between v=1 and v=0

Outputs PNGs + analysis/summary.json + analysis/SUMMARY.md under OUTPUT_ROOT.

    python -m experiments.skill_probe_hidden.analyze [--datasets ...] [--umap]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.skill_probe_hidden import config  # noqa: E402

ANALYSIS_DIR = config.OUTPUT_ROOT / "analysis"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_method_jsons(dataset: str) -> list[dict]:
    ds_dir = config.OUTPUT_ROOT / dataset
    return [json.loads(p.read_text()) for p in sorted(ds_dir.glob("*__*.json"))
            if p.name != "index.json"]


def load_vector(hs_rel_path: str) -> np.ndarray:
    """Load a pooled hidden-state array [num_layers, hidden] (float16->float32)."""
    return np.load(config.OUTPUT_ROOT / hs_rel_path).astype(np.float32)


def collect_unique_probes(jsons: list[dict]) -> list[dict]:
    """One row per UNIQUE hidden-state file (dedup across methods/conditions).

    Returns rows {hs_path, v, condition, rank|None}. v is probe correctness.
    """
    seen: dict[str, dict] = {}
    for j in jsons:
        ns = j["probes"]["no_skill"]
        seen.setdefault(ns["hidden_state_path"], {"hs_path": ns["hidden_state_path"],
                        "v": ns["v"], "condition": "no_skill", "rank": None})
        gs = j["probes"].get("gold_skill")
        if gs:
            seen.setdefault(gs["hidden_state_path"], {"hs_path": gs["hidden_state_path"],
                            "v": gs["v"], "condition": "gold", "rank": None})
        for c in j["candidates"]:
            seen.setdefault(c["hidden_state_path"], {"hs_path": c["hidden_state_path"],
                            "v": c["v"], "condition": "candidate", "rank": c["rank"]})
    return list(seen.values())


# --------------------------------------------------------------------------
# (a) retrieval comparison + (b) utility vs rank
# --------------------------------------------------------------------------
def retrieval_summary(jsons: list[dict]) -> dict:
    by_method = defaultdict(list)
    for j in jsons:
        by_method[j["method"]].append(j)
    out = {}
    for method, js in by_method.items():
        n = len(js)
        top1_pos = np.mean([1.0 if (j["candidates"] and j["candidates"][0]["utility"] > 0) else 0.0
                            for j in js]) if n else 0.0
        gold_in10 = np.mean([1.0 if j["metrics"]["gold_in_top10"] else 0.0 for j in js]) if n else 0.0
        gold_ranks = [j["metrics"]["gold_rank_in_candidates"] for j in js
                      if j["metrics"]["gold_rank_in_candidates"] is not None]
        hitk = {}
        for k in (1, 5, 10):
            hitk[f"hit@{k}"] = float(np.mean([
                1.0 if any(c["is_gold"] and c["rank"] <= k for c in j["candidates"]) else 0.0
                for j in js])) if n else 0.0
        # utility-vs-rank (Spearman) over all candidate rows
        ranks, utils = [], []
        for j in js:
            for c in j["candidates"]:
                ranks.append(c["rank"]); utils.append(c["utility"])
        spearman = _spearman(ranks, utils)
        out[method] = {
            "n_queries": n, "pct_top1_positive_utility": float(top1_pos),
            "pct_gold_in_top10": float(gold_in10),
            "median_gold_rank": float(np.median(gold_ranks)) if gold_ranks else None,
            **hitk, "spearman_rank_utility": spearman,
            "mean_v_no": float(np.mean([j["metrics"]["v_no"] for j in js])) if n else 0.0,
            "mean_v_gold": float(np.mean([j["metrics"]["v_gold"] for j in js
                                          if j["metrics"]["v_gold"] is not None])) if n else 0.0,
            "mean_num_positive_utility": float(np.mean([j["metrics"]["num_positive_utility"]
                                                        for j in js])) if n else 0.0,
        }
    return out


def _spearman(a, b) -> float | None:
    if len(a) < 3:
        return None
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        ar = ar - ar.mean(); br = br - br.mean()
        denom = np.sqrt((ar ** 2).sum() * (br ** 2).sum())
        return float((ar * br).sum() / denom) if denom else None


def plot_utility_vs_rank(jsons: list[dict], out_png: Path) -> None:
    by_method = defaultdict(lambda: defaultdict(list))
    for j in jsons:
        for c in j["candidates"]:
            by_method[j["method"]][c["rank"]].append(c["utility"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, rank_map in by_method.items():
        ranks = sorted(rank_map)
        means = [np.mean(rank_map[r]) for r in ranks]
        ax.plot(ranks, means, marker="o", label=f"{method} (mean utility)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("candidate rank"); ax.set_ylabel("mean utility (v_i - v_no)")
    ax.set_title("Utility vs retrieval rank"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# (c) layer-wise linear probe + (f) centroid distance
# --------------------------------------------------------------------------
def layerwise_probe(rows: list[dict], label_key="v") -> dict | None:
    """Per-layer 5-fold CV logistic-regression accuracy predicting label from hidden vec."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    y = np.array([r[label_key] for r in rows])
    if len(set(y.tolist())) < 2:
        return None
    counts = np.bincount(y) if y.min() >= 0 else None
    n_splits = int(min(5, counts.min())) if counts is not None else 5
    if n_splits < 2:
        return None

    mats = [load_vector(r["hs_path"]) for r in rows]          # each [L, H]
    L = mats[0].shape[0]
    accs, stds = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    for layer in range(L):
        X = np.stack([m[layer] for m in mats])               # [N, H]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        sc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
        accs.append(float(sc.mean())); stds.append(float(sc.std()))
    return {"n": len(rows), "n_splits": n_splits, "baseline": float(max(np.mean(y), 1 - np.mean(y))),
            "layer_acc": accs, "layer_std": stds}


def centroid_distance(rows: list[dict]) -> list[float] | None:
    y = np.array([r["v"] for r in rows])
    if len(set(y.tolist())) < 2:
        return None
    mats = [load_vector(r["hs_path"]) for r in rows]
    L = mats[0].shape[0]
    dist = []
    pos = [m for m, v in zip(mats, y) if v == 1]
    neg = [m for m, v in zip(mats, y) if v == 0]
    for layer in range(L):
        c1 = np.mean([m[layer] for m in pos], axis=0)
        c0 = np.mean([m[layer] for m in neg], axis=0)
        dist.append(float(np.linalg.norm(c1 - c0)))
    return dist


def plot_layer_curves(probe_v, probe_cond, centroid, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if probe_v:
        xs = range(len(probe_v["layer_acc"]))
        ax.plot(xs, probe_v["layer_acc"], marker="o", label=f"probe v (acc), base={probe_v['baseline']:.2f}")
        ax.axhline(probe_v["baseline"], color="gray", ls="--", lw=0.8)
    ax.set_xlabel("layer (0=embed)"); ax.set_ylabel("CV accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Layer-wise decodability of correctness (v)"); ax.legend(loc="lower right")
    if centroid:
        ax2 = ax.twinx()
        ax2.plot(range(len(centroid)), centroid, color="tab:red", alpha=0.5, label="centroid dist")
        ax2.set_ylabel("‖centroid(v=1)-centroid(v=0)‖", color="tab:red")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# (d) 2D projection
# --------------------------------------------------------------------------
def plot_projection(rows: list[dict], layer: int, out_prefix: Path, use_umap=False) -> None:
    mats = [load_vector(r["hs_path"]) for r in rows]
    L = mats[0].shape[0]
    layer = min(layer, L - 1)
    X = np.stack([m[layer] for m in mats])
    methods = {"pca": _pca2(X)}
    if X.shape[0] >= 6:
        methods["tsne"] = _tsne2(X)
    if use_umap:
        um = _umap2(X)
        if um is not None:
            methods["umap"] = um
    color_v = np.array([r["v"] for r in rows])
    color_cond = np.array([{"no_skill": 0, "gold": 1, "candidate": 2}[r["condition"]] for r in rows])
    for name, emb in methods.items():
        if emb is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sc0 = axes[0].scatter(emb[:, 0], emb[:, 1], c=color_v, cmap="coolwarm", s=18)
        axes[0].set_title(f"{name} layer {layer} — color=correct(v)")
        fig.colorbar(sc0, ax=axes[0])
        sc1 = axes[1].scatter(emb[:, 0], emb[:, 1], c=color_cond, cmap="viridis", s=18)
        axes[1].set_title(f"{name} layer {layer} — color=condition (0=no,1=gold,2=cand)")
        fig.colorbar(sc1, ax=axes[1])
        fig.tight_layout(); fig.savefig(f"{out_prefix}_{name}_L{layer}.png", dpi=120); plt.close(fig)


def _pca2(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=config.SEED).fit_transform(X)


def _tsne2(X):
    from sklearn.manifold import TSNE
    perp = max(2, min(30, X.shape[0] // 4))
    return TSNE(n_components=2, perplexity=perp, random_state=config.SEED, init="pca").fit_transform(X)


def _umap2(X):
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=config.SEED).fit_transform(X)
    except Exception:
        return None


# --------------------------------------------------------------------------
# (e) cosine summary
# --------------------------------------------------------------------------
def cosine_summary(jsons: list[dict], layer: int = -1) -> dict:
    def vec(path):
        m = load_vector(path)
        return m[min(layer, m.shape[0] - 1)] if layer >= 0 else m[-1]

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    cos_no_gold, cos_cand_gold_pos, cos_cand_gold_neg = [], [], []
    for j in jsons:
        ns = vec(j["probes"]["no_skill"]["hidden_state_path"])
        gs = j["probes"].get("gold_skill")
        if gs:
            g = vec(gs["hidden_state_path"])
            cos_no_gold.append(cos(ns, g))
            for c in j["candidates"]:
                cv = vec(c["hidden_state_path"])
                (cos_cand_gold_pos if c["utility"] > 0 else cos_cand_gold_neg).append(cos(cv, g))
    return {
        "mean_cos_no_vs_gold": float(np.mean(cos_no_gold)) if cos_no_gold else None,
        "mean_cos_cand_vs_gold_util_pos": float(np.mean(cos_cand_gold_pos)) if cos_cand_gold_pos else None,
        "mean_cos_cand_vs_gold_util_le0": float(np.mean(cos_cand_gold_neg)) if cos_cand_gold_neg else None,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def analyze_dataset(dataset: str, use_umap=False) -> dict:
    jsons = load_method_jsons(dataset)
    if not jsons:
        return {"dataset": dataset, "error": "no method JSONs found"}
    out_dir = ANALYSIS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_unique_probes(jsons)
    retr = retrieval_summary(jsons)
    plot_utility_vs_rank(jsons, out_dir / "utility_vs_rank.png")

    probe_v = layerwise_probe(rows, "v")
    cdist = centroid_distance(rows)
    plot_layer_curves(probe_v, None, cdist, out_dir / "layerwise_probe.png")

    L = load_vector(rows[0]["hs_path"]).shape[0]
    plot_projection(rows, layer=L - 1, out_prefix=str(out_dir / "proj_last"), use_umap=use_umap)
    plot_projection(rows, layer=L // 2, out_prefix=str(out_dir / "proj_mid"), use_umap=use_umap)

    summary = {
        "dataset": dataset, "n_method_jsons": len(jsons), "n_unique_probes": len(rows),
        "retrieval": retr, "cosine": cosine_summary(jsons),
        "layerwise_probe_v": {k: probe_v[k] for k in ("n", "baseline")} if probe_v else None,
        "best_probe_layer_acc": (float(np.max(probe_v["layer_acc"])) if probe_v else None),
        "best_probe_layer": (int(np.argmax(probe_v["layer_acc"])) if probe_v else None),
    }
    (out_dir / "summary.json").write_text(json.dumps(
        {**summary, "layer_acc": probe_v["layer_acc"] if probe_v else None,
         "centroid_distance": cdist}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze skill-probe hidden states")
    ap.add_argument("--datasets", nargs="*", default=config.DATASETS)
    ap.add_argument("--umap", action="store_true", help="also compute UMAP (needs umap-learn)")
    args = ap.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for ds in args.datasets:
        s = analyze_dataset(ds, use_umap=args.umap)
        summaries.append(s)
        print(f"[{ds}] {s.get('n_unique_probes', 0)} probes; "
              f"best layer-probe acc={s.get('best_probe_layer_acc')}")

    (ANALYSIS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    _write_markdown(summaries)
    print(f"\nwrote {ANALYSIS_DIR}/summary.json + SUMMARY.md + per-dataset plots")


def _write_markdown(summaries: list[dict]) -> None:
    lines = ["# Skill-Probe Hidden-State Analysis — Summary", "",
             f"Generator: {config.MODEL_ID} (HF, thinking OFF). Datasets: {config.DATASETS}", "",
             "## Retrieval (M4 vs CE-HYRR)", "",
             "| dataset | method | n | %top1 util>0 | hit@1 | hit@5 | hit@10 | spearman(rank,util) | mean v_no | mean v_gold |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in summaries:
        for m, r in (s.get("retrieval") or {}).items():
            lines.append(f"| {s['dataset']} | {m} | {r['n_queries']} | {r['pct_top1_positive_utility']:.2f} | "
                         f"{r['hit@1']:.2f} | {r['hit@5']:.2f} | {r['hit@10']:.2f} | "
                         f"{r['spearman_rank_utility']} | {r['mean_v_no']:.2f} | {r['mean_v_gold']:.2f} |")
    lines += ["", "## Hidden-state separability (predict correctness v per layer)", "",
              "| dataset | n probes | base acc | best layer | best acc |", "|---|--:|--:|--:|--:|"]
    for s in summaries:
        lp = s.get("layerwise_probe_v")
        base = f"{lp['baseline']:.2f}" if lp else "—"
        lines.append(f"| {s['dataset']} | {s.get('n_unique_probes',0)} | {base} | "
                     f"{s.get('best_probe_layer')} | {s.get('best_probe_layer_acc')} |")
    (ANALYSIS_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
