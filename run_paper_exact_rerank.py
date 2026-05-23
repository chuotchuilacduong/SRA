#!/usr/bin/env python3
"""Paper-exact Track A reproduction: BM25 top-50 → LLM rerank.

Wraps `sragents rerank` over the 6 SRA-Bench datasets using a local
Qwen3-4B model (via Ollama OpenAI-compatible endpoint by default),
then computes Recall@k and nDCG@k for k in {1, 10, 50, 100}.

Note: rerank operates on a fixed BM25 pool of 50 candidates, so any
metric at k > 50 caps at the BM25@50 ceiling.
"""
import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

DATASETS = ["champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"]
KS = [1, 10, 50, 100]

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1")
WORKERS = int(os.environ.get("WORKERS", "4"))  # Ollama serializes by default
RERANK_TOP_K = 50
SUBSET_PER_DATASET = int(os.environ.get("SUBSET_PER_DATASET", "0"))  # 0 = full

BM25_DIR = REPO / "results/retrieval_bm25"
OUTPUT_DIR = REPO / "results/retrieval_rerank_paper"
SUBSET_DIR = REPO / "results/retrieval_bm25_subset"
INSTANCES_DIR = REPO / "data/bench/instances"
CORPUS_PATH = REPO / "data/bench/corpus/corpus.json"

PY = sys.executable


def make_subset_bm25(ds: str, n: int) -> Path:
    """Write a BM25 JSON containing only the first ``n`` results."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    src = json.loads((BM25_DIR / f"{ds}-bm25.json").read_text())
    src["results"] = src["results"][:n]
    src.setdefault("metadata", {})["subset"] = {"first_n": n}
    dst = SUBSET_DIR / f"{ds}-bm25-first{n}.json"
    dst.write_text(json.dumps(src, indent=2))
    return dst


def run_rerank(ds: str, model: str, api_base: str) -> tuple[Path | None, float, int]:
    """Run rerank for one dataset. Returns (output_path, wall_seconds, n_queries)."""
    bm25_path = BM25_DIR / f"{ds}-bm25.json"
    instances_path = INSTANCES_DIR / f"{ds}.json"
    if not bm25_path.exists():
        print(f"[skip] {bm25_path} missing", flush=True)
        return None, 0.0, 0

    if SUBSET_PER_DATASET > 0:
        bm25_path = make_subset_bm25(ds, SUBSET_PER_DATASET)
        suffix = f"-subset{SUBSET_PER_DATASET}"
    else:
        suffix = ""
    out_path = OUTPUT_DIR / (
        f"{ds}-rerank-{model.replace(':', '_').replace('/', '_')}{suffix}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_queries = len(json.loads(bm25_path.read_text())["results"])

    cmd = [
        PY, "-m", "sragents.cli.main", "rerank",
        "--input", str(bm25_path),
        "--instances", str(instances_path),
        "--corpus", str(CORPUS_PATH),
        "--output", str(out_path),
        "--model", model,
        "--api-base", api_base,
        "--top-k", str(RERANK_TOP_K),
        "--workers", str(WORKERS),
    ]
    print(f"\n[{ds}] launching rerank ({n_queries} queries)", flush=True)
    print("  CMD: " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    per_q = dt / n_queries if n_queries else 0.0
    print(f"  rerank({ds}) exit={rc}  wall={dt/60:.1f}m  per_query={per_q*1000:.0f}ms", flush=True)
    if rc != 0 or not out_path.exists():
        return None, dt, n_queries
    return out_path, dt, n_queries


def load_gold(ds: str) -> dict:
    inst = json.loads((INSTANCES_DIR / f"{ds}.json").read_text())
    return {
        i.get("instance_id") or i.get("id"):
        i.get("skill_annotations") or i.get("gold_skill_ids") or []
        for i in inst
    }


def recompute_metrics(path: Path, ks=KS) -> dict:
    d = json.loads(path.read_text())
    results = d["results"]
    ds = path.stem.split("-")[0]
    gold_lookup = load_gold(ds)
    metrics = {}
    for k in ks:
        rec_sum = ndcg_sum = 0.0
        valid = 0
        for r in results:
            gold = gold_lookup.get(r["instance_id"], [])
            if not gold:
                continue
            retrieved = [x["skill_id"] for x in r["retrieved"][:k]]
            gset = set(gold)
            rec = len(set(retrieved) & gset) / len(gset)
            rec_sum += rec
            dcg = sum(1.0 / math.log2(i + 2)
                      for i, sid in enumerate(retrieved) if sid in gset)
            ideal = sum(1.0 / math.log2(i + 2)
                        for i in range(min(len(gset), k)))
            ndcg_sum += dcg / ideal if ideal else 0
            valid += 1
        if valid:
            metrics[f"Recall@{k}"] = rec_sum / valid
            metrics[f"nDCG@{k}"] = ndcg_sum / valid
    return metrics


def main():
    model = os.environ.get("MODEL", DEFAULT_MODEL)
    api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)
    datasets = sys.argv[1:] or DATASETS

    print(f"Model: {model}", flush=True)
    print(f"API:   {api_base}", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(f"Datasets: {datasets}", flush=True)

    summary = {}
    latency = {}  # ds -> {"wall_s": float, "n_queries": int, "per_query_ms": float}
    for ds in datasets:
        path, wall_s, n_q = run_rerank(ds, model=model, api_base=api_base)
        latency[ds] = {
            "wall_s": wall_s,
            "n_queries": n_q,
            "per_query_ms": (wall_s / n_q * 1000) if n_q else 0.0,
        }
        if path is None:
            continue
        metrics = recompute_metrics(path)
        summary[ds] = metrics
        # Persist re-computed metrics + latency back into the file
        d = json.loads(path.read_text())
        d["metrics"] = metrics
        d.setdefault("metadata", {})["recomputed_at"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        d["metadata"]["latency"] = latency[ds]
        path.write_text(json.dumps(d, indent=2))
        print(f"[{ds}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
        print(f"[{ds}] latency: wall={wall_s/60:.1f}m  per_query={latency[ds]['per_query_ms']:.0f}ms", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(f"SUMMARY: BM25 top-{RERANK_TOP_K} → {model} rerank", flush=True)
    print("=" * 110, flush=True)
    header = f"{'Dataset':<14}" + "".join(f" {'R@' + str(k):>9}" for k in KS) + \
             "".join(f" {'nDCG@' + str(k):>10}" for k in KS) + f" {'wall_min':>9} {'ms/query':>9}"
    print(header)
    print("-" * len(header))
    total_wall = 0.0
    total_q = 0
    for ds, m in summary.items():
        row = f"{ds:<14}"
        for k in KS:
            row += f" {m.get(f'Recall@{k}', 0):>9.4f}"
        for k in KS:
            row += f" {m.get(f'nDCG@{k}', 0):>10.4f}"
        lat = latency.get(ds, {})
        row += f" {lat.get('wall_s', 0)/60:>9.1f} {lat.get('per_query_ms', 0):>9.0f}"
        total_wall += lat.get('wall_s', 0)
        total_q += lat.get('n_queries', 0)
        print(row)
    if total_q:
        print(f"\nTOTAL: {total_q} queries  wall={total_wall/60:.1f}m  "
              f"avg={total_wall/total_q*1000:.0f}ms/query", flush=True)

    # Write a latency summary JSON for downstream comparison
    lat_path = OUTPUT_DIR / f"latency-{model.replace(':', '_').replace('/', '_')}.json"
    lat_path.write_text(json.dumps({
        "model": model,
        "workers": WORKERS,
        "api_base": api_base,
        "datasets": latency,
    }, indent=2))
    print(f"Saved latency summary: {lat_path}")


if __name__ == "__main__":
    main()
