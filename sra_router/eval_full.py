"""Full-benchmark evaluation + inference timing.

Runs the three retrievers (BM25, BGE-small zero-shot, fine-tuned SR-Emb) on
the full 26,262-skill corpus and all 5,400 instances of SRA-Bench, then
reports R@1 / R@10 / nDCG@10 per source dataset and per-query latency.

  python -m sra_router.eval_full \
      --sr-emb results/sra_router/run1/encoder/encoder \
      --output-dir results/sra_router/full_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from sra_router.corpus import SkillCorpus
from sra_router.encoder import SkillEncoder, pick_device
from sra_router.instances import SRAInstance, load_instances
from sra_router.metrics import aggregate, ndcg_at_k, recall_at_k

ALL_DATASETS = ("theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench")


def per_query_metrics(ranked_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    return {
        "Recall@1":  recall_at_k(ranked_ids, gold_ids, 1),
        "Recall@5":  recall_at_k(ranked_ids, gold_ids, 5),
        "Recall@10": recall_at_k(ranked_ids, gold_ids, 10),
        "Recall@50": recall_at_k(ranked_ids, gold_ids, 50),
        "nDCG@10":   ndcg_at_k(ranked_ids, gold_ids, 10),
    }


def time_indexing_bm25(corpus: SkillCorpus):
    """Build BM25 index and return (retriever, build_time_seconds)."""
    from sragents.retrieve.bm25 import BM25Retriever
    t0 = time.perf_counter()
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    r = BM25Retriever()
    r.build_index(ids, texts)
    return r, time.perf_counter() - t0


def time_indexing_dense(corpus: SkillCorpus, model: SentenceTransformer,
                       *, batch_size: int = 128) -> tuple[np.ndarray, list[str], float]:
    t0 = time.perf_counter()
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    return embs, ids, time.perf_counter() - t0


def time_indexing_sr_emb(corpus: SkillCorpus, enc: SkillEncoder,
                         *, batch_size: int = 128) -> tuple[np.ndarray, list[str], float]:
    t0 = time.perf_counter()
    ids = corpus.ids()
    texts = [corpus[sid].full_text() for sid in ids]
    embs = enc.encode_eval(texts, is_query=False, batch_size=batch_size).cpu().numpy()
    return embs, ids, time.perf_counter() - t0


def time_bm25_retrieval(
    retriever, instances: list[SRAInstance], top_k: int,
) -> tuple[list[list[tuple[str, float]]], float, float]:
    """Return (results, batch_time, single_query_time_avg).

    ``single_query_time_avg`` measures wall-time per query when called one at a
    time on a 50-query sample (warm-up: 5 queries first).
    """
    queries = [inst.query for inst in instances]

    t0 = time.perf_counter()
    batch_results = retriever.retrieve(queries, top_k=top_k)
    batch_time = time.perf_counter() - t0

    # Warm-up
    sample = queries[:5]
    retriever.retrieve(sample, top_k=top_k)
    # Per-query timing (50 single-query calls)
    sample = queries[:50] if len(queries) >= 50 else queries
    t0 = time.perf_counter()
    for q in sample:
        retriever.retrieve([q], top_k=top_k)
    single_avg = (time.perf_counter() - t0) / len(sample)
    return batch_results, batch_time, single_avg


def time_dense_retrieval(
    encode_fn: Callable[[list[str]], np.ndarray],
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    instances: list[SRAInstance],
    top_k: int,
) -> tuple[list[list[tuple[str, float]]], float, float]:
    queries = [inst.query for inst in instances]
    t0 = time.perf_counter()
    q_emb = encode_fn(queries)
    sims = q_emb @ corpus_emb.T
    results = []
    for i in range(len(queries)):
        top = np.argpartition(-sims[i], range(top_k))[:top_k]
        top = top[np.argsort(-sims[i, top])]
        results.append([(corpus_ids[j], float(sims[i, j])) for j in top])
    batch_time = time.perf_counter() - t0

    # Warm-up + per-query single-call timing
    encode_fn(queries[:5])
    sample = queries[:50] if len(queries) >= 50 else queries
    t0 = time.perf_counter()
    for q in sample:
        emb = encode_fn([q])
        s = emb @ corpus_emb.T
        idx = np.argpartition(-s[0], range(top_k))[:top_k]
        idx = idx[np.argsort(-s[0, idx])]
    single_avg = (time.perf_counter() - t0) / len(sample)
    return results, batch_time, single_avg


def score_results(
    results: list[list[tuple[str, float]]],
    instances: list[SRAInstance],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    per_q = []
    by_ds: dict[str, list[dict[str, float]]] = {}
    for inst, ranked in zip(instances, results):
        ids = [sid for sid, _ in ranked]
        m = per_query_metrics(ids, inst.gold_skill_ids)
        per_q.append(m)
        by_ds.setdefault(inst.dataset, []).append(m)
    return aggregate(per_q), {ds: aggregate(rows) for ds, rows in sorted(by_ds.items())}


def write_markdown(report: dict, out: Path) -> None:
    lines: list[str] = []
    lines.append("# SRA-Skill-Router — Full SRA-Bench Evaluation\n")
    lines.append(f"_Generated: {report['timestamp']}_  \n")
    lines.append(f"_Corpus: {report['corpus_size']} skills | "
                 f"Instances: {report['total_instances']} across {report['n_datasets']} datasets_\n\n")

    lines.append("## R@1 và R@10 trên từng dataset\n")
    lines.append("| Dataset | N | BM25 R@1 | BM25 R@10 | BGE-small R@1 | BGE-small R@10 | SR-Emb R@1 | SR-Emb R@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in report["datasets"]:
        n = report["counts"][ds]
        row = [f"**{ds}**", str(n)]
        for method in ("BM25", "BGE-small-zero-shot", "SR-Emb-finetuned"):
            m = report["per_dataset"][method][ds]
            row.append(f"{m['Recall@1']:.3f}")
            row.append(f"{m['Recall@10']:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # Overall row
    row = ["**overall**", str(report["total_instances"])]
    for method in ("BM25", "BGE-small-zero-shot", "SR-Emb-finetuned"):
        m = report["overall"][method]
        row.append(f"**{m['Recall@1']:.3f}**")
        row.append(f"**{m['Recall@10']:.3f}**")
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Đầy đủ metric overall\n")
    lines.append("| Method | R@1 | R@5 | R@10 | R@50 | nDCG@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in ("BM25", "BGE-small-zero-shot", "SR-Emb-finetuned"):
        m = report["overall"][method]
        lines.append(f"| {method} | {m['Recall@1']:.3f} | {m['Recall@5']:.3f} | "
                     f"{m['Recall@10']:.3f} | {m['Recall@50']:.3f} | {m['nDCG@10']:.3f} |")
    lines.append("")

    lines.append("## Inference time\n")
    lines.append("**Index build** (one-off cost — load + encode toàn 26,262 skill):\n")
    lines.append("| Method | Index build (s) |")
    lines.append("|---|---:|")
    for method, t in report["index_build_s"].items():
        lines.append(f"| {method} | {t:.2f} |")
    lines.append("")
    lines.append(f"**Retrieval latency** (top-50, đo trên {report['total_instances']} query):\n")
    lines.append("| Method | Batch total (s) | Batch per query (ms) | Single-query avg (ms) | QPS (batch) |")
    lines.append("|---|---:|---:|---:|---:|")
    n = report["total_instances"]
    for method in ("BM25", "BGE-small-zero-shot", "SR-Emb-finetuned"):
        t_batch = report["batch_retrieval_s"][method]
        t_single = report["single_query_s"][method]
        per_q_batch_ms = t_batch / n * 1000
        per_q_single_ms = t_single * 1000
        qps = n / t_batch
        lines.append(f"| {method} | {t_batch:.2f} | {per_q_batch_ms:.2f} | "
                     f"{per_q_single_ms:.2f} | {qps:.1f} |")
    lines.append("")
    lines.append(f"_Hardware: {report['device']}._\n")

    out.write_text("\n".join(lines))
    print(f"Saved markdown → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bench/corpus/corpus.json")
    ap.add_argument("--instances-dir", default="data/bench/instances")
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS))
    ap.add_argument("--sr-emb", required=True,
                    help="Path to fine-tuned SR-Emb checkpoint dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {args.corpus} ...")
    corpus = SkillCorpus.from_json(args.corpus)
    print(f"  Corpus size: {len(corpus)}")

    all_inst: list[SRAInstance] = []
    counts: dict[str, int] = {}
    for ds in args.datasets:
        inst = load_instances(Path(args.instances_dir) / f"{ds}.json")
        all_inst.extend(inst)
        counts[ds] = len(inst)
    print(f"  Instances: {len(all_inst)} | per dataset: {counts}")

    report: dict = {
        "corpus_size": len(corpus),
        "total_instances": len(all_inst),
        "n_datasets": len(args.datasets),
        "datasets": args.datasets,
        "counts": counts,
        "top_k": args.top_k,
        "device": str(pick_device(args.device)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_build_s": {},
        "batch_retrieval_s": {},
        "single_query_s": {},
        "overall": {},
        "per_dataset": {},
    }

    # ----- BM25 -----
    print("\n=== BM25 ===")
    bm25, t_idx = time_indexing_bm25(corpus)
    report["index_build_s"]["BM25"] = t_idx
    print(f"  Index build: {t_idx:.2f}s")
    results, t_batch, t_single = time_bm25_retrieval(bm25, all_inst, args.top_k)
    report["batch_retrieval_s"]["BM25"] = t_batch
    report["single_query_s"]["BM25"] = t_single
    overall, by_ds = score_results(results, all_inst)
    report["overall"]["BM25"] = overall
    report["per_dataset"]["BM25"] = by_ds
    print(f"  Batch retrieval: {t_batch:.2f}s | Single-query avg: {t_single*1000:.2f}ms")
    print(f"  Overall: R@1={overall['Recall@1']:.4f}  R@10={overall['Recall@10']:.4f}  "
          f"nDCG@10={overall['nDCG@10']:.4f}")

    # ----- BGE-small zero-shot -----
    print("\n=== BGE-small zero-shot ===")
    device = pick_device(args.device)
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device=str(device))
    bge_corpus_emb, bge_ids, t_idx = time_indexing_dense(corpus, bge)
    report["index_build_s"]["BGE-small-zero-shot"] = t_idx
    print(f"  Index build: {t_idx:.2f}s")
    bge_prefix = "Represent this sentence for searching relevant passages: "

    def bge_encode(qs: list[str]) -> np.ndarray:
        return bge.encode([bge_prefix + q for q in qs], batch_size=128,
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False)

    results, t_batch, t_single = time_dense_retrieval(
        bge_encode, bge_corpus_emb, bge_ids, all_inst, args.top_k,
    )
    report["batch_retrieval_s"]["BGE-small-zero-shot"] = t_batch
    report["single_query_s"]["BGE-small-zero-shot"] = t_single
    overall, by_ds = score_results(results, all_inst)
    report["overall"]["BGE-small-zero-shot"] = overall
    report["per_dataset"]["BGE-small-zero-shot"] = by_ds
    print(f"  Batch retrieval: {t_batch:.2f}s | Single-query avg: {t_single*1000:.2f}ms")
    print(f"  Overall: R@1={overall['Recall@1']:.4f}  R@10={overall['Recall@10']:.4f}  "
          f"nDCG@10={overall['nDCG@10']:.4f}")
    del bge

    # ----- SR-Emb fine-tuned -----
    print(f"\n=== SR-Emb fine-tuned ({args.sr_emb}) ===")
    sr = SkillEncoder(model_name=args.sr_emb, device=device)
    sr.eval()
    sr_corpus_emb, sr_ids, t_idx = time_indexing_sr_emb(corpus, sr)
    report["index_build_s"]["SR-Emb-finetuned"] = t_idx
    print(f"  Index build: {t_idx:.2f}s")

    def sr_encode(qs: list[str]) -> np.ndarray:
        return sr.encode_eval(qs, is_query=True, batch_size=128).cpu().numpy()

    results, t_batch, t_single = time_dense_retrieval(
        sr_encode, sr_corpus_emb, sr_ids, all_inst, args.top_k,
    )
    report["batch_retrieval_s"]["SR-Emb-finetuned"] = t_batch
    report["single_query_s"]["SR-Emb-finetuned"] = t_single
    overall, by_ds = score_results(results, all_inst)
    report["overall"]["SR-Emb-finetuned"] = overall
    report["per_dataset"]["SR-Emb-finetuned"] = by_ds
    print(f"  Batch retrieval: {t_batch:.2f}s | Single-query avg: {t_single*1000:.2f}ms")
    print(f"  Overall: R@1={overall['Recall@1']:.4f}  R@10={overall['Recall@10']:.4f}  "
          f"nDCG@10={overall['nDCG@10']:.4f}")

    (out_dir / "full_eval.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved JSON → {out_dir / 'full_eval.json'}")
    write_markdown(report, out_dir / "full_eval.md")


if __name__ == "__main__":
    main()
