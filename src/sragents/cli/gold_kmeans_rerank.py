"""``sragents gold-kmeans-rerank`` — gold-anchored K-means reranking of BM25 results.

Clusters only gold (non-web) skills so centroids reflect actual answer
semantics.  Web skills are assigned post-hoc.  Scoring propagates from
each cluster centroid outward:

    affinity = cosine(query, centroid[c]) * cosine(skill_emb, centroid[c])
    final    = alpha * bm25_norm + (1 - alpha) * affinity_norm
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from sragents.cli._common import require_exists
from sragents.corpus import load_corpus, skill_text
from sragents.prompts import build_prompt
from sragents.retrieve import compute_retrieval_metrics
from sragents.retrieve.kmeans_rerank import GoldKMeansReranker
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "gold-kmeans-rerank",
        help="Gold-anchored K-means reranking of a retrieval result file (stage 1)",
        description=(
            "Clusters only gold skills so centroids are anchored to answer semantics. "
            "Web skills are assigned post-hoc to the nearest gold centroid. "
            "Scoring: affinity = cosine(query, centroid) * cosine(skill, centroid), "
            "then blended with BM25: alpha*bm25 + (1-alpha)*affinity."
        ),
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Input retrieval JSON (e.g. from sragents retrieve --retriever bm25)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output retrieval JSON (reranked)")
    p.add_argument("--instances", type=Path, required=True,
                   help="Instances JSON (to rebuild queries)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Corpus JSON (default: package default)")
    p.add_argument("--n-clusters", type=int, default=100,
                   help="Number of K-means clusters over gold skills (default: 100)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="BM25 weight in [0,1]; (1-alpha) goes to propagation affinity (default: 0.5)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of candidates to rerank per query (default: 50)")
    p.add_argument("--model-path", default="BAAI/bge-base-en-v1.5",
                   help="Sentence-transformer model for embeddings (default: BAAI/bge-base-en-v1.5)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Encoding batch size (default: 256)")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory to cache corpus embeddings and cluster assignments")
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.input, "input")
    require_exists(args.instances, "instances")
    if args.corpus is not None:
        require_exists(args.corpus, "corpus")

    print("Loading corpus...", flush=True)
    corpus_list = load_corpus(args.corpus)
    corpus_ids = [s["skill_id"] for s in corpus_list]
    corpus_texts = [skill_text(s) for s in corpus_list]
    n_gold = sum(1 for cid in corpus_ids if not cid.startswith("web_"))
    print(f"  {len(corpus_ids)} skills ({n_gold} gold + {len(corpus_ids) - n_gold} web)")

    source = json.loads(args.input.read_text())
    source_records = source["results"]
    instances = {
        i["instance_id"]: i
        for i in json.loads(args.instances.read_text())
    }

    print(
        f"Building gold K-means index: K={args.n_clusters}, alpha={args.alpha}...",
        flush=True,
    )
    reranker = GoldKMeansReranker(
        model_name_or_path=args.model_path,
        n_clusters=args.n_clusters,
        alpha=args.alpha,
        batch_size=args.batch_size,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    reranker.build_index(corpus_ids, corpus_texts)

    print(f"Reranking {len(source_records)} queries...", flush=True)
    records_out = []
    for entry in tqdm(source_records, desc="  rerank", file=sys.stderr):
        inst_id = entry["instance_id"]
        candidates = entry["retrieved"][: args.top_k]
        inst = instances.get(inst_id)

        if not candidates or inst is None:
            records_out.append({
                "instance_id": inst_id,
                "gold_skill_ids": entry["gold_skill_ids"],
                "retrieved": candidates,
            })
            continue

        _, query = build_prompt(inst)
        reranked = reranker.rerank(query, candidates)

        records_out.append({
            "instance_id": inst_id,
            "gold_skill_ids": entry["gold_skill_ids"],
            "retrieved": [{"skill_id": sid, "score": score} for sid, score in reranked],
        })

    metrics = compute_retrieval_metrics(records_out, top_k=args.top_k)
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    result = RetrievalResults(
        retriever=f"gold_kmeans_rerank_{source['metadata'].get('retriever', 'unknown')}",
        top_k=args.top_k,
        corpus_size=source["metadata"].get("corpus_size", len(corpus_ids)),
        records=[
            RetrievalRecord(
                instance_id=r["instance_id"],
                gold_skill_ids=r["gold_skill_ids"],
                retrieved=r["retrieved"],
            )
            for r in records_out
        ],
        metrics=metrics,
        dataset=source["metadata"].get("dataset"),
        extra={
            "source": str(args.input),
            "n_clusters": args.n_clusters,
            "alpha": args.alpha,
            "model_path": args.model_path,
            "gold_anchored": True,
        },
    )
    result.dump(args.output)
    print(f"Saved: {args.output}")
