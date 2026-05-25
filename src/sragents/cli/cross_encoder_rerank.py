"""``sragents cross-encoder-rerank`` — cross-encoder reranking of BM25 results.

Reads a retrieval JSON produced by ``sragents retrieve``, re-scores each
query's candidates by running every (query, passage) pair through a
cross-encoder, and writes a new retrieval JSON in the same schema.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from sragents.cli._common import require_exists
from sragents.corpus import load_corpus, skill_text
from sragents.prompts import build_prompt
from sragents.retrieve import compute_retrieval_metrics
from sragents.retrieve.cross_encoder_rerank import CrossEncoderReranker
from sragents.retrieve.schema import RetrievalRecord, RetrievalResults


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "cross-encoder-rerank",
        help="Cross-encoder reranking of a retrieval result file (stage 1)",
        description=(
            "Re-scores BM25 candidates by running each (query, passage) pair "
            "through a cross-encoder model.  Default: pure cross-encoder ranking "
            "(alpha=0).  Set --alpha > 0 to blend with the original BM25 score: "
            "final = (1-alpha)*cross_encoder + alpha*bm25."
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
    p.add_argument("--model-path", default=CrossEncoderReranker.DEFAULT_MODEL,
                   help=f"HuggingFace cross-encoder model "
                        f"(default: {CrossEncoderReranker.DEFAULT_MODEL})")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Inference batch size per query (default: 32)")
    p.add_argument("--max-length", type=int, default=512,
                   help="Max token length for the cross-encoder (default: 512)")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="BM25 blend weight in [0,1]; 0=pure cross-encoder (default: 0.0)")
    p.add_argument("--top-k", type=int, default=50,
                   help="Number of candidates to rerank per query (default: 50)")
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
    print(f"  {len(corpus_ids)} skills")

    source = json.loads(args.input.read_text())
    source_records = source["results"]
    instances = {
        i["instance_id"]: i
        for i in json.loads(args.instances.read_text())
    }

    print(
        f"Loading cross-encoder: {args.model_path} "
        f"(alpha={args.alpha}, max_length={args.max_length})...",
        flush=True,
    )
    reranker = CrossEncoderReranker(
        model_name_or_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        alpha=args.alpha,
    )
    reranker.build_index(corpus_ids, corpus_texts)
    reranker._load_model()

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

    source_retriever = source["metadata"].get("retriever", "unknown")
    result = RetrievalResults(
        retriever=f"cross_encoder_{source_retriever}",
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
            "model_path": args.model_path,
            "alpha": args.alpha,
            "max_length": args.max_length,
        },
    )
    result.dump(args.output)
    print(f"Saved: {args.output}")
