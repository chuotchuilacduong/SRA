"""CE-Raw Phase B — fine-tune the bi-encoder retriever (full SkillRouter recipe).

SkillRouter's encoder stage: fine-tune a dense bi-encoder with InfoNCE + hard negatives
(here = MultipleNegativesRankingLoss over the same full-corpus-mined hard negatives from
Phase 1). Base = BAAI/bge-base-en-v1.5 (the retriever already used in this repo). Trains on
data/ce_raw/de_triples_train.jsonl (query, positive_id, negative_ids).

The fine-tuned encoder is saved to results/models/sr-emb-bge-v1/ and is consumed at eval
time (ceraw_eval.py --retrievers bge_ft) to retrieve top-D from the FULL corpus before
CE-Raw reranking.

Requires: sentence-transformers (>=2.2), torch. GPU (H100) strongly recommended — it
re-encodes triples each epoch. CPU works but is slow.

    python src/kmeans/scripts/ceraw_train_retriever.py --epochs 2 --batch-size 64 \
        --out results/models/sr-emb-bge-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict

DATA = PROJECT_ROOT / "data" / "ce_raw"


def skill_text(s: dict, cap: int = 2500) -> str:
    return f"{s.get('name','')} | {s.get('description','')} | {str(s.get('content',''))[:cap]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune bi-encoder (SkillRouter Phase B)")
    ap.add_argument("--base", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--triples", type=Path, default=DATA / "de_triples_train.jsonl")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/models/sr-emb-bge-v1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-negs", type=int, default=6, help="hard negatives per anchor in the InputExample")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    args = ap.parse_args()
    if not args.triples.exists():
        raise SystemExit(f"missing {args.triples} (run ceraw_build_dataset.py first)")

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    corpus = load_corpus_dict()
    examples = []
    with args.triples.open() as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            pos = corpus.get(t["positive_id"])
            if pos is None:
                continue
            negs = [corpus[n] for n in t["negative_ids"][:args.max_negs] if n in corpus]
            texts = [t["query"], skill_text(pos)] + [skill_text(n) for n in negs]
            if len(texts) >= 2:
                examples.append(InputExample(texts=texts))
    print(f"[retriever] {len(examples)} training anchors | base={args.base} -> {args.out}", flush=True)

    model = SentenceTransformer(args.base)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)  # in-batch InfoNCE + provided hard negs
    steps = max(1, len(loader) * args.epochs)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=int(steps * args.warmup_ratio),
              output_path=str(args.out), show_progress_bar=True)
    (args.out / "train_meta.json").write_text(json.dumps(
        {"base": args.base, "anchors": len(examples), "epochs": args.epochs,
         "batch_size": args.batch_size, "max_negs": args.max_negs,
         "loss": "MultipleNegativesRankingLoss"}, indent=2))
    print(f"[retriever] saved -> {args.out}")


if __name__ == "__main__":
    main()
