"""``sragents train-rocketqav2`` — joint training of dual-encoder + cross-encoder.

Implements dynamic listwise distillation (RocketQAv2, Ren et al., EMNLP 2021):

    L = KL(p̃_de || p̃_ce) + cross_entropy(s_ce, positive_idx)

Both the dual-encoder retriever and the cross-encoder re-ranker are updated
jointly.  After training, models are saved as HuggingFace checkpoints:

    <output-dir>/dual_encoder   — use with: sragents retrieve --retriever bge
                                    --retriever-arg model_path=<dir>/dual_encoder
    <output-dir>/cross_encoder  — use with: sragents cross-encoder-rerank
                                    --model-path <dir>/cross_encoder
"""

import json
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.corpus import load_corpus, skill_text
from sragents.retrieve.rocketqav2_trainer import RocketQAv2Trainer


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train-rocketqav2",
        help="Joint training of dual-encoder + cross-encoder via listwise distillation (RocketQAv2)",
        description=(
            "Jointly fine-tunes a dual-encoder retriever and a cross-encoder re-ranker "
            "using dynamic listwise distillation (RocketQAv2).  "
            "Requires instances with gold skill_annotations and a pre-computed BM25 result "
            "file for hard-negative mining.  "
            "Loss = KL(p̃_DE || p̃_CE) + cross_entropy(s_CE, positive_idx)."
        ),
    )
    p.add_argument("--instances", type=Path, required=True,
                   help="Training instances JSON (must have skill_annotations field)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Corpus JSON (default: package default)")
    p.add_argument("--bm25-results", type=Path, required=True,
                   help="Pre-computed BM25 retrieval JSON used for hard-negative mining")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to save trained dual_encoder/ and cross_encoder/ checkpoints")
    p.add_argument("--de-model-path", default=RocketQAv2Trainer.DE_DEFAULT,
                   help=f"Dual-encoder init checkpoint "
                        f"(default: {RocketQAv2Trainer.DE_DEFAULT})")
    p.add_argument("--ce-model-path", default=RocketQAv2Trainer.CE_DEFAULT,
                   help=f"Cross-encoder init checkpoint "
                        f"(default: {RocketQAv2Trainer.CE_DEFAULT})")
    p.add_argument("--epochs", type=int, default=3,
                   help="Training epochs (default: 3)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Queries per gradient step (default: 8)")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate (default: 1e-5)")
    p.add_argument("--n-hard-neg", type=int, default=15,
                   help="Hard negatives per query (default: 15)")
    p.add_argument("--device", default=None,
                   help="Training device: cuda or cpu (default: auto-detect)")
    p.add_argument("--encode-sub-batch", type=int, default=4,
                   help="Passages encoded per sub-batch (limits peak VRAM, default: 4)")
    p.add_argument("--fp16", action="store_true", default=False,
                   help="Enable mixed-precision (fp16) forward pass via torch.autocast")
    p.add_argument("--de-max-length", type=int, default=None,
                   help="Max token length for dual-encoder (default: 256)")
    p.add_argument("--ce-max-length", type=int, default=None,
                   help="Max token length for cross-encoder (default: 256)")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print loss every N gradient steps (default: 50)")
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.instances, "instances")
    require_exists(args.bm25_results, "bm25-results")
    if args.corpus is not None:
        require_exists(args.corpus, "corpus")

    print("Loading corpus...", flush=True)
    corpus_list = load_corpus(args.corpus)
    corpus_texts = {s["skill_id"]: skill_text(s) for s in corpus_list}
    print(f"  {len(corpus_texts)} skills")

    print("Loading instances...", flush=True)
    instances = json.loads(args.instances.read_text())
    usable = [i for i in instances if i.get("skill_annotations")]
    print(f"  {len(usable)} / {len(instances)} instances have skill_annotations")

    print("Loading BM25 results...", flush=True)
    bm25_data = json.loads(args.bm25_results.read_text())
    bm25_lookup = {r["instance_id"]: r["retrieved"] for r in bm25_data["results"]}
    print(f"  {len(bm25_lookup)} entries")

    print("Initialising RocketQAv2 trainer...", flush=True)
    trainer = RocketQAv2Trainer(
        de_model_path=args.de_model_path,
        ce_model_path=args.ce_model_path,
        n_hard_negatives=args.n_hard_neg,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        encode_sub_batch=args.encode_sub_batch,
        fp16=args.fp16,
        de_max_length=args.de_max_length,
        ce_max_length=args.ce_max_length,
    )

    print(
        f"Training: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, n_hard_neg={args.n_hard_neg}, "
        f"sub_batch={args.encode_sub_batch}, fp16={args.fp16}",
        flush=True,
    )
    trainer.train(
        instances=usable,
        corpus_texts=corpus_texts,
        bm25_lookup=bm25_lookup,
        epochs=args.epochs,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )

    print(f"\nDone.  Models saved to: {args.output_dir}")
    print(f"  Dual-encoder:  {args.output_dir}/dual_encoder")
    print(f"  Cross-encoder: {args.output_dir}/cross_encoder")
    print()
    print("To use trained models:")
    print(f"  sragents retrieve --retriever bge "
          f"--retriever-arg model_path={args.output_dir}/dual_encoder ...")
    print(f"  sragents cross-encoder-rerank "
          f"--model-path {args.output_dir}/cross_encoder ...")
