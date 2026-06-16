"""Stage 10 — train CE-ProbeHYRR from the built dataset.

Glue between the dataset pipeline output (`utility_pairs_{split}.jsonl`) and the existing
``sragents.train.train_cross_encoder`` trainer. The two formats are almost identical; the
only real adaptation is the label key (``label_binary`` -> ``label``). The dev signal and
the corpus/instance lookups the trainer needs for its nDCG early-stop are rebuilt from the
M4 cache + corpus.

Notes
-----
* The trainer ONLY writes a checkpoint when a dev metric improves, so a dev pool is
  REQUIRED — without it nothing is ever saved. We build the dev pool from the M4 dev cache
  (re-rank the top-50 and score nDCG@10).
* ``utility_pairs`` already drops the ``ambiguous``/``unverifiable`` rows and carries the
  collapsed binary label, so no extra filtering is needed here.
* Default loss is ``bce`` (decision D2 "binary first"). For grouped ranking use
  ``--loss listwise`` (implies ``--group-by-query``); ~half the groups have no positive and
  contribute zero to the listwise objective, which is expected for the no-load/classification
  groups.

Usage
-----
    python -m sragents.probehyrr_h100.train_ce_probehyrr \
        --output-dir results/ce_probehyrr_h100/model_v0 \
        --loss bce --epochs 3
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl
from sragents.train.train_cross_encoder import TrainConfig, train_cross_encoder

log = logging.getLogger(__name__)


def load_train_pairs(split: str) -> list[dict]:
    """``utility_pairs`` rows -> trainer pairs. Only the label key differs."""
    pairs: list[dict] = []
    for r in iter_jsonl(C.utility_pairs_path(split)):
        pairs.append({
            "instance_id": r["instance_id"],
            "dataset": r["dataset"],
            "question": r["question"],
            "skill": r["skill"],
            "label": float(r["label_binary"]),
        })
    return pairs


def load_dev_eval(split: str) -> tuple[list[dict], dict]:
    """Build (dev_pool_records, instances_by_id) from the M4 cache for the nDCG early-stop.

    The trainer's dev re-ranker scores every candidate in the pool, so we expose the full
    M4 top-50 (not just the probed ones) and look each query's text up by ``instance_id``.
    """
    pool_records: list[dict] = []
    instances: dict[str, dict] = {}
    for r in iter_jsonl(C.m4_cache_path(split)):
        qid = r["qid"]
        instances[qid] = {"question": r["query"], "dataset": r["dataset"]}
        pool_records.append({
            "instance_id": qid,
            "gold_skill_ids": r["gold_skill_ids"],
            "retrieved": [{"skill_id": c["skill_id"]} for c in r["top50"]],
        })
    return pool_records, instances


def load_corpus_jsonl(path: Path) -> dict[str, dict]:
    """Corpus keyed by skill_id. (``sragents.corpus.load_corpus`` expects a JSON array;
    the pipeline's canonical corpus is line-delimited, so load it directly.)"""
    return {s["skill_id"]: s for s in iter_jsonl(path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CE-ProbeHYRR from utility_pairs")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--dev-split", default="dev")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-model", default=TrainConfig.base_model)
    ap.add_argument("--loss", default="bce", choices=["bce", "pairwise", "listwise"])
    ap.add_argument("--group-by-query", action="store_true",
                    help="bundle all pairs of N queries per batch (auto-on for listwise)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--queries-per-batch", type=int, default=8)
    ap.add_argument("--max-content-chars", type=int, default=1800)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--early-stop-metric", default="dev_ndcg@10")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    train_pairs = load_train_pairs(args.train_split)
    dev_pool_records, instances_by_id = load_dev_eval(args.dev_split)
    corpus = load_corpus_jsonl(C.OUT_CORPUS_JSONL)

    group_by_query = args.group_by_query or args.loss == "listwise"
    config = TrainConfig(
        base_model=args.base_model,
        loss_name=args.loss,
        group_by_query=group_by_query,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        queries_per_batch=args.queries_per_batch,
        max_content_chars=args.max_content_chars,
        max_length=args.max_length,
        early_stop_metric=args.early_stop_metric,
    )

    n_pos = sum(1 for p in train_pairs if p["label"] > 0.5)
    print(f"[train] train_split={args.train_split} pairs={len(train_pairs)} "
          f"pos={n_pos} neg={len(train_pairs) - n_pos} | loss={args.loss} "
          f"group_by_query={group_by_query} | dev_split={args.dev_split} "
          f"dev_queries={len(dev_pool_records)} corpus={len(corpus)}")

    summary = train_cross_encoder(
        train_pairs=train_pairs,
        dev_pool_records=dev_pool_records,
        corpus=corpus,
        instances_by_id=instances_by_id,
        config=config,
        output_dir=Path(args.output_dir),
    )
    print(f"[train] done best_epoch={summary['best_epoch']} "
          f"best({config.early_stop_metric})={summary['best_score']:.4f} "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
