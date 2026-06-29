"""Fine-tune a cross-encoder on (question, packed_skill) pairs.

The trainer is intentionally small and standalone — we do not depend on
sentence-transformers training API because we need control over the
packer/chunker and the bookkeeping for ``query_ids`` (needed by pairwise
and listwise losses).

Device policy:

* CUDA available -> cuda + fp16 (config opt-in).
* MPS available -> **disabled** (documented unstable for CE forward).
* otherwise -> CPU.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from sragents.retrieve.skill_packer import SkillPacker
from sragents.retrieve.metrics import compute_retrieval_metrics
from sragents.train.losses import get_loss

log = logging.getLogger(__name__)


# ------------------------------------------------------------ device


def pick_device(prefer_cuda: bool = True) -> torch.device:
    """CUDA > CPU. MPS is opt-in via env (SRA_ALLOW_MPS=1) for local Mac runs.

    Default behaviour is unchanged (CUDA on H100, CPU otherwise). Set
    ``SRA_ALLOW_MPS=1`` to use Apple-GPU acceleration locally; fp16 stays off on
    MPS (CUDA-only here), so the forward/backward runs in fp32.
    """
    import os
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("SRA_ALLOW_MPS") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------ data


@dataclass
class TrainConfig:
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    fp16_if_cuda: bool = True
    grad_accum_steps: int = 1
    loss_name: str = "bce"
    packing_mode: str = "field_tagged"
    include_tools: bool = False
    max_content_chars: int = 1800
    early_stop_metric: str = "dev_ndcg@10"
    early_stop_patience: int = 1
    balanced_by_dataset: bool = False
    group_by_query: bool = False
    queries_per_batch: int = 3

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(
            base_model=cfg["model"]["base"],
            max_length=cfg["model"]["max_length"],
            batch_size=cfg["trainer"]["batch_size"],
            lr=float(cfg["trainer"]["lr"]),
            epochs=cfg["trainer"]["epochs"],
            weight_decay=cfg["trainer"].get("weight_decay", 0.01),
            warmup_ratio=cfg["trainer"].get("warmup_ratio", 0.1),
            seed=cfg["trainer"].get("seed", 42),
            fp16_if_cuda=cfg["trainer"].get("fp16_if_cuda", True),
            grad_accum_steps=cfg["trainer"].get("grad_accum_steps", 1),
            loss_name=cfg.get("loss", "bce"),
            packing_mode=cfg["packing"]["mode"],
            include_tools=cfg["packing"].get("include_tools", False),
            max_content_chars=cfg["packing"].get("max_content_chars", 1800),
            early_stop_metric=cfg["trainer"].get("early_stop_metric", "dev_ndcg@10"),
            early_stop_patience=cfg["trainer"].get("early_stop_patience", 1),
            balanced_by_dataset=cfg["trainer"].get("balanced_by_dataset", False),
            group_by_query=cfg["trainer"].get("group_by_query", False),
            queries_per_batch=cfg["trainer"].get("queries_per_batch", 3),
        )


class PairsDataset(Dataset):
    """Holds ``(question, skill, label, instance_id)`` rows.

    The query-id is encoded as a Python int for fast torch.tensor packing
    in the collator — we keep a parallel string list for debugging.
    """

    def __init__(self, pairs: list[dict], packer: SkillPacker):
        self.pairs = pairs
        self.packer = packer
        self._qid_to_int: dict[str, int] = {}
        for p in pairs:
            qid = p["instance_id"]
            if qid not in self._qid_to_int:
                self._qid_to_int[qid] = len(self._qid_to_int)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        skill_text = self.packer.pack_skill(p["skill"])
        return {
            "question": p["question"],
            "skill_text": skill_text,
            "label": float(p["label"]),
            "query_id": self._qid_to_int[p["instance_id"]],
        }


class QueryGroupedBatchSampler(Sampler):
    """Each batch contains *all* pairs from N queries.

    Needed by listwise softmax loss: a query's positive vs. its hard
    negatives must coexist in the same forward pass.

    Args:
        pairs:               The full list of training pairs.
        queries_per_batch:   How many queries to bundle per batch.
        balanced_by_dataset: If True, sample queries with inverse-dataset-size
                             weights (uniform marginal per dataset).
        seed:                RNG seed; the iter() is deterministic per call.
    """

    def __init__(
        self,
        pairs: list[dict],
        *,
        queries_per_batch: int = 3,
        balanced_by_dataset: bool = False,
        seed: int = 42,
    ):
        self.pairs = pairs
        self.queries_per_batch = queries_per_batch
        self.balanced_by_dataset = balanced_by_dataset
        self.seed = seed

        groups: dict[str, list[int]] = {}
        for idx, p in enumerate(pairs):
            groups.setdefault(p["instance_id"], []).append(idx)
        self.groups = list(groups.values())
        self.query_dataset = [
            pairs[idxs[0]].get("dataset", "_") for idxs in self.groups
        ]
        if balanced_by_dataset:
            ds_counts: dict[str, int] = {}
            for ds in self.query_dataset:
                ds_counts[ds] = ds_counts.get(ds, 0) + 1
            self.query_weights = np.array(
                [1.0 / ds_counts[ds] for ds in self.query_dataset], dtype=np.float64,
            )
            self.query_weights /= self.query_weights.sum()
        else:
            self.query_weights = None

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        n_queries = len(self.groups)
        if self.query_weights is not None:
            order = rng.choice(
                n_queries, size=n_queries, replace=True, p=self.query_weights,
            )
        else:
            order = rng.permutation(n_queries)
        for start in range(0, len(order), self.queries_per_batch):
            chosen = order[start : start + self.queries_per_batch]
            batch_idxs: list[int] = []
            for q in chosen:
                batch_idxs.extend(self.groups[q])
            if batch_idxs:
                yield batch_idxs

    def __len__(self) -> int:
        return (len(self.groups) + self.queries_per_batch - 1) // self.queries_per_batch


def _make_collate(tokenizer, max_length: int):
    def collate(batch: list[dict]) -> dict:
        questions = [b["question"] for b in batch]
        skills = [b["skill_text"] for b in batch]
        enc = tokenizer(
            questions, skills,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        qids = torch.tensor([b["query_id"] for b in batch], dtype=torch.long)
        return {**enc, "labels": labels, "query_ids": qids}
    return collate


# ------------------------------------------------------------ eval helper


def _score_pool_for_eval(
    model,
    tokenizer,
    packer: SkillPacker,
    device: torch.device,
    pool_records: list[dict],
    corpus: dict[str, dict],
    instances_by_id: dict[str, dict],
    batch_size: int = 64,
    max_length: int = 256,
    question_field: str = "question",
    top_k: int = 100,
) -> list[dict]:
    """Re-rank the pool entries with the current model. Returns RetrievalResults-shaped records."""
    model.eval()
    out: list[dict] = []
    with torch.no_grad():
        for rec in pool_records:
            inst = instances_by_id.get(rec["instance_id"])
            if inst is None:
                continue
            question = inst.get(question_field) or ""
            cands = rec.get("retrieved", [])[:top_k]
            if not cands:
                out.append({"instance_id": rec["instance_id"],
                            "gold_skill_ids": rec["gold_skill_ids"],
                            "retrieved": []})
                continue
            pairs_q, pairs_d = [], []
            valid: list[dict] = []
            for c in cands:
                sk = corpus.get(c["skill_id"])
                if sk is None:
                    continue
                pairs_q.append(question)
                pairs_d.append(packer.pack_skill(sk))
                valid.append(c)
            scores: list[float] = []
            for i in range(0, len(pairs_q), batch_size):
                enc = tokenizer(
                    pairs_q[i : i + batch_size],
                    pairs_d[i : i + batch_size],
                    padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits.squeeze(-1)
                scores.extend(logits.detach().float().cpu().tolist())
            ranked = sorted(zip(valid, scores), key=lambda x: -x[1])
            retrieved = [{"skill_id": c["skill_id"], "score": float(s), "rank": i + 1}
                         for i, (c, s) in enumerate(ranked)]
            out.append({"instance_id": rec["instance_id"],
                        "gold_skill_ids": rec["gold_skill_ids"],
                        "retrieved": retrieved})
    return out


# ------------------------------------------------------------ trainer


def train_cross_encoder(
    *,
    train_pairs: list[dict],
    dev_pool_records: list[dict] | None,
    corpus: dict[str, dict],
    instances_by_id: dict[str, dict],
    config: TrainConfig,
    output_dir: Path,
    log_every: int = 25,
) -> dict:
    """End-to-end training entry. Returns a small history dict.

    Only the best checkpoint (highest ``early_stop_metric`` on dev) is saved.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(config.seed)
    device = pick_device()
    use_fp16 = device.type == "cuda" and config.fp16_if_cuda
    log.info("device=%s fp16=%s", device, use_fp16)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, num_labels=1,
    ).to(device)

    packer = SkillPacker(
        mode=config.packing_mode,
        include_tools=config.include_tools,
        max_content_chars=config.max_content_chars,
    )
    dataset = PairsDataset(train_pairs, packer)

    # Sampler selection:
    #   - group_by_query=True: each batch contains all pairs from N queries.
    #     Required by listwise loss; works with balanced_by_dataset.
    #   - balanced_by_dataset only: WeightedRandomSampler per pair.
    #   - neither: plain shuffle.
    sampler = None
    batch_sampler = None
    shuffle = True
    if config.group_by_query:
        if config.balanced_by_dataset and not all("dataset" in p for p in train_pairs):
            raise ValueError(
                "balanced_by_dataset=True requires every pair to carry a 'dataset' field."
            )
        batch_sampler = QueryGroupedBatchSampler(
            train_pairs,
            queries_per_batch=config.queries_per_batch,
            balanced_by_dataset=config.balanced_by_dataset,
            seed=config.seed,
        )
        shuffle = False
        log.info("query-grouped batch sampler: queries_per_batch=%d balanced=%s "
                 "n_queries=%d n_batches=%d",
                 config.queries_per_batch, config.balanced_by_dataset,
                 len(batch_sampler.groups), len(batch_sampler))
    elif config.balanced_by_dataset:
        if not all("dataset" in p for p in train_pairs):
            raise ValueError(
                "balanced_by_dataset=True requires every pair to carry a "
                "'dataset' field (set by strategy2_v2_prepare.py)."
            )
        counts: dict[str, int] = {}
        for p in train_pairs:
            counts[p["dataset"]] = counts.get(p["dataset"], 0) + 1
        weights = torch.tensor(
            [1.0 / counts[p["dataset"]] for p in train_pairs],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_pairs), replacement=True,
        )
        shuffle = False
        log.info("balanced-by-dataset sampler enabled: %s", counts)

    if batch_sampler is not None:
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            collate_fn=_make_collate(tokenizer, config.max_length),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler,
            collate_fn=_make_collate(tokenizer, config.max_length),
            num_workers=0,
        )

    loss_fn = get_loss(config.loss_name)
    optim = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(loader) * config.epochs // max(1, config.grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []
    # Per-step train loss history for convergence plots.
    step_loss_history: list[dict] = []
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        step = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            qids = batch.pop("query_ids")
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(**batch).logits.squeeze(-1)
                    loss = loss_fn(logits, labels, qids) / config.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits = model(**batch).logits.squeeze(-1)
                loss = loss_fn(logits, labels, qids) / config.grad_accum_steps
                loss.backward()
            if (step + 1) % config.grad_accum_steps == 0:
                if use_fp16:
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                scheduler.step()
                optim.zero_grad()
            step_loss_val = float(loss.detach()) * config.grad_accum_steps
            running += step_loss_val
            step += 1
            global_step += 1
            # Record EVERY step's loss for plotting (cheap).
            step_loss_history.append({
                "global_step": global_step, "epoch": epoch, "step": step,
                "loss": step_loss_val,
            })
            if step % log_every == 0:
                log.info("epoch=%d step=%d/%d loss=%.4f",
                         epoch, step, len(loader), running / max(1, step))

        dev_metrics: dict[str, float] = {}
        if dev_pool_records:
            scored = _score_pool_for_eval(
                model, tokenizer, packer, device,
                dev_pool_records, corpus, instances_by_id,
                batch_size=max(8, config.batch_size),
                max_length=config.max_length,
            )
            dev_metrics = compute_retrieval_metrics(scored, top_k=100)
            log.info("dev epoch=%d %s", epoch,
                     "  ".join(f"{k}={v:.4f}" for k, v in dev_metrics.items()))

        history.append({
            "epoch": epoch, "loss": running / max(1, step),
            "dev_metrics": dev_metrics,
        })

        score = dev_metrics.get(_metric_key(config.early_stop_metric), float("-inf"))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "train_config.json").write_text(
                json.dumps(config.__dict__, indent=2)
            )
        else:
            no_improve += 1
            if no_improve > config.early_stop_patience:
                log.info("early-stop after epoch=%d (no improvement on %s)",
                         epoch, config.early_stop_metric)
                break

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "early_stop_metric": config.early_stop_metric,
        "history": history,
        "step_loss_history": step_loss_history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _metric_key(name: str) -> str:
    """Translate ``dev_ndcg@10`` from config to ``nDCG@10`` used by metrics."""
    n = name.lower().replace("dev_", "")
    table = {
        "ndcg@10": "nDCG@10", "ndcg@5": "nDCG@5", "ndcg@1": "nDCG@1",
        "ndcg@50": "nDCG@50", "ndcg@100": "nDCG@100",
        "recall@1": "Recall@1", "recall@5": "Recall@5",
        "recall@10": "Recall@10", "recall@50": "Recall@50",
        "recall@100": "Recall@100",
        "hit@1": "Hit@1", "hit@10": "Hit@10",
        "mrr@10": "MRR@10", "mrr@100": "MRR@100",
        "p@1": "P@1", "p@5": "P@5", "p@10": "P@10",
    }
    return table.get(n, name)
