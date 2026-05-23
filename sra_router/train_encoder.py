"""Phase 3 — bi-encoder training loop.

Logs per-step loss + grad norm to a JSONL and periodically dumps eval R@K on a
small validation subset to track convergence.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from sra_router.corpus import SkillCorpus
from sra_router.encoder import SkillEncoder, pick_device
from sra_router.instances import SRAInstance
from sra_router.losses import info_nce_with_hard_negs
from sra_router.metrics import ndcg_at_k, recall_at_k


@dataclass
class TrainConfig:
    base_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 4
    n_hard_negs: int = 4              # cut from 10 to 4 to fit on CPU/MPS
    max_steps: int = 300
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    tau: float = 0.05
    max_length: int = 192
    eval_every: int = 50
    log_every: int = 5
    seed: int = 42
    grad_clip: float = 1.0
    device: str | None = None
    eval_subset_size: int = 200
    eval_top_k: int = 50


@dataclass
class TrainState:
    config: TrainConfig
    step: int = 0
    loss_log: list[dict] = field(default_factory=list)
    eval_log: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


def _cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _batch_iter(pairs: list[dict], batch_size: int, rng: random.Random):
    pool = list(pairs)
    while True:
        rng.shuffle(pool)
        for i in range(0, len(pool) - batch_size + 1, batch_size):
            yield pool[i:i + batch_size]


def _eval_step(
    encoder: SkillEncoder, corpus: SkillCorpus, eval_instances: list[SRAInstance],
    *, top_k: int,
) -> dict[str, float]:
    ids = corpus.ids()
    skill_texts = [corpus[sid].full_text() for sid in ids]
    corpus_emb = encoder.encode_eval(skill_texts, is_query=False, batch_size=128)
    eval_queries = [inst.query for inst in eval_instances]
    q_emb = encoder.encode_eval(eval_queries, is_query=True, batch_size=128)
    sims = (q_emb @ corpus_emb.T).cpu().numpy()
    per_query = []
    for i, inst in enumerate(eval_instances):
        order = np.argsort(-sims[i])[:top_k]
        ranked = [ids[j] for j in order]
        per_query.append({
            "Recall@1":  recall_at_k(ranked, inst.gold_skill_ids, 1),
            "Recall@5":  recall_at_k(ranked, inst.gold_skill_ids, 5),
            "Recall@10": recall_at_k(ranked, inst.gold_skill_ids, 10),
            "nDCG@10":   ndcg_at_k(ranked, inst.gold_skill_ids, 10),
        })
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}


def train(
    pairs: list[dict],
    corpus: SkillCorpus,
    eval_subset: list[SRAInstance],
    *,
    config: TrainConfig,
    output_dir: str | Path,
) -> TrainState:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    device = pick_device(config.device)
    print(f"  Device: {device}")
    encoder = SkillEncoder(config.base_model, device=device, max_length=config.max_length)
    optim = AdamW(encoder.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    warmup = max(1, int(config.warmup_frac * config.max_steps))
    state = TrainState(config=config)
    log_path = output_dir / "train_log.jsonl"
    eval_path = output_dir / "eval_log.jsonl"
    log_f = log_path.open("w")
    eval_f = eval_path.open("w")

    print(f"  Starting training: {config.max_steps} steps, "
          f"batch={config.batch_size}, n_hard={config.n_hard_negs}, "
          f"tau={config.tau}, lr={config.lr}")

    encoder.train()
    bar = tqdm(total=config.max_steps, desc="train")
    batch_gen = _batch_iter(pairs, config.batch_size, rng)

    # Eval at step 0 to capture baseline.
    metrics0 = _eval_step(encoder, corpus, eval_subset, top_k=config.eval_top_k)
    state.eval_log.append({"step": 0, **metrics0})
    eval_f.write(json.dumps(state.eval_log[-1]) + "\n"); eval_f.flush()
    print(f"  step=0   eval={metrics0}")

    for step in range(1, config.max_steps + 1):
        batch = next(batch_gen)
        queries = [b["query"] for b in batch]
        positives = [corpus[b["positive_id"]].full_text() for b in batch]
        # Hard negatives — flattened [B*N]. Pad short lists by sampling
        # extra negs from random in-batch negs if needed.
        hn_flat: list[str] = []
        for b in batch:
            negs = b["hard_neg_ids"][:config.n_hard_negs]
            while len(negs) < config.n_hard_negs:
                # Backfill from another batch entry's hard negs to keep the
                # tensor rectangular; cheap and rare in practice.
                pad_src = batch[rng.randrange(len(batch))]["hard_neg_ids"]
                if not pad_src:
                    pad_src = [random.choice(corpus.ids())]
                negs.append(pad_src[rng.randrange(len(pad_src))])
            hn_flat.extend(corpus[nid].full_text() for nid in negs)

        q_emb = encoder.encode_train(queries, is_query=True)
        p_emb = encoder.encode_train(positives, is_query=False)
        hn_emb = encoder.encode_train(hn_flat, is_query=False)

        loss = info_nce_with_hard_negs(
            q_emb, p_emb, hn_emb, n_hard_per_q=config.n_hard_negs, tau=config.tau,
        )

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip)
        lr_now = _cosine_lr(step - 1, config.max_steps, warmup, config.lr)
        for g in optim.param_groups:
            g["lr"] = lr_now
        optim.step()
        state.step = step

        if step % config.log_every == 0 or step == 1:
            entry = {
                "step": step, "loss": float(loss.detach().item()),
                "grad_norm": float(grad_norm.item()), "lr": lr_now,
                "wall_s": time.time() - state.started_at,
            }
            state.loss_log.append(entry)
            log_f.write(json.dumps(entry) + "\n"); log_f.flush()
        bar.update(1)
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

        if step % config.eval_every == 0 or step == config.max_steps:
            metrics = _eval_step(encoder, corpus, eval_subset, top_k=config.eval_top_k)
            state.eval_log.append({"step": step, **metrics})
            eval_f.write(json.dumps(state.eval_log[-1]) + "\n"); eval_f.flush()
            print(f"  step={step}   eval={metrics}")
            encoder.train()

    bar.close()
    log_f.close()
    eval_f.close()

    # Save model + config.
    ckpt = output_dir / "encoder"
    encoder.model.save(str(ckpt))
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"  Saved encoder to {ckpt}")
    return state
