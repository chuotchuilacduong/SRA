"""RocketQAv2: joint dual-encoder + cross-encoder training via listwise distillation.

    L = KL(p̃_DE ‖ p̃_CE)  +  CrossEntropy(s_CE, positive_idx)

Accepts the same ``train_pairs`` format produced by ``sragents mine-negatives``
(each pair has ``question``, ``skill`` (dict), ``label``, ``instance_id``).
Uses :class:`~sragents.retrieve.skill_packer.SkillPacker` for text formatting,
matching the cross-encoder training convention.

After training, checkpoints are saved to::

    <output_dir>/dual_encoder/    ← use with: sragents retrieve --retriever bge
                                       --retriever-arg model_path=<dir>/dual_encoder
    <output_dir>/cross_encoder/   ← use with: sragents rerank-topk
                                       --model-path <dir>/cross_encoder

Reference: Ren et al., "RocketQAv2", EMNLP 2021.
"""

from __future__ import annotations

import json
import logging
import random
from contextlib import nullcontext as _nullctx
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sragents.retrieve.metrics import compute_retrieval_metrics
from sragents.retrieve.skill_packer import SkillPacker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- config


@dataclass
class RocketQAv2Config:
    """Mirrors :class:`~sragents.train.train_cross_encoder.TrainConfig` structure.

    The same YAML file used for ``train-rerank`` is accepted; add a
    ``rocketqav2:`` section for DE-specific settings.
    """

    # Models
    de_model: str = "BAAI/bge-base-en-v1.5"
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    de_query_prefix: str = "Represent this sentence for searching relevant passages: "
    de_max_length: int = 256
    ce_max_length: int = 256
    encode_sub_batch: int = 4    # passages encoded per sub-batch (limits peak VRAM)

    # Training
    batch_size: int = 8          # queries per gradient step
    lr: float = 1e-5
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    fp16_if_cuda: bool = True
    grad_accum_steps: int = 1

    # Skill formatting (same knobs as TrainConfig)
    packing_mode: str = "field_tagged"
    include_tools: bool = False
    max_content_chars: int = 1800

    # Dev / early stopping
    early_stop_metric: str = "dev_ndcg@10"
    early_stop_patience: int = 1

    log_every: int = 50

    # --- F1-F4 distillation knobs (added for KL_FULL_RUN_PLAN.md) -------------
    distill_temperature: float = 4.0
    """Softmax temperature T for KL distillation. Larger T → softer, less
    scale-sensitive; the KL term is scaled by T² to keep gradient magnitude
    comparable to the unscaled supervised loss (Hinton et al., 2015)."""

    lambda_kl_max: float = 1.0
    """Upper bound for the KL loss weight. Used together with
    :attr:`lambda_kl_warmup_epochs` to ramp KL in gradually."""

    lambda_kl_warmup_epochs: int = 2
    """Number of epochs over which lambda_kl ramps 0 → ``lambda_kl_max``.
    Set to 0 to disable warmup (lambda_kl always at max)."""

    grad_clip_norm: float = 1.0
    """Max ℓ₂ norm for ``torch.nn.utils.clip_grad_norm_`` over the union of
    DE + CE parameters. Set ≤ 0 to disable clipping."""

    distill_direction: str = "de_from_ce"
    """KL direction. One of:

    * ``"de_from_ce"`` (default) — DE student, CE teacher (CE detached).
      Preferred when CE has stronger prior knowledge (e.g. MS-MARCO).
    * ``"ce_from_de"`` — CE student, DE teacher (DE detached).
    * ``"bidirectional"`` — symmetric JSD = ½(KL(p_de||m) + KL(p_ce||m)).
    """

    @classmethod
    def from_yaml(cls, path: Path) -> "RocketQAv2Config":
        cfg = yaml.safe_load(Path(path).read_text())
        rq = cfg.get("rocketqav2", {})
        trainer = cfg.get("trainer", {})
        model = cfg.get("model", {})
        packing = cfg.get("packing", {})
        return cls(
            de_model=rq.get("de_model", cls.de_model),
            ce_model=rq.get("ce_model", model.get("base", cls.ce_model)),
            de_query_prefix=rq.get("de_query_prefix", cls.de_query_prefix),
            de_max_length=rq.get("de_max_length", model.get("max_length", cls.de_max_length)),
            ce_max_length=rq.get("ce_max_length", model.get("max_length", cls.ce_max_length)),
            encode_sub_batch=rq.get("encode_sub_batch", cls.encode_sub_batch),
            batch_size=trainer.get("batch_size", cls.batch_size),
            lr=float(trainer.get("lr", cls.lr)),
            epochs=trainer.get("epochs", cls.epochs),
            weight_decay=trainer.get("weight_decay", cls.weight_decay),
            warmup_ratio=trainer.get("warmup_ratio", cls.warmup_ratio),
            seed=trainer.get("seed", cls.seed),
            fp16_if_cuda=trainer.get("fp16_if_cuda", cls.fp16_if_cuda),
            grad_accum_steps=trainer.get("grad_accum_steps", cls.grad_accum_steps),
            packing_mode=packing.get("mode", cls.packing_mode),
            include_tools=packing.get("include_tools", cls.include_tools),
            max_content_chars=packing.get("max_content_chars", cls.max_content_chars),
            early_stop_metric=trainer.get("early_stop_metric", cls.early_stop_metric),
            early_stop_patience=trainer.get("early_stop_patience", cls.early_stop_patience),
            log_every=trainer.get("log_every", cls.log_every),
            distill_temperature=rq.get("distill_temperature", cls.distill_temperature),
            lambda_kl_max=rq.get("lambda_kl_max", cls.lambda_kl_max),
            lambda_kl_warmup_epochs=rq.get(
                "lambda_kl_warmup_epochs", cls.lambda_kl_warmup_epochs,
            ),
            grad_clip_norm=rq.get("grad_clip_norm", cls.grad_clip_norm),
            distill_direction=rq.get("distill_direction", cls.distill_direction),
        )


# ---------------------------------------------------------------------- helpers


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except (ValueError, OSError):
        return AutoTokenizer.from_pretrained(model_path, use_fast=False)


def _mean_pool(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last_hidden_state, L2-normalise. Returns (B, D)."""
    token_emb = outputs.last_hidden_state          # (B, L, D)
    mask = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
    summed = (token_emb * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2, dim=-1)


def kl_distill_loss(
    de_scores: torch.Tensor,
    ce_raw: torch.Tensor,
    *,
    temperature: float = 4.0,
    direction: str = "de_from_ce",
) -> torch.Tensor:
    """Temperature-scaled, stop-gradient KL between DE and CE listwise dists.

    See docs/KL_DE_CE_ANALYSIS.md §2 for why the legacy bi-directional KL
    without temperature kept ranking unchanged (Spearman ρ=0.99) while
    flattening the CE distribution.

    Args:
        de_scores: (N,) cosine logits over the candidate pool.
        ce_raw:    (N,) cross-encoder logits over the candidate pool.
        temperature: Softmax temperature ``T``. Output is multiplied by
            ``T²`` so gradient magnitude is comparable across temperatures.
        direction: One of ``"de_from_ce"``, ``"ce_from_de"``,
            ``"bidirectional"`` (JSD).
    """
    T = float(temperature)
    if T <= 0.0:
        raise ValueError(f"distill_temperature must be > 0, got {T}")
    if direction == "de_from_ce":
        log_p_student = F.log_softmax(de_scores / T, dim=0)
        p_teacher     = F.softmax(ce_raw / T, dim=0).detach()
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    if direction == "ce_from_de":
        log_p_student = F.log_softmax(ce_raw / T, dim=0)
        p_teacher     = F.softmax(de_scores / T, dim=0).detach()
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    if direction == "bidirectional":
        # symmetric JSD around the average distribution m
        log_p_de = F.log_softmax(de_scores / T, dim=0)
        log_p_ce = F.log_softmax(ce_raw     / T, dim=0)
        log_m = torch.logaddexp(log_p_de, log_p_ce) - torch.log(
            torch.tensor(2.0, device=log_p_de.device),
        )
        kl_de = F.kl_div(log_m, log_p_de.exp(), reduction="batchmean")
        kl_ce = F.kl_div(log_m, log_p_ce.exp(), reduction="batchmean")
        return 0.5 * (kl_de + kl_ce) * (T * T)
    raise ValueError(f"unknown distill_direction={direction!r}")


def _lambda_kl_for_epoch(
    epoch: int, *, max_lambda: float, warmup_epochs: int,
) -> float:
    """Linear warmup from 0 → ``max_lambda`` over the first ``warmup_epochs``."""
    if warmup_epochs <= 0:
        return float(max_lambda)
    return float(max_lambda) * min(1.0, (epoch + 1) / float(warmup_epochs))


def _build_candidate_list(pairs: list[dict]) -> tuple[list[dict], int] | None:
    """Order pairs as [positive, neg1, neg2, ...]; return (ordered, pos_idx=0).

    Uses first label=1 pair as the positive. Returns None if no positive or no
    negatives are available.
    """
    positives = [p for p in pairs if int(p["label"]) == 1]
    negatives = [p for p in pairs if int(p["label"]) == 0]
    if not positives or not negatives:
        return None
    return [positives[0]] + negatives, 0


# ---------------------------------------------------------------------- DE/CE forward


def _de_encode(
    texts: list[str],
    tokenizer,
    model,
    max_length: int,
    encode_sub_batch: int,
    device: torch.device,
    ctx,
) -> torch.Tensor:
    """Encode texts with the dual-encoder; returns (N, D) normalised embeddings.

    Processes in sub-batches to limit peak VRAM; gradient graph preserved via
    torch.cat.
    """
    chunks: list[torch.Tensor] = []
    for i in range(0, len(texts), encode_sub_batch):
        chunk = texts[i: i + encode_sub_batch]
        enc = tokenizer(
            chunk, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with ctx:
            outputs = model(**enc)
        chunks.append(_mean_pool(outputs, enc["attention_mask"]))
    return torch.cat(chunks, dim=0)


def _ce_score(
    questions: list[str],
    skill_texts: list[str],
    tokenizer,
    model,
    max_length: int,
    encode_sub_batch: int,
    device: torch.device,
    ctx,
) -> torch.Tensor:
    """Score (question, skill) pairs with the cross-encoder; returns (N,)."""
    chunks: list[torch.Tensor] = []
    for i in range(0, len(questions), encode_sub_batch):
        q_chunk = questions[i: i + encode_sub_batch]
        d_chunk = skill_texts[i: i + encode_sub_batch]
        enc = tokenizer(
            q_chunk, d_chunk,
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with ctx:
            outputs = model(**enc)
        chunks.append(outputs.logits.squeeze(-1))
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------- dev eval


def _score_pool_ce(
    ce_model,
    ce_tokenizer,
    packer: SkillPacker,
    device: torch.device,
    pool_records: list[dict],
    corpus: dict[str, dict],
    instances_by_id: dict[str, dict],
    batch_size: int,
    max_length: int,
    top_k: int = 100,
) -> list[dict]:
    """Re-rank dev pool candidates with the current CE. Returns retrieval-metrics-ready records."""
    ce_model.eval()
    out: list[dict] = []
    with torch.no_grad():
        for rec in pool_records:
            inst = instances_by_id.get(rec["instance_id"])
            question = (inst.get("question") or "") if inst else ""
            cands = rec.get("retrieved", [])[:top_k]
            if not cands:
                out.append({"instance_id": rec["instance_id"],
                            "gold_skill_ids": rec.get("gold_skill_ids", []),
                            "retrieved": []})
                continue
            qs, ds, valid = [], [], []
            for c in cands:
                sk = corpus.get(c["skill_id"])
                if sk is None:
                    continue
                qs.append(question)
                ds.append(packer.pack_skill(sk))
                valid.append(c)
            scores: list[float] = []
            for i in range(0, len(qs), batch_size):
                enc = ce_tokenizer(
                    qs[i: i + batch_size], ds[i: i + batch_size],
                    padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                logits = ce_model(**enc).logits.squeeze(-1)
                scores.extend(logits.detach().float().cpu().tolist())
            ranked = sorted(zip(valid, scores), key=lambda x: -x[1])
            out.append({
                "instance_id": rec["instance_id"],
                "gold_skill_ids": rec.get("gold_skill_ids", []),
                "retrieved": [{"skill_id": c["skill_id"], "score": float(s), "rank": i + 1}
                               for i, (c, s) in enumerate(ranked)],
            })
    return out


def _metric_key(name: str) -> str:
    n = name.lower().replace("dev_", "")
    table = {
        "ndcg@10": "nDCG@10", "ndcg@5": "nDCG@5", "ndcg@1": "nDCG@1",
        "ndcg@50": "nDCG@50", "ndcg@100": "nDCG@100",
        "recall@1": "Recall@1", "recall@5": "Recall@5",
        "recall@10": "Recall@10", "recall@50": "Recall@50",
        "recall@100": "Recall@100",
    }
    return table.get(n, name)


# ---------------------------------------------------------------------- main


def train_rocketqav2(
    *,
    train_pairs: list[dict],
    dev_pool_records: list[dict] | None,
    corpus: dict[str, dict],
    instances_by_id: dict[str, dict],
    config: RocketQAv2Config,
    output_dir: Path,
) -> dict:
    """Joint dual-encoder + cross-encoder training via listwise distillation.

    Args:
        train_pairs:       Output of ``sragents mine-negatives`` — list of dicts
                           with ``question``, ``skill`` (dict), ``label``,
                           ``instance_id``, ``gold_skill_ids``.
        dev_pool_records:  Optional pool JSON ``results`` list for early stopping
                           (scored with CE after each epoch).
        corpus:            ``{skill_id: skill_dict}`` from :func:`load_corpus_dict`.
        instances_by_id:   ``{instance_id: instance_dict}`` for dev evaluation.
        config:            :class:`RocketQAv2Config`.
        output_dir:        Where to save ``dual_encoder/`` and ``cross_encoder/``.

    Returns:
        Summary dict with ``best_epoch``, ``best_score``, ``history``.
    """
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    _set_seed(config.seed)
    device = _pick_device()
    use_fp16 = device.type == "cuda" and config.fp16_if_cuda
    log.info("device=%s fp16=%s", device, use_fp16)

    # Load models
    log.info("Loading dual-encoder: %s", config.de_model)
    de_tokenizer = _load_tokenizer(config.de_model)
    de_model = AutoModel.from_pretrained(config.de_model).to(device)

    log.info("Loading cross-encoder: %s", config.ce_model)
    ce_tokenizer = _load_tokenizer(config.ce_model)
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        config.ce_model, num_labels=1,
    ).to(device)

    packer = SkillPacker(
        mode=config.packing_mode,
        include_tools=config.include_tools,
        max_content_chars=config.max_content_chars,
    )

    # Group train_pairs by instance_id
    groups: dict[str, list[dict]] = {}
    for p in train_pairs:
        groups.setdefault(p["instance_id"], []).append(p)
    query_list = list(groups.values())
    log.info("Train queries: %d  pairs: %d", len(query_list), len(train_pairs))

    # Optimizer over both models
    optimizer = torch.optim.AdamW(
        list(de_model.parameters()) + list(ce_model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(1, len(query_list) // max(1, config.batch_size))
    total_steps = max(1, steps_per_epoch * config.epochs // max(1, config.grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_fp16 else _nullctx()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("-inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []
    step_loss_history: list[dict] = []
    global_step = 0

    for epoch in range(config.epochs):
        de_model.train()
        ce_model.train()
        random.shuffle(query_list)
        running = 0.0
        step = 0

        for batch_start in range(0, len(query_list), config.batch_size):
            batch_groups = query_list[batch_start: batch_start + config.batch_size]
            batch_loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for group in batch_groups:
                result = _build_candidate_list(group)
                if result is None:
                    continue
                ordered, pos_idx = result

                question = ordered[0]["question"]
                skill_texts = [packer.pack_skill(p["skill"]) for p in ordered]

                # DE forward: encode query and all passages
                de_q_texts = [config.de_query_prefix + question]
                de_p_texts = skill_texts

                q_emb = _de_encode(
                    de_q_texts, de_tokenizer, de_model,
                    config.de_max_length, config.encode_sub_batch, device, ctx,
                )  # (1, D)
                p_emb = _de_encode(
                    de_p_texts, de_tokenizer, de_model,
                    config.de_max_length, config.encode_sub_batch, device, ctx,
                )  # (N, D)
                de_scores = (q_emb @ p_emb.T).squeeze(0)  # (N,)

                # CE forward: score all (question, skill) pairs
                ce_raw = _ce_score(
                    [question] * len(skill_texts), skill_texts,
                    ce_tokenizer, ce_model,
                    config.ce_max_length, config.encode_sub_batch, device, ctx,
                )  # (N,)

                # F1+F2: stop-grad teacher, temperature-scaled KL
                # F3: λ warmup applied at the batch_loss accumulation below
                L_kl = kl_distill_loss(
                    de_scores, ce_raw,
                    temperature=config.distill_temperature,
                    direction=config.distill_direction,
                )
                # L_sup: listwise cross-entropy on CE scores (Eq.5 in paper)
                L_sup = F.cross_entropy(
                    ce_raw.unsqueeze(0),
                    torch.tensor([pos_idx], device=device),
                )

                lam_kl = _lambda_kl_for_epoch(
                    epoch,
                    max_lambda=config.lambda_kl_max,
                    warmup_epochs=config.lambda_kl_warmup_epochs,
                )
                batch_loss = batch_loss + lam_kl * L_kl + L_sup
                n_valid += 1

            if n_valid == 0:
                continue

            loss = batch_loss / n_valid / config.grad_accum_steps
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                # F4: gradient clipping over union of DE+CE parameters.
                # For fp16, unscale BEFORE clipping (otherwise we clip scaled grads).
                if config.grad_clip_norm and config.grad_clip_norm > 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(de_model.parameters()) + list(ce_model.parameters()),
                        max_norm=config.grad_clip_norm,
                    )
                if use_fp16:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_val = float(loss.detach()) * config.grad_accum_steps
            running += loss_val
            step += 1
            global_step += 1
            step_loss_history.append({
                "global_step": global_step, "epoch": epoch, "step": step,
                "loss": loss_val,
            })
            if step % config.log_every == 0:
                log.info("epoch=%d step=%d loss=%.4f", epoch, step, running / max(1, step))

        # Dev evaluation (CE-based, consistent with train_cross_encoder)
        dev_metrics: dict[str, float] = {}
        if dev_pool_records:
            scored = _score_pool_ce(
                ce_model, ce_tokenizer, packer, device,
                dev_pool_records, corpus, instances_by_id,
                batch_size=max(8, config.batch_size * config.encode_sub_batch),
                max_length=config.ce_max_length,
            )
            dev_metrics = compute_retrieval_metrics(scored, top_k=100)
            log.info("dev epoch=%d %s", epoch,
                     "  ".join(f"{k}={v:.4f}" for k, v in dev_metrics.items()))

        history.append({
            "epoch": epoch,
            "loss": running / max(1, step),
            "dev_metrics": dev_metrics,
        })

        score = dev_metrics.get(_metric_key(config.early_stop_metric), float("-inf"))
        # When no dev set, save every epoch and treat last as best.
        save_this_epoch = dev_pool_records is None or score > best_score
        if save_this_epoch:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            _save(de_model, de_tokenizer, ce_model, ce_tokenizer, config, output_dir)
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


def _save(
    de_model, de_tokenizer,
    ce_model, ce_tokenizer,
    config: RocketQAv2Config,
    output_dir: Path,
) -> None:
    de_dir = output_dir / "dual_encoder"
    ce_dir = output_dir / "cross_encoder"
    de_dir.mkdir(parents=True, exist_ok=True)
    ce_dir.mkdir(parents=True, exist_ok=True)

    de_model.save_pretrained(str(de_dir))
    de_tokenizer.save_pretrained(str(de_dir))
    ce_model.save_pretrained(str(ce_dir))
    ce_tokenizer.save_pretrained(str(ce_dir))

    (output_dir / "train_config.json").write_text(
        json.dumps(config.__dict__, indent=2)
    )
    log.info("Saved dual_encoder → %s", de_dir)
    log.info("Saved cross_encoder → %s", ce_dir)
