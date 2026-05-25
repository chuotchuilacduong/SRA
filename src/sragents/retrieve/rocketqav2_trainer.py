"""RocketQAv2 joint trainer: dynamic listwise distillation.

Jointly trains a dual-encoder retriever and a cross-encoder re-ranker:

    L_KL  = KL(p̃_de || p̃_ce)          — pushes retriever toward re-ranker
    L_sup = cross_entropy(s_ce, pos_idx) — supervised listwise loss for re-ranker
    L     = L_KL + L_sup                 — updates BOTH models simultaneously

where p̃_de and p̃_ce are the softmax-normalized relevance distributions of the
dual-encoder and cross-encoder over the candidate list for each query.

After training, models are saved to <output_dir>/dual_encoder and
<output_dir>/cross_encoder.  Both are standard HuggingFace checkpoints and
can be loaded by the existing DenseRetriever and CrossEncoderReranker via
their --model-path / --retriever-arg model_path= arguments.

Reference: Ren et al., "RocketQAv2", EMNLP 2021.
"""

from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext as _nullctx
from pathlib import Path

import torch
import torch.nn.functional as F


class RocketQAv2Trainer:
    DE_DEFAULT = "BAAI/bge-base-en-v1.5"
    CE_DEFAULT = "BAAI/bge-reranker-base"
    DE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    DE_MAX_LENGTH = 256   # reduced from 512 — skills fit in 256 tokens and saves VRAM
    CE_MAX_LENGTH = 256

    def __init__(
        self,
        de_model_path: str = DE_DEFAULT,
        ce_model_path: str = CE_DEFAULT,
        n_hard_negatives: int = 15,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        device: str | None = None,
        encode_sub_batch: int = 4,   # passages encoded this many at a time (limits peak VRAM)
        fp16: bool = False,          # mixed-precision forward pass (halves activation memory)
        de_max_length: int | None = None,
        ce_max_length: int | None = None,
    ):
        self.n_hard_negatives = n_hard_negatives
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encode_sub_batch = encode_sub_batch
        self.fp16 = fp16
        self._de_max_length = de_max_length or self.DE_MAX_LENGTH
        self._ce_max_length = ce_max_length or self.CE_MAX_LENGTH
        self._de_model_path = de_model_path
        self._ce_model_path = ce_model_path
        self._de_tokenizer = None
        self._de_model = None
        self._ce_tokenizer = None
        self._ce_model = None
        self._optimizer = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tokenizer(model_path: str):
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained(model_path)
        except (ValueError, OSError):
            return AutoTokenizer.from_pretrained(model_path, use_fast=False)

    def _load_models(self) -> None:
        if self._de_model is not None:
            return
        from transformers import AutoModel, AutoModelForSequenceClassification

        print(f"  Loading dual-encoder: {self._de_model_path}")
        self._de_tokenizer = self._load_tokenizer(self._de_model_path)
        self._de_model = AutoModel.from_pretrained(self._de_model_path).to(self.device)

        print(f"  Loading cross-encoder: {self._ce_model_path}")
        self._ce_tokenizer = self._load_tokenizer(self._ce_model_path)
        self._ce_model = AutoModelForSequenceClassification.from_pretrained(
            self._ce_model_path
        ).to(self.device)

        from torch.optim import AdamW
        self._optimizer = AdamW(
            list(self._de_model.parameters()) + list(self._ce_model.parameters()),
            lr=self.learning_rate,
        )
        print(f"  Device: {self.device}")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _mean_pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool last_hidden_state weighted by attention_mask, then L2-normalise."""
        token_emb = outputs.last_hidden_state          # (B, L, D)
        mask = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
        summed = (token_emb * mask).sum(dim=1)         # (B, D)
        counts = mask.sum(dim=1).clamp(min=1e-9)       # (B, 1)
        emb = summed / counts                          # (B, D)
        return F.normalize(emb, p=2, dim=-1)

    def _encode(self, texts: list[str], is_query: bool) -> torch.Tensor:
        """Encode texts with the dual-encoder; returns (N, D) normalised embeddings.

        Processes in sub-batches of size ``encode_sub_batch`` to limit peak VRAM.
        Gradients are preserved across sub-batches via torch.cat.
        """
        if is_query:
            texts = [self.DE_QUERY_PREFIX + t for t in texts]
        chunks = []
        ctx = torch.autocast(device_type=self.device, dtype=torch.float16) if self.fp16 else _nullctx()
        for i in range(0, len(texts), self.encode_sub_batch):
            chunk = texts[i: i + self.encode_sub_batch]
            enc = self._de_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self._de_max_length,
                return_tensors="pt",
            ).to(self.device)
            with ctx:
                outputs = self._de_model(**enc)
            chunks.append(self._mean_pool(outputs, enc["attention_mask"]))
        return torch.cat(chunks, dim=0)   # (N, D)

    def _ce_scores(self, query: str, passage_texts: list[str]) -> torch.Tensor:
        """Score each (query, passage) pair with the cross-encoder; returns (N,).

        Processes in sub-batches of size ``encode_sub_batch`` to limit peak VRAM.
        """
        chunks = []
        ctx = torch.autocast(device_type=self.device, dtype=torch.float16) if self.fp16 else _nullctx()
        for i in range(0, len(passage_texts), self.encode_sub_batch):
            chunk = passage_texts[i: i + self.encode_sub_batch]
            queries = [query] * len(chunk)
            enc = self._ce_tokenizer(
                queries,
                chunk,
                padding=True,
                truncation=True,
                max_length=self._ce_max_length,
                return_tensors="pt",
            ).to(self.device)
            with ctx:
                outputs = self._ce_model(**enc)
            chunks.append(outputs.logits.squeeze(-1))
        return torch.cat(chunks, dim=0)   # (N,)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _build_candidate_list(
        self,
        gold_ids: list[str],
        bm25_candidates: list[dict],
        corpus_texts: dict[str, str],
        n_hard_neg: int,
    ) -> tuple[list[str], list[str], int] | None:
        """Build (skill_ids, passage_texts, positive_idx) for one query.

        Returns None if the positive skill is not in corpus_texts.
        Positive is placed at index 0; hard negatives follow.
        """
        gold_set = set(gold_ids)
        positive_id = gold_ids[0]
        if positive_id not in corpus_texts:
            return None

        hard_negs = [
            c["skill_id"]
            for c in bm25_candidates
            if c["skill_id"] not in gold_set and c["skill_id"] in corpus_texts
        ][:n_hard_neg]

        if not hard_negs:
            return None

        skill_ids = [positive_id] + hard_negs
        passage_texts = [corpus_texts[sid] for sid in skill_ids]
        return skill_ids, passage_texts, 0  # positive at index 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        instances: list[dict],
        corpus_texts: dict[str, str],
        bm25_lookup: dict[str, list[dict]],
        epochs: int,
        output_dir: str | Path,
        log_every: int = 50,
    ) -> None:
        """Joint listwise distillation training loop.

        Args:
            instances:     List of instance dicts (must have "skill_annotations",
                           "instance_id", and fields read by build_prompt).
            corpus_texts:  {skill_id: text} mapping for the full corpus.
            bm25_lookup:   {instance_id: [{skill_id, score}, ...]} from a BM25 result file.
            epochs:        Number of full passes over the training data.
            output_dir:    Where to write trained model checkpoints.
            log_every:     Print loss every this many gradient steps.
        """
        self._load_models()
        output_dir = Path(output_dir)

        from sragents.prompts import build_prompt

        def _build_query(inst: dict) -> str:
            system, user = build_prompt(inst)
            parts = [user]
            if system:
                parts.append(system)
            return "\n".join(parts)

        # Keep only instances that have gold annotations and BM25 results
        usable = [
            inst for inst in instances
            if inst.get("skill_annotations")
            and inst["instance_id"] in bm25_lookup
        ]
        print(f"  Usable training instances: {len(usable)} / {len(instances)}")

        global_step = 0
        for epoch in range(1, epochs + 1):
            random.shuffle(usable)
            epoch_loss = 0.0
            n_steps = 0

            # Iterate in batches
            for batch_start in range(0, len(usable), self.batch_size):
                batch = usable[batch_start: batch_start + self.batch_size]
                batch_loss = torch.tensor(0.0, device=self.device)
                n_valid = 0

                for inst in batch:
                    gold_ids = inst["skill_annotations"]
                    bm25_cands = bm25_lookup[inst["instance_id"]]
                    result = self._build_candidate_list(
                        gold_ids, bm25_cands, corpus_texts, self.n_hard_negatives
                    )
                    if result is None:
                        continue
                    _, passage_texts, pos_idx = result
                    query_text = _build_query(inst)

                    # --- Dual-encoder forward ---
                    q_emb = self._encode([query_text], is_query=True)   # (1, D)
                    p_emb = self._encode(passage_texts, is_query=False) # (N, D)
                    de_scores = (q_emb @ p_emb.T).squeeze(0)            # (N,)

                    # --- Cross-encoder forward ---
                    ce_raw = self._ce_scores(query_text, passage_texts)  # (N,)

                    # --- Listwise distributions ---
                    p_de_log = F.log_softmax(de_scores, dim=0)   # log p̃_de
                    p_ce_log = F.log_softmax(ce_raw, dim=0)      # log p̃_ce
                    p_de = p_de_log.exp()                         # p̃_de

                    # L_KL = KL(p̃_de || p̃_ce)
                    # F.kl_div(input=log_ce, target=p_de) = Σ p_de*(log p_de - log_ce) ✓
                    L_KL = F.kl_div(p_ce_log, p_de, reduction="sum")

                    # L_sup: listwise cross-entropy on CE scores (Eq.5 in paper)
                    L_sup = F.cross_entropy(
                        ce_raw.unsqueeze(0),
                        torch.tensor([pos_idx], device=self.device),
                    )

                    batch_loss = batch_loss + L_KL + L_sup
                    n_valid += 1

                if n_valid == 0:
                    continue

                batch_loss = batch_loss / n_valid
                self._optimizer.zero_grad()
                batch_loss.backward()
                self._optimizer.step()

                epoch_loss += batch_loss.item()
                n_steps += 1
                global_step += 1

                if global_step % log_every == 0:
                    print(
                        f"  epoch {epoch} | step {global_step} | "
                        f"loss {batch_loss.item():.4f}",
                        flush=True,
                    )

            avg = epoch_loss / max(n_steps, 1)
            print(f"  Epoch {epoch} done — avg loss {avg:.4f}", flush=True)

        self.save(output_dir)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        """Save both trained models as HuggingFace checkpoints."""
        output_dir = Path(output_dir)
        de_dir = output_dir / "dual_encoder"
        ce_dir = output_dir / "cross_encoder"
        de_dir.mkdir(parents=True, exist_ok=True)
        ce_dir.mkdir(parents=True, exist_ok=True)

        self._de_model.save_pretrained(str(de_dir))
        self._de_tokenizer.save_pretrained(str(de_dir))
        self._ce_model.save_pretrained(str(ce_dir))
        self._ce_tokenizer.save_pretrained(str(ce_dir))

        # Record provenance
        meta = {
            "de_base": self._de_model_path,
            "ce_base": self._ce_model_path,
            "n_hard_negatives": self.n_hard_negatives,
            "learning_rate": self.learning_rate,
        }
        (output_dir / "train_meta.json").write_text(
            json.dumps(meta, indent=2)
        )
        print(f"  Saved dual_encoder → {de_dir}")
        print(f"  Saved cross_encoder → {ce_dir}")
