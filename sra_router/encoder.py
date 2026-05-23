"""Bi-encoder wrapper around a sentence-transformers backbone.

We use ``BAAI/bge-small-en-v1.5`` (33M params, 384-dim) as the laptop-friendly
substitute for ``Qwen3-Embedding-0.6B``. The plan calls for mean pooling and
L2 normalisation, both already provided by SentenceTransformer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = (
    "Instruct: Given a task description, retrieve the most relevant skill "
    "document that would help an agent complete the task\nQuery: "
)


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SkillEncoder(torch.nn.Module):
    """Thin wrapper to expose differentiable forward + tokenisation.

    SentenceTransformer's high-level ``encode`` API is inference-only; for
    training we re-use the same underlying Transformer + pooling but call them
    via ``__call__`` to keep gradients.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: torch.device | None = None, max_length: int = 256):
        super().__init__()
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.max_length = max_length
        # Disable any built-in normalisation layers that aren't part of training.
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _format_query(self, text: str) -> str:
        return QUERY_PREFIX + text

    def encode_train(self, texts: list[str], *, is_query: bool) -> torch.Tensor:
        if is_query:
            texts = [self._format_query(t) for t in texts]
        # Bypass sentence-transformers' preprocess (which mixes tensors and
        # strings) and tokenize directly so every value is a tensor we can
        # move to-device. ``model(...)`` only needs input_ids + attention_mask.
        feats = self.model.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        feats = {k: v.to(self.device) for k, v in feats.items()
                 if hasattr(v, "to")}
        out = self.model(feats)
        emb = out["sentence_embedding"]
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def encode_eval(self, texts: list[str], *, is_query: bool,
                    batch_size: int = 64) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            if is_query:
                texts = [self._format_query(t) for t in texts]
            out = self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True,
                convert_to_tensor=True, show_progress_bar=False,
            )
            return out
        finally:
            if was_training:
                self.train()
