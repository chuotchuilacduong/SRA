"""LinearRAG graph-based retriever adapter for SR-Agents.

Wraps the LinearRAG core (sragents.retrieve._linearrag) and exposes
it through the sragents.retrieve.Retriever protocol.

Design notes (per plan):
- Passage text = description + content (name is excluded; name only used in
  inference prompts, not as retrieval signal).
- The CLI passes corpus_texts built via sragents.corpus.skill_text (which
  includes name); the adapter ignores corpus_texts and rebuilds text per
  skill_id by looking up sragents.corpus.load_corpus_dict().
- Passage-passage adjacency edges are NOT used (removed in the copied core).
- LinearRAG's retrieve() reads top_k from config.retrieval_top_k; we ask for
  retrieval_top_k_cap and slice down to the caller's top_k.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sragents.retrieve._linearrag import LinearRAG, LinearRAGConfig
from sragents.retrieve.base import register

_PREFIX_RE = re.compile(r"^(\d+):")


class LinearRAGRetriever:
    def __init__(
        self,
        embedding_model_path: str = "sentence-transformers/all-mpnet-base-v2",
        spacy_model: str = "en_core_web_sm",
        working_dir: str | None = None,
        dataset_name: str = "sragents_corpus",
        retrieval_top_k_cap: int | str = 200,
        batch_size: int | str = 64,
        max_workers: int | str = 4,
        max_chars_per_passage: int | str = 4000,
        use_vectorized_retrieval: bool | str = False,
        enable_passage_adjacency: bool | str = False,
    ):
        retrieval_top_k_cap = int(retrieval_top_k_cap)
        batch_size = int(batch_size)
        max_workers = int(max_workers)
        max_chars_per_passage = int(max_chars_per_passage)

        def _coerce_bool(v):
            return v.lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)
        use_vectorized_retrieval = _coerce_bool(use_vectorized_retrieval)
        enable_passage_adjacency = _coerce_bool(enable_passage_adjacency)

        from sentence_transformers import SentenceTransformer
        from sragents.config import PROJECT_ROOT

        self._embedding_model = SentenceTransformer(embedding_model_path)
        self._working_dir = working_dir or str(Path(PROJECT_ROOT) / "import")
        self._dataset_name = dataset_name
        self._top_k_cap = retrieval_top_k_cap
        self._max_chars = max_chars_per_passage

        self._cfg = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=self._embedding_model,
            spacy_model=spacy_model,
            working_dir=self._working_dir,
            retrieval_top_k=retrieval_top_k_cap,
            batch_size=batch_size,
            max_workers=max_workers,
            use_vectorized_retrieval=use_vectorized_retrieval,
            enable_passage_adjacency=enable_passage_adjacency,
        )
        self._idx_to_skill_id: Dict[int, str] = {}
        self._rag: LinearRAG | None = None

    def build_index(self, corpus_ids: List[str], corpus_texts: List[str]) -> None:
        # corpus_texts from CLI is skill_text() = name+description+content.
        # Per design we use description+content only — rebuild via corpus_dict.
        try:
            from sragents.corpus import load_corpus_dict
            cdict = load_corpus_dict()
        except Exception:
            cdict = {}

        self._idx_to_skill_id = {i: sid for i, sid in enumerate(corpus_ids)}
        passages: List[str] = []
        for i, sid in enumerate(corpus_ids):
            skill = cdict.get(sid)
            if skill is None:
                # Fallback: strip first line if it looks like a short title (name).
                raw = corpus_texts[i]
                first, _, rest = raw.partition("\n")
                text = rest if rest and len(first) <= 80 else raw
            else:
                parts = [skill.get("description") or "", skill.get("content") or ""]
                text = "\n".join(p for p in parts if p)
            passages.append(f"{i}:{text[: self._max_chars]}")

        mapping_path = Path(self._working_dir) / self._dataset_name / "skill_id_map.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(json.dumps(self._idx_to_skill_id))

        self._rag = LinearRAG(global_config=self._cfg)
        self._rag.index(passages)

    def retrieve(
        self, queries: List[str], top_k: int
    ) -> List[List[Tuple[str, float]]]:
        assert self._rag is not None, "Call build_index first."
        # Ensure LinearRAG returns >= top_k passages.
        self._cfg.retrieval_top_k = max(top_k, self._top_k_cap)

        wrapped = [{"question": q, "answer": ""} for q in queries]
        raw = self._rag.retrieve(wrapped)

        out: List[List[Tuple[str, float]]] = []
        for r in raw:
            ranked: List[Tuple[str, float]] = []
            seen: set[str] = set()
            for passage_text, score in zip(r["sorted_passage"], r["sorted_passage_scores"]):
                m = _PREFIX_RE.match(passage_text.strip())
                if not m:
                    continue
                idx = int(m.group(1))
                sid = self._idx_to_skill_id.get(idx)
                if sid is None or sid in seen:
                    continue
                seen.add(sid)
                ranked.append((sid, float(score)))
                if len(ranked) >= top_k:
                    break
            out.append(ranked)
        return out


@register("linearrag")
def _factory(**kwargs) -> LinearRAGRetriever:
    return LinearRAGRetriever(**kwargs)
