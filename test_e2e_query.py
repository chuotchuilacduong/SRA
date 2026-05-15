#!/usr/bin/env python3
"""
SAR-gated end-to-end pipeline: query → UE probe → (retrieve?) → LLM answer.

Step 0  UE GATE  — probe Qwen via vLLM to measure uncertainty (sent_sar).
                   score > threshold → RETRIEVE; else → answer directly.

Step 1-7 LinearRAG retrieval (only executed when gate says RETRIEVE):
  1. Load corpus
  2. Build LinearRAG index
  3. Seed entities (spacy NER)
  4. Entity node activation (BFS)
  5. Sentence node activation
  6. Passage weights (dense + entity bonus)
  7. PPR ranking

Step 8  LLM ANSWER — Qwen2.5-0.5B via vLLM, with or without skills.

Run from SRA project root (vLLM must be running on port 8000):
    python test_e2e_query.py
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np

_src = Path(__file__).parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv
load_dotenv()

# ── Defaults (overridable via CLI args) ────────────────────────────────────────

_DEFAULT_QUERY    = "Please solve the equation 2*x^3 + e^x = 10 using Newton-Raphson method."
_DEFAULT_CORPUS   = "SRA-Bench/corpus/corpus_1000.json"
_DEFAULT_VLLM     = "http://localhost:8000/v1"
_DEFAULT_MODEL    = "Qwen/Qwen2.5-0.5B-Instruct"
_DEFAULT_UE_MODE  = "sent_sar"
_DEFAULT_THRESH   = -6.1
_DEFAULT_SAMPLES  = 5
_DEFAULT_TOPK     = 5

SEP2 = "=" * 70
SHOW_TOP = 5


def _parse_args():
    ap = argparse.ArgumentParser(description="SAR-gated e2e query pipeline")
    ap.add_argument("--query",       default=_DEFAULT_QUERY,   help="Query string")
    ap.add_argument("--corpus",      default=_DEFAULT_CORPUS,  help="Corpus JSON path")
    ap.add_argument("--api-base",    default=_DEFAULT_VLLM,    help="vLLM base URL")
    ap.add_argument("--model",       default=_DEFAULT_MODEL,   help="Model name")
    ap.add_argument("--ue-mode",     default=_DEFAULT_UE_MODE, help="UE mode")
    ap.add_argument("--threshold",   type=float, default=_DEFAULT_THRESH, help="UE threshold")
    ap.add_argument("--n-samples",   type=int,   default=_DEFAULT_SAMPLES, help="Probe samples")
    ap.add_argument("--topk",        type=int,   default=_DEFAULT_TOPK,   help="Top-K skills")
    ap.add_argument("--max-tokens",  type=int,   default=512,  help="Max answer tokens")
    return ap.parse_args()


def section(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def _wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    return textwrap.indent(textwrap.fill(text[:500], width=width), indent)


def _skill_text(skill: dict) -> str:
    parts = [skill.get(k) or "" for k in ("name", "description", "content")]
    return "\n".join(p for p in parts if p)


def _skill_id_from_passage_text(passage_text: str, idx_to_skill: dict) -> str:
    prefix = passage_text.split(":")[0]
    return idx_to_skill.get(int(prefix), "?") if prefix.isdigit() else "?"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    QUERY        = args.query
    CORPUS_JSON  = Path(args.corpus)
    VLLM_BASE    = args.api_base
    VLLM_MODEL   = args.model
    UE_MODE      = args.ue_mode
    UE_THRESHOLD = args.threshold
    UE_N_SAMPLES = args.n_samples
    TOP_K        = args.topk
    MAX_TOKENS   = args.max_tokens
    DATASET_NAME = "sar_pipeline"

    from openai import OpenAI
    from sragents.ue.sar import SARScorer

    client = OpenAI(base_url=VLLM_BASE, api_key="EMPTY")

    # ── Step 0: UE Gate ───────────────────────────────────────────────────────
    section("0. UE GATE  (SAR uncertainty estimation)")
    print(f"\n  Query    : {QUERY!r}")
    print(f"  Model    : {VLLM_MODEL}")
    print(f"  UE mode  : {UE_MODE}")
    print(f"  Threshold: {UE_THRESHOLD}  (score > threshold → RETRIEVE)\n")

    scorer = SARScorer(
        mode=UE_MODE,
        n_samples=UE_N_SAMPLES,
        temperature=0.8,
        threshold=UE_THRESHOLD,
    )
    ue_result = scorer.score(QUERY, client, VLLM_MODEL, max_tokens=64)

    score_str = f"{ue_result.score:.4f}" if ue_result.score != float("inf") else "inf"
    decision  = "RETRIEVE  ← score > threshold" if ue_result.needs_retrieval \
                else "DIRECT    ← score ≤ threshold"

    print(f"  UE score : {score_str}   [K={ue_result.n_generations} samples]")
    print(f"  Decision : {decision}")

    skill_context = ""

    if ue_result.needs_retrieval:
        # ── Steps 1-7: LinearRAG retrieval ────────────────────────────────────

        # 1. Corpus
        section("1. CORPUS")
        if not CORPUS_JSON.exists():
            sys.exit(f"[ERROR] {CORPUS_JSON} not found.")
        corpus: list[dict] = json.loads(CORPUS_JSON.read_text())
        corpus_dict: dict[str, dict] = {s["skill_id"]: s for s in corpus}
        corpus_ids   = [s["skill_id"] for s in corpus]
        corpus_texts = [_skill_text(s) for s in corpus]
        print(f"  Loaded {len(corpus):,} skills from {CORPUS_JSON}")

        # 2. Index
        section(f"2. BUILD INDEX  (cached in import/{DATASET_NAME}/)")
        from sragents.retrieve.linearrag import LinearRAGRetriever
        retriever = LinearRAGRetriever(
            dataset_name=DATASET_NAME,
            retrieval_top_k_cap=50,
            batch_size=32,
            max_workers=2,
            max_chars_per_passage=2000,
        )
        retriever.build_index(corpus_ids, corpus_texts)
        print("  Index ready.")
        retriever.retrieve([QUERY], top_k=TOP_K)
        rag = retriever._rag

        # 3. Seed entities
        section("3. SEED ENTITIES  (spacy NER on query → matched in corpus graph)")
        print(f"\n  Query: {QUERY!r}\n")
        question_embedding = rag.config.embedding_model.encode(
            QUERY, normalize_embeddings=True, show_progress_bar=False,
            batch_size=rag.config.batch_size,
        )
        seed_indices, seed_entities, seed_hash_ids, seed_scores = rag.get_seed_entities(QUERY)
        if not seed_entities:
            print("  No named entities — using dense-only fallback.")
        else:
            print(f"  {len(seed_entities)} seed entities:")
            for ent, score in zip(seed_entities, seed_scores):
                print(f"    score={score:.4f}  entity={ent!r}")

        # 4. Entity node activation
        section("4. ENTITY NODE ACTIVATION")
        if not seed_entities:
            actived_entities: dict = {}
            entity_weights = np.zeros(len(rag.graph.vs["name"]))
            print("  (skipped — no seed entities)")
        else:
            entity_weights, actived_entities = rag.calculate_entity_scores(
                question_embedding, seed_indices, seed_entities, seed_hash_ids, seed_scores
            )
            sorted_ents = sorted(
                actived_entities.items(), key=lambda kv: kv[1][1], reverse=True
            )
            print(f"\n  {len(actived_entities)} entity nodes activated.")
            for e_hash, (_, e_score, tier) in sorted_ents[:SHOW_TOP]:
                e_text = rag.entity_embedding_store.hash_id_to_text.get(e_hash, "?")
                print(f"    score={e_score:.5f}  tier={tier}  {e_text!r}")

        # 5. Sentence node activation
        section("5. SENTENCE NODE ACTIVATION")
        if actived_entities:
            sent_score: dict[str, float] = {}
            for e_hash, (_, e_score, _) in actived_entities.items():
                for s_hash in rag.entity_hash_id_to_sentence_hash_ids.get(e_hash, []):
                    sent_score[s_hash] = max(sent_score.get(s_hash, 0.0), e_score)
            sorted_sents = sorted(sent_score.items(), key=lambda kv: kv[1], reverse=True)
            print(f"\n  {len(sent_score)} sentence nodes linked to activated entities.")
            for s_hash, s_score in sorted_sents[:SHOW_TOP]:
                s_text = rag.sentence_embedding_store.hash_id_to_text.get(s_hash, "?")
                print(f"  score={s_score:.5f}")
                print(_wrap(s_text))
                print()
        else:
            print("  (skipped)")

        # 6. Passage weights
        section("6. PASSAGE WEIGHTS  (dense + entity bonus)")
        if actived_entities:
            from sragents.retrieve._linearrag.utils import min_max_normalize
            d_indices, d_scores_raw = rag.dense_passage_retrieval(question_embedding)
            d_scores_norm = min_max_normalize(d_scores_raw)
            rows = []
            for i, d_idx in enumerate(d_indices):
                p_hash = rag.passage_embedding_store.hash_ids[d_idx]
                p_text_lower = rag.passage_embedding_store.hash_id_to_text[p_hash].lower()
                entity_bonus = 0.0
                for e_hash, (_, e_score, tier) in actived_entities.items():
                    e_lower = rag.entity_embedding_store.hash_id_to_text[e_hash].lower()
                    occ = p_text_lower.count(e_lower)
                    if occ > 0:
                        entity_bonus += e_score * math.log(1 + occ) / max(tier, 1)
                combined = rag.config.passage_ratio * d_scores_norm[i] + math.log(1 + entity_bonus)
                rows.append((p_hash, d_scores_norm[i], entity_bonus, combined))
            rows.sort(key=lambda r: r[3], reverse=True)
            print(f"\n  {'Rank':>4}  {'dense':>8}  {'entity_bonus':>12}  {'combined':>10}  skill")
            print(f"  {'─'*4}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*30}")
            for rank, (p_hash, d_norm, ebonus, combined) in enumerate(rows[:SHOW_TOP], 1):
                p_text = rag.passage_embedding_store.hash_id_to_text[p_hash]
                sid = _skill_id_from_passage_text(p_text, retriever._idx_to_skill_id)
                name = corpus_dict.get(sid, {}).get("name", sid)
                print(f"  {rank:>4}  {d_norm:>8.5f}  {ebonus:>12.6f}  {combined:>10.6f}  {name!r}")

        # 7. PPR
        section("7. PPR OUTPUT  (Personalized PageRank)")
        if seed_entities:
            passage_weights = rag.calculate_passage_scores(
                QUERY, question_embedding, actived_entities
            )
            node_weights = entity_weights + passage_weights
            ppr_hash_ids, ppr_scores = rag.run_ppr(node_weights)
        else:
            d_indices, d_scores_raw = rag.dense_passage_retrieval(question_embedding)
            ppr_hash_ids = [rag.passage_embedding_store.hash_ids[i] for i in d_indices]
            ppr_scores = d_scores_raw

        print(f"\n  Top-{args.topk} passages after PPR:\n")
        skill_passages: list[str] = []
        for rank in range(args.topk):
            p_hash = ppr_hash_ids[rank]
            p_text = rag.passage_embedding_store.hash_id_to_text[p_hash]
            skill_id = _skill_id_from_passage_text(p_text, retriever._idx_to_skill_id)
            skill = corpus_dict.get(skill_id, {})
            name = skill.get("name") or skill_id
            desc = (skill.get("description") or "").strip()
            content = (skill.get("content") or "").strip()
            full = f"{desc}\n{content}".strip()
            skill_passages.append(full)
            print(f"  [{rank+1}] skill_id={skill_id}  ppr_score={ppr_scores[rank]:.8f}")
            print(f"       Name : {name}")
            print(_wrap(full[:300]))
            print()

        skill_context = "\n\n---\n\n".join(
            f"[Skill {i+1}]\n{t[:600]}" for i, t in enumerate(skill_passages)
        )

    # ── Step 8: LLM Answer (Qwen via vLLM) ───────────────────────────────────
    section(f"8. LLM ANSWER  ({VLLM_MODEL})")

    if skill_context:
        system_msg = (
            "You are a helpful assistant. Use the retrieved skill passages "
            "below as context to answer the user's question concisely."
        )
        user_msg = f"Relevant skill passages:\n\n{skill_context}\n\nQuestion: {QUERY}"
        print("  [With retrieved skills]\n")
    else:
        system_msg = "You are a helpful assistant. Answer the question concisely."
        user_msg = QUERY
        print("  [Direct — no skills retrieved]\n")

    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=MAX_TOKENS,
    )
    answer = response.choices[0].message.content or ""
    print(textwrap.indent(answer.strip(), "  "))
    print()


if __name__ == "__main__":
    main()
