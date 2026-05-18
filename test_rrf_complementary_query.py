#!/usr/bin/env python3
"""
End-to-end RRF Complementary pipeline trace: query → BM25 + Dense + LinearRAG
→ RRF fusion → top-k passages → local Qwen LLM answer.

Run from the SRA project root inside conda env linearag311:
    python test_rrf_complementary_query.py

Pipeline:
  1.  Load full corpus (26,262 skills)
  2.  Build BM25 index (~4s)
  3.  Build Dense (BGE-base-en-v1.5) index (loads from cache if available, else ~9 min)
  4.  Load cached LinearRAG results for one query
  5.  BM25 RETRIEVE        → top-100 sparse/lexical
  6.  DENSE RETRIEVE       → top-100 semantic
  7.  LinearRAG RETRIEVE   → top-50 graph-based
  8.  RRF COMPLEMENTARY FUSION  → top-K final (k_rrf=60, equal weights)
  9.  LOCAL Qwen LLM ANSWER  (OpenAI-compat vLLM endpoint)

Each retriever captures different aspects of relevance:
  - BM25:      exact keyword matching, rare technical terms
  - Dense BGE: semantic similarity, paraphrase robustness
  - LinearRAG: entity-graph propagation, multi-hop reasoning
"""

import json
import os
import sys
import textwrap
import time
from pathlib import Path

_src = Path(__file__).parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

# Example query (a math/engineering problem -> mix of lexical + semantic)
QUERY = "Please solve the equation 2*x^3 + e^x = 10 using Newton-Raphson method."

CORPUS_JSON = Path("data/bench/corpus/corpus.json")
LINEARRAG_CACHE = "results/retrieval_all/theoremqa-linearrag.json"  # any dataset cache
TOP_K = 5
SPARSE_TOP_K = 100  # BM25 + Dense pool size

# Local Qwen via vLLM (OpenAI-compatible). Override via env if needed.
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B-Instruct")
API_BASE = os.environ.get("API_BASE", "http://localhost:8000/v1")

# Dense model: same BGE used by LinearRAG
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_BATCH = 64
MAX_CHARS = 3000

SEP = "─" * 70
SEP2 = "=" * 70
SHOW_TOP = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def _wrap(text: str, width: int = 76, indent: str = "    ") -> str:
    return textwrap.indent(textwrap.fill(text[:500], width=width), indent)


def _skill_text(skill: dict) -> str:
    parts = [skill.get(k) or "" for k in ("name", "description", "content")]
    return "\n".join(p for p in parts if p)


def build_corpus_text(corpus: list[dict]) -> tuple[list[str], list[str]]:
    """Same passage format as LinearRAG (description + content)."""
    corpus_ids = []
    corpus_texts = []
    for skill in corpus:
        parts = [skill.get("description") or "", skill.get("content") or ""]
        text = "\n".join(p for p in parts if p)
        corpus_ids.append(skill["skill_id"])
        corpus_texts.append(text[:MAX_CHARS])
    return corpus_ids, corpus_texts


def show_ranking(name: str, ranking: list[dict], corpus_dict: dict, top: int = SHOW_TOP) -> None:
    print(f"\n  {name} top-{top}:")
    print(f"  {'Rank':>4}  {'Score':>10}  skill_id / name")
    print(f"  {'─'*4}  {'─'*10}  {'─'*40}")
    for i, item in enumerate(ranking[:top]):
        sid = item["skill_id"]
        name = corpus_dict.get(sid, {}).get("name", sid)
        score = item.get("score", 0.0)
        print(f"  {i+1:>4}  {score:>10.6f}  {sid} / {name!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from sragents.retrieve.bm25 import BM25Retriever
    from sragents.retrieve.dense import DenseRetriever
    from sragents.retrieve.fusion import complementary_fusion

    # ── 1. Corpus ─────────────────────────────────────────────────────────────
    section("1. CORPUS")
    if not CORPUS_JSON.exists():
        sys.exit(f"[ERROR] {CORPUS_JSON} not found — run from the SRA project root.")
    corpus: list[dict] = json.loads(CORPUS_JSON.read_text())
    corpus_dict: dict[str, dict] = {s["skill_id"]: s for s in corpus}
    corpus_ids, corpus_texts = build_corpus_text(corpus)
    print(f"  Loaded {len(corpus):,} skills from {CORPUS_JSON}")

    # ── 2. BM25 index ─────────────────────────────────────────────────────────
    section("2. BM25 INDEX  (sparse, lexical — exact keywords)")
    t0 = time.time()
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(corpus_ids, corpus_texts)
    print(f"  BM25 ready in {time.time() - t0:.1f}s")

    # ── 3. Dense index ────────────────────────────────────────────────────────
    section("3. DENSE INDEX  (BGE-base-en-v1.5 — semantic, paraphrase)")
    t0 = time.time()
    dense = DenseRetriever(
        model_name_or_path=DENSE_MODEL,
        query_prefix=DENSE_QUERY_PREFIX,
        batch_size=DENSE_BATCH,
    )
    dense.build_index(corpus_ids, corpus_texts)
    print(f"  Dense ready in {(time.time() - t0)/60:.1f} min")

    # ── 4. LinearRAG (cached) ─────────────────────────────────────────────────
    section("4. LINEARRAG CACHED RESULTS  (graph + dense, entity propagation)")
    # Strategy: build a per-query LinearRAG result on-the-fly via the
    # LinearRAGRetriever (uses NER cache already built). This makes the
    # script self-contained — any QUERY works without precomputed file.
    from sragents.retrieve.linearrag import LinearRAGRetriever

    lr_retriever = LinearRAGRetriever(
        dataset_name="bench_full",
        retrieval_top_k_cap=SPARSE_TOP_K,
        batch_size=32,
        max_workers=1,
        max_chars_per_passage=MAX_CHARS,
    )
    print("  Building/loading LinearRAG index (uses cached NER+embeddings)...")
    t0 = time.time()
    lr_retriever.build_index(corpus_ids, corpus_texts)
    print(f"  LinearRAG index ready in {(time.time() - t0)/60:.1f} min")

    # ── 5. BM25 retrieve ──────────────────────────────────────────────────────
    section("5. BM25 RETRIEVE  (sparse top-100)")
    print(f"\n  Query: {QUERY!r}")
    t0 = time.time()
    bm25_raw = bm25.retrieve([QUERY], top_k=SPARSE_TOP_K)[0]
    bm25_ranks = [
        {"skill_id": sid, "score": float(sc), "rank": r + 1}
        for r, (sid, sc) in enumerate(bm25_raw)
    ]
    print(f"  Retrieved {len(bm25_ranks)} in {time.time() - t0:.1f}s")
    show_ranking("BM25", bm25_ranks, corpus_dict)

    # ── 6. Dense retrieve ─────────────────────────────────────────────────────
    section("6. DENSE RETRIEVE  (BGE top-100)")
    t0 = time.time()
    dense_raw = dense.retrieve([QUERY], top_k=SPARSE_TOP_K)[0]
    dense_ranks = [
        {"skill_id": sid, "score": float(sc), "rank": r + 1}
        for r, (sid, sc) in enumerate(dense_raw)
    ]
    print(f"  Retrieved {len(dense_ranks)} in {time.time() - t0:.1f}s")
    show_ranking("Dense BGE", dense_ranks, corpus_dict)

    # ── 7. LinearRAG retrieve ─────────────────────────────────────────────────
    section("7. LINEARRAG RETRIEVE  (graph+dense top-50)")
    t0 = time.time()
    lr_raw = lr_retriever.retrieve([QUERY], top_k=SPARSE_TOP_K)[0]
    lr_ranks = [
        {"skill_id": sid, "score": float(sc), "rank": r + 1}
        for r, (sid, sc) in enumerate(lr_raw)
    ]
    print(f"  Retrieved {len(lr_ranks)} in {time.time() - t0:.1f}s")
    show_ranking("LinearRAG", lr_ranks, corpus_dict)

    # ── 8. RRF COMPLEMENTARY FUSION ───────────────────────────────────────────
    section("8. RRF COMPLEMENTARY FUSION  (BM25 ⊕ Dense ⊕ LinearRAG, k_rrf=60)")
    print(
        "\n  Formula (per skill):\n"
        "    rrf_score(d) = Σ_{i ∈ rankers} weight_i / (k_rrf + rank_i(d))\n"
        "  where rank_i = position (1-indexed) of d in ranker i's top-100.\n"
        "  Skills ranked highly by multiple rankers get CUMULATIVE evidence.\n"
    )
    t0 = time.time()
    fused = complementary_fusion(
        bm25_ranks=bm25_ranks,
        dense_ranks=dense_ranks,
        linearrag_ranks=lr_ranks,
        bm25_weight=1.0,
        dense_weight=1.0,
        linearrag_weight=1.0,
        top_k=TOP_K,
        k_rrf=60,
    )
    print(f"  Fused in {time.time() - t0:.3f}s")
    show_ranking("⭐ RRF Complementary", fused, corpus_dict, top=TOP_K)

    # Breakdown: where did the winners come from?
    print(f"\n  Provenance (which rankers contributed each top-{TOP_K}):")
    bm25_set = {x["skill_id"]: x["rank"] for x in bm25_ranks}
    dense_set = {x["skill_id"]: x["rank"] for x in dense_ranks}
    lr_set = {x["skill_id"]: x["rank"] for x in lr_ranks}
    print(f"  {'Rank':>4}  {'skill_id':<24}  {'BM25':>6}  {'Dense':>6}  {'LinRAG':>7}")
    print(f"  {'─'*4}  {'─'*24}  {'─'*6}  {'─'*6}  {'─'*7}")
    for i, item in enumerate(fused[:TOP_K]):
        sid = item["skill_id"]
        b = bm25_set.get(sid, "—")
        d = dense_set.get(sid, "—")
        l = lr_set.get(sid, "—")
        b_str = f"#{b}" if b != "—" else "—"
        d_str = f"#{d}" if d != "—" else "—"
        l_str = f"#{l}" if l != "—" else "—"
        print(f"  {i+1:>4}  {sid:<24}  {b_str:>6}  {d_str:>6}  {l_str:>7}")

    # Collect passage texts for LLM
    passage_texts: list[str] = []
    for item in fused[:TOP_K]:
        sid = item["skill_id"]
        skill = corpus_dict.get(sid, {})
        name = skill.get("name") or sid
        desc = (skill.get("description") or "").strip()
        content = (skill.get("content") or "").strip()
        full = f"{name}\n{desc}\n{content}".strip()
        passage_texts.append(full)

    # ── 9. LOCAL Qwen LLM answer ──────────────────────────────────────────────
    section(f"9. LLM ANSWER  (local Qwen via vLLM · {MODEL} @ {API_BASE})")
    from sragents.llm import create_llm_client

    client = create_llm_client(api_base=API_BASE)

    context_block = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{t[:800]}" for i, t in enumerate(passage_texts)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the retrieved skill passages "
                "below as context to answer the user's question. "
                "Be concise and show your reasoning step-by-step."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Relevant skill passages:\n\n{context_block}"
                f"\n\nQuestion: {QUERY}"
            ),
        },
    ]

    print(f"  Calling LLM ({MODEL}) … ", end="", flush=True)
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages, # type: ignore
            temperature=0.3,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
        print(f"done in {time.time() - t0:.1f}s.\n")
        print(textwrap.indent(answer.strip(), "  ")) # type: ignore
    except Exception as e:
        print(f"FAILED: {e}")
        print(
            "\n  Hint: ensure local vLLM is running:\n"
            f"    vllm serve {MODEL} --host 0.0.0.0 --port 8000 --max-model-len 8192\n"
            "  Or set API_BASE/MODEL env vars to point at your endpoint."
        )
    print()


if __name__ == "__main__":
    main()
