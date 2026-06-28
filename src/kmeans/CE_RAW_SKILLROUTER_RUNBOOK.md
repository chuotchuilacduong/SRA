# CE-Raw / SkillRouter Runbook (train + eval on H100)

End-to-end recipe for a **standalone CE pipeline trained on raw-corpus-mined data** (no M4
Stage-1 @100 pool, no fusion), inspired by SkillRouter (full-text retrieve-and-rerank,
hard-negative mining, false-negative filtering, listwise reranking). Everything is evaluated
on the **same held-out `query_gen-test` (1,079 queries)** as §20, so it drops straight into
`FULL_M4_V2_RESULTS.md` (a new **§21**).

## What's different vs the existing CE (M5/M7 = ce-joint-v3)

| | M5 / M7 (existing) | **CE-Raw (this runbook)** |
|---|---|---|
| Training negatives | mined from M4 **stage-1 `extended@100`** pool | mined from the **FULL 26,262-skill corpus** (bm25/bge/cluster/random) + SkillRouter false-neg filter |
| Inference candidates | rerank the **M4 @100/@500** pool | retrieve top-D from the **full corpus** (BGE / RRF / fine-tuned BGE) |
| Stage-1 fusion | M7 fuses `0.7·CE+0.3·M4`; M5 = raw CE | **none** (pure CE) |
| Retriever | M4/RRF (fixed) | base BGE, RRF, **or fine-tuned bi-encoder (Phase B)** |

Both train on `query_gen-train` and evaluate on `query_gen-test` → fair, leakage-free.

---

## 0. Push from laptop → pull on H100

Only **code** is committed (scripts/config/runbook); all data/models are regenerated on H100.

```bash
# laptop
git add src/kmeans/scripts/ceraw_*.py src/kmeans/configs/ceraw_listwise.yaml src/kmeans/CE_RAW_SKILLROUTER_RUNBOOK.md
# NOTE: docs/ and experiments/ are gitignored — the runbook lives under src/kmeans/ so it pulls on H100.
git commit -m "CE-Raw / SkillRouter standalone retrieve-and-rerank pipeline"
git push
# H100
cd /home/quynhtl/projects/SRA && git pull
```

## 1. Prerequisites on H100 (must already exist)

These are produced by the upstream pipeline / earlier rsync. Verify:

```bash
cd /home/quynhtl/projects/SRA && export PYTHONPATH=src
ls data/bench/corpus/corpus.json data/bench/instances/*.json
ls results/splits/*-query_gen.json
ls results/retrieval_bm25/*-bm25.json results/retrieval_dense/*-dense-bge.json
ls results/bge/corpus_emb.npy results/bge/corpus_ids.json results/clusters.json
ls results/m4_v2/cache/query_emb/*.npy
# optional (for §21 significance + context rows):
ls results/qsc_ltr/ce500/*.jsonl results/qsc_ltr/m5_500/*.jsonl results/comparisons/fair_supervised_query_gen.json
```

Python deps (GPU box): `torch` (CUDA), `transformers`, `sentence-transformers>=2.2`, `numpy`,
`scipy`, `pyyaml`. Base checkpoints are downloaded from HuggingFace on first use
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-base-en-v1.5`) — keep HF **online** for the
first run (do NOT set `HF_HUB_OFFLINE=1` until they're cached).

## 2. Phase 1 — build the raw dataset (CPU, ~2–5 min)

```bash
python src/kmeans/scripts/ceraw_build_dataset.py        # mix 4 bm25 / 3 bge / 2 cluster / 1 random
```
Writes `data/ce_raw/`: `ce_pairs_train.json`, `dev_pool.json`, `instances_all.json`,
`de_triples_train.jsonl`, `stats.json`. Check `stats.json` (pairs, source_counts,
`false_negatives_removed`).

## 3. Phase 2 — train CE-Raw (GPU/H100, ~10–30 min)

```bash
python src/kmeans/scripts/ceraw_train_ce.py \
    --config src/kmeans/configs/ceraw_listwise.yaml \
    --out results/models/ce-raw-v1
```
Listwise, full-text, MiniLM-L6 — identical recipe to ce-joint-v3; only the data is raw.
Saves the best dev-nDCG@10 checkpoint to `results/models/ce-raw-v1/`.

## 4. Phase B — fine-tune the retriever (full SkillRouter; GPU/H100, ~20–60 min)

```bash
python src/kmeans/scripts/ceraw_train_retriever.py \
    --epochs 2 --batch-size 64 --out results/models/sr-emb-bge-v1
```
InfoNCE (MultipleNegativesRankingLoss) over the same hard negatives → `sr-emb-bge-v1/`.
(Skip this step if you only want Phase A; then drop `bge_ft` from `--retrievers` below.)

## 5. Phase 3+4 — evaluate on query_gen-test + write §21 (GPU recommended)

```bash
# full first stage (no @100 cut); rerank a deep shortlist. depth 1000 captures ~all gold.
python src/kmeans/scripts/ceraw_eval.py \
    --retrievers bge_base,rrf,bge_ft \
    --rerank-depth 1000 \
    --ce-model results/models/ce-raw-v1
```
- `--rerank-depth full` reranks the **entire corpus** per query (≈26,262 × 1,079 ≈ 28M CE
  pairs — heavy: ~1–2 h on one H100). `1000` is the practical "no @100 cut" choice (BGE
  R@1000 ≈ ceiling; far cheaper). Use `500`/`2000` to trade speed vs ceiling.
- Outputs: `results/comparisons/ceraw_query_gen.{json,md}` and appends **§21** to
  `FULL_M4_V2_RESULTS.md` (CE-Raw·bge_base / ·rrf / ·bge_ft + retriever-only rows + §20
  context rows + significance vs held-out M5/M7-CE@500).

## 6. Smoke test first (recommended, ~2 min on GPU)

Catch wiring issues before the full run:
```bash
python src/kmeans/scripts/ceraw_build_dataset.py
python src/kmeans/scripts/ceraw_train_ce.py --out results/models/ce-raw-smoke    # 1 epoch is enough to verify it saves
python src/kmeans/scripts/ceraw_eval.py --retrievers bge_base --rerank-depth 100 --ce-model results/models/ce-raw-smoke
```
If §21 appears with sane numbers, run the full Phase 2/B/3 above.

## 7. Expectations (honest)

- First-stage recall **bounds** CE-Raw (the reranker cannot recover gold outside the shortlist).
  Per §20, full-corpus **BGE R@100 ≈ 85%**, **RRF ≈ 93%**, vs the M4 pool ~96%. So expect
  **CE-Raw·bge_base < M5/M7**; **CE-Raw·rrf** (and Phase-B **bge_ft**, which should lift recall)
  are the fair comparisons. Report retriever-only R@100 next to nDCG@10 to show the ceiling.
- This experiment measures the **cost of decoupling from the strong M4 stage-1** and whether a
  SkillRouter-style standalone CE is competitive on our 26K-skill pool.

## Files

| File | Phase | Device |
|---|---|---|
| `src/kmeans/scripts/ceraw_build_dataset.py` | 1 — raw dataset | CPU |
| `src/kmeans/configs/ceraw_listwise.yaml` | 2 — CE config | — |
| `src/kmeans/scripts/ceraw_train_ce.py` | 2 — train CE-Raw | GPU |
| `src/kmeans/scripts/ceraw_train_retriever.py` | B — fine-tune retriever | GPU |
| `src/kmeans/scripts/ceraw_eval.py` | 3+4 — eval + §21 | GPU |
