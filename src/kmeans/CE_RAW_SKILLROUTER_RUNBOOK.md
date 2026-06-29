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

Python deps (GPU box): `torch` (CUDA), `transformers`, `sentence-transformers>=2.2`,
`datasets` (required by ST's `.fit()` in Phase B — `pip install datasets`), `numpy`,
`scipy`, `pyyaml`. Base checkpoints are downloaded from HuggingFace on first use
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-base-en-v1.5`) — keep HF **online** for the
first run (do NOT set `HF_HUB_OFFLINE=1` until they're cached).

## 1b. Local execution on a Mac (Apple-GPU / MPS)

The whole pipeline runs locally without CUDA. The trainer/reranker/eval default to
CUDA-or-CPU (MPS off, since it was historically unstable for CE). Set **`SRA_ALLOW_MPS=1`**
to opt into Apple-GPU acceleration — this is the only change vs the H100 commands, and it is
a no-op on machines with CUDA (CUDA always wins) so the same scripts run unchanged on H100.

```bash
export PYTHONPATH=src SRA_ALLOW_MPS=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
```

Observed on an M-series (~64 GB unified): Phase 2 ≈ 25 min, Phase B ≈ 18 min, deep eval
(3 retrievers @1000) ≈ 2 h. **MPS memory caps matter for Phase B** — bge-base at the default
512 seq-len OOMs the unified pool; pass the memory-safe flags:

```bash
SRA_ALLOW_MPS=1 python src/kmeans/scripts/ceraw_train_retriever.py \
    --epochs 2 --batch-size 16 --max-negs 4 --max-seq-length 256 \
    --out results/models/sr-emb-bge-v1
```

On CUDA you can drop these flags (use the §4 batch 64 defaults). fp16 stays CUDA-only;
MPS runs fp32.

> Note: the trainer (`train_cross_encoder.py`) and the eval/retriever scripts are tracked and
> carry the `SRA_ALLOW_MPS` gate. The eval **reranker** lives in
> `src/sragents/retrieve/cross_rerank.py`, which is git-ignored in this repo (`.gitignore`),
> so its MPS gate is a local-only patch — H100/CUDA is unaffected (CUDA never takes the MPS
> branch); a fresh *local* clone wanting MPS during eval must re-apply that one-line gate.

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

## 5b. §22 — unified fair comparison vs ALL methods (no GPU, ~5 s)

After §20 (`run_fair_eval.py`) and §21 (`ceraw_eval.py`) have both written their JSONs,
merge them into one fair table — CE-Raw vs every other method on the **same query_gen-test**,
with the four per-dataset tables (Recall@1, Recall@10, nDCG@1, nDCG@10) + macro:

```bash
python src/kmeans/scripts/unified_compare.py     # appends §22 to FULL_M4_V2_RESULTS.md
```

Reads `results/comparisons/{fair_supervised_query_gen,ceraw_query_gen}.json` (no re-eval), so
it is safe to re-run after any new §21 eval. **Run order matters:** §21 must be written before
§22 (the §21 writer truncates everything from `## 21.` onward, which would drop a pre-existing
§22). Outputs `results/comparisons/unified_query_gen.{json,md}` + §22.

## 6. Smoke test first (recommended, ~2 min on GPU)

Catch wiring issues before the full run:
```bash
python src/kmeans/scripts/ceraw_build_dataset.py
python src/kmeans/scripts/ceraw_train_ce.py --out results/models/ce-raw-smoke    # 1 epoch is enough to verify it saves
python src/kmeans/scripts/ceraw_eval.py --retrievers bge_base --rerank-depth 100 --ce-model results/models/ce-raw-smoke
```
If §21 appears with sane numbers, run the full Phase 2/B/3 above.

## 7. Results (measured — local M-series/MPS run, query_gen-test, depth 1000)

First-stage recall **bounds** CE-Raw (the reranker cannot recover gold outside the shortlist):

| First stage | retriever R@100 | → CE-Raw nDCG@10 |
|---|---|---|
| bge_base | 84.63 | 55.22 |
| rrf | 92.80 | **56.48** (best CE-Raw) |
| **bge_ft** (Phase B) | **99.49** | 53.08 |

Reference lines on the same test: **L6-final nDCG@10 = 67.43**, **M7-CE@500 = 69.42**,
**M5-CE@500 = 61.44** (full unified table = §22).

Takeaways (honest):
- **Phase-B fine-tuning works as a retriever**: bge_ft lifts first-stage R@100 to **99.5%**
  (>> bge_base 84.6%, rrf 92.8%, even > the M4 pool ~96%). The SkillRouter encoder stage is the win.
- **But CE-Raw·bge_ft's nDCG@10 is *lower*** despite the higher recall: the cross-encoder was
  trained on bm25/bge/cluster negatives, so it ranks poorly inside bge_ft's much harder/broader
  top-1000. To realize bge_ft's recall, retrain the CE with **bge_ft-mined hard negatives**
  (re-run Phase 1 sourcing from `sr-emb-bge-v1`, then Phase 2). CE-Raw·**rrf** is the best CE-Raw today.
- CE-Raw (standalone) trails pipeline-CE (M5/M7, which rerank the strong M4 @500 pool):
  CE-Raw·rrf vs M7-CE@500 is **−10.3 pp nDCG@10** (p≈0, §21.3). This quantifies the **cost of
  decoupling from the M4 Stage-1** — the core question this experiment answers.

## Files

| File | Phase | Device |
|---|---|---|
| `src/kmeans/scripts/ceraw_build_dataset.py` | 1 — raw dataset | CPU |
| `src/kmeans/configs/ceraw_listwise.yaml` | 2 — CE config | — |
| `src/kmeans/scripts/ceraw_train_ce.py` | 2 — train CE-Raw | GPU |
| `src/kmeans/scripts/ceraw_train_retriever.py` | B — fine-tune retriever | GPU |
| `src/kmeans/scripts/ceraw_eval.py` | 3+4 — eval + §21 | GPU |
| `src/kmeans/scripts/unified_compare.py` | 5b — merge §20+§21 → §22 (all methods) | CPU |
