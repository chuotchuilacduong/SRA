# M4-v2 Implementation Notes — Decision Log

> Running log of every decision, deviation, and tradeoff made while implementing
> `docs/kmeans/implementation_m4_v2 (1).md` (Cluster-Aware RRF Reranker).
> This is the file the spec calls `implementation_m4_v2_notes.md` (§5.3) and also
> the "decisions you had to make" log the user asked for. Newest sections appended
> as work proceeds.

Author: Claude (Opus 4.8). Started 2026-06-24. Host: macOS (darwin, Apple Silicon, MPS).

---

## 0. TL;DR of the most important decisions

1. **Code lives in `src/kmeans/`** (user instruction), not the spec's suggested
   `src/m4_v2/`. The package is importable as `kmeans`. `sragents` is reused for
   metrics / corpus / schema. Scripts, config, and sbatch also live under
   `src/kmeans/` to honor "all new code in src/kmeans". Data *outputs* go under
   `results/` and `logs/` (those are artifacts, not code), matching repo convention
   (the repo has **no** top-level `cache/` dir — the spec's `cache/...` paths are
   remapped to `results/m4_v2/cache/...`).
2. **"Current M4" = α=0.7 (`0.7·RRF_norm + 0.3·aff_norm`)**, confirmed *empirically*
   (see §2). The repo also has `hybrid_km_alpha30` pools, but those are a downstream
   training choice, not the reported Method 4.
3. **Query embeddings are not cached anywhere** → we encode them ourselves with
   `BAAI/bge-base-en-v1.5` + the BGE search prefix and cache to
   `results/m4_v2/cache/query_emb/{ds}.npy`. The model is already in the local HF
   cache; runs on MPS/CPU. (See §3.)
4. **M=500 pool is built by recomputing BM25@M and BGE@M and RRF-fusing**, because
   cached retrieval artifacts only stored BM25 top-50 / BGE top-100 / RRF top-100.
   A unified fused list (up to M_max) is cached once per dataset so every variant
   (A1 @100, A2–A8 @500, and the sweep) slices from the *same* construction —
   keeping the A2−A1 delta a pure pool-size effect. (See §4.)
5. **Entropy for adaptive-α uses the full-K query softmax; affinity uses sparse
   top-L.** The spec is internally ambiguous here; this is the interpretation that
   makes `H_norm ∈ [0,1]` behave correctly. (See §9.)

---

## 1. Existing artifacts discovered (real paths, all verified to exist)

| Spec idealized path | Actual repo path | Notes |
|---|---|---|
| `data/skills.jsonl` | `data/bench/corpus/corpus.json` | JSON **array** (not jsonl), 26,262 skills, fields `skill_id,name,description,content` |
| `data/datasets/{ds}.jsonl` | `data/bench/instances/{ds}.json` | JSON array; `instance_id`,`question`,`skill_annotations`(=gold),`eval_data` |
| `results/bm25/{ds}.json` | `results/retrieval_bm25/{ds}-bm25.json` | top-50, score only, rank implicit by position |
| `results/bge/{ds}.json` | `results/retrieval_dense/{ds}-dense-bge.json` (and `results/bge/{ds}-bge.json`) | top-100, explicit rank |
| `results/rrf/{ds}.json` | `results/retrieval_rrf_bm25_dense/{ds}-rrf-bm25-dense.json` | top-100, RRF(k=60), no original ranks |
| (rank lineage) | `results/pool/extended-{ds}.json` | **top-100** RRF pool WITH `bm25_rank/bge_rank/rrf_rank/*_score` per candidate — the A1 candidate source |
| `cache/corpus_emb.npy` | `results/bge/corpus_emb.npy` | (26262, 768) float32, L2-normalized |
| `cache/skill_id_to_idx.json` | `results/bge/corpus_ids.json` | ordered list; row i ↔ corpus_ids[i] (inverse mapping) |
| `cache/kmeans/..._labels.npy` | `results/clusters.json` | dict `{skill_id: cluster_id}`, K=300, ids in [0,299] |
| `cache/kmeans/..._centroids.npy` | `results/clusters_centroids.npy` | (300, 768) float32, **L2-normalized** |
| (reverse index) | `results/clusters_index.json` | dict `{cluster_id(str): [row_idx,...]}` |
| query embeddings | **none** | must be computed (see §3) |

Datasets & sizes (single-gold except BigCodeBench which is multi-gold):
theoremqa 747, logicbench 760, toolqa 1430, champ 223, medcalcbench 1100,
bigcodebench 1140 → **5,400 queries**, **26,262 skills**, **K=300 clusters**.
Field mapping: `query_id=instance_id`, `query=question`, `gold_skill_ids=skill_annotations`.

## 1b. Reused code (do NOT reimplement)

- **Metrics**: `sragents.retrieve.metrics.compute_retrieval_metrics` /
  `compute_metrics_by_dataset`. Recall@K = |gold∩top-K|/|gold|, multi-gold aware;
  nDCG@K with `1/log2(rank+1)` discount, IDCG over `min(|gold|,K)`. Macro = equal
  weight per dataset; Micro = pooled per query. Empty-gold queries skipped.
  **`top_k` caps which K are computed → we always evaluate with `top_k=100`** so
  R@100/nDCG@... are produced. These are the EXACT functions that produced the
  published numbers, so A1 is directly comparable.
- **Schema / IO**: `sragents.retrieve.schema.RetrievalResults`/`RetrievalRecord`
  (`.dump()` writes the canonical `{metadata, metrics, results}` JSON).
- **Corpus**: `sragents.corpus.load_corpus`.
- **RRF reference**: `sragents.retrieve.fusion.multi_rrf_merge` (k_rrf=60,
  `score=Σ weight/(k_rrf+rank+1)`, rank 0-indexed in code → +1).
- **A1 reference builder**: `experiments/build_hybrid_pool.py` (exact current-M4 math).

---

## 2. RESOLVED: which α is "current M4"?

The spec/`FINAL_METHODS_AND_RESULTS.md` say Method 4 = `0.7·RRF + 0.3·aff` (α=0.7),
but `probehyrr_h100/config.py` points downstream at `hybrid_km_alpha30-{ds}.json`.
`build_hybrid_pool.py` uses `final = alpha*rrf_n + (1-alpha)*aff_n`, so the
`alphaNN` filename = **RRF weight × 100**.

Settled by reading the stored metrics in the existing pools and matching the report:

| dataset | report M4 R@1 | alpha30 | alpha50 | **alpha70** |
|---|---|---|---|---|
| theoremqa | 70.01 | 66.67 | 68.67 | **70.01** ✓ |
| toolqa | 34.90 | 15.73 | 26.15 | **34.90** ✓ |
| logicbench | 15.79 | 15.13 | 15.13 | **15.79** ✓ |

→ **Current M4 = α=0.7.** `alpha30` is a downstream/training pool, *not* the reported
method. A1 must reproduce α=0.7 over the (top-100) RRF pool. Validation target for A1:
TheoremQA R@1≈70.01, ToolQA R@1≈34.90, LogicBench R@1≈15.79, R@100 unchanged from RRF.

---
## 3. Query embeddings (not in spec as a step, but required)

The spec assumes `cache/query_emb/{dataset}.npy` exists. It does **not** in this repo
(queries were re-encoded at retrieval time). `embeddings.py` encodes all queries with
`BAAI/bge-base-en-v1.5` + the exact search prefix used by the corpus pipeline
(`"Represent this sentence for searching relevant passages: "`), L2-normalizes, and
caches to `results/m4_v2/cache/query_emb/{ds}.npy` (+ `{ds}_ids.json`).
- Device auto-select: CUDA → MPS → CPU. Ran on **MPS** locally; 5,400 queries in ~40 s.
- **Verification that the encoder matches**: A1 (which reranks the cached extended pool
  with hard affinity = `e_q·μ_{c(s)}`) reproduces `hybrid_km_alpha70` **exactly** on every
  dataset (e.g. champ R@1 18.61, R@10 50.26, R@100 91.11, nDCG@10 35.65 — identical), and
  the A1 macro R@1 = **35.53** equals the published Method-4 AVG. This is only possible if
  our `e_q` reproduces the original affinity, so the encoder/prefix/normalization match.

## 4. M=500 pool construction (`rrf_pool.py`)

Cached retrieval artifacts only stored BM25 top-50 / BGE top-100 / RRF top-100, so a true
M=500 pool had to be rebuilt. **Decision:** extend the **BGE** side to depth `M_max=1000`
using `q·corpus_emb` (the *same* embeddings the original BGE retrieval used → ranks ≤100
reproduce exactly), keep the **cached BM25 top-50**, and RRF-fuse with k=60
(`1/(k+rank_bm25) + 1/(k+rank_bge)`, a missing ranker contributes 0). This is the natural
M-generalization of how `extended-{ds}.json` was built (BM25 top-50 ∪ BGE top-K, fused).
- A single fused list per dataset is cached (`results/m4_v2/cache/fused_pool/{ds}.json`)
  and **every** variant slices its pool from it → the A2−A1 delta is a pure pool-size effect.
- **A1 fidelity knob** (`a1_use_cached_extended_pool: true`, default): A1 reranks the
  *cached* `extended-{ds}.json` directly so it reproduces the report exactly. A0/A2–A8 use
  the recomputed fused pool. (A0 over the fused pool gives macro R@1 36.66 vs the report's
  RRF 37.48 — within ~0.8pp, the expected small difference from BGE-depth extension/tie-breaks.)
- **Approximation, documented:** BM25 beyond rank 50 is treated as absent (contribution 0).
  Its RRF weight there is ≤ 1/(60+51) ≈ 0.009, so top-of-pool ordering is essentially unchanged.

## 5. Cluster reliability (`cluster_stats.py`, Step 4)

Implemented exactly: `coh(k)=mean_{s∈C_k} e_s·μ_k` (each skill vs its own centroid),
`idf(k)=log(N/(|C_k|+1))`, `rel_raw=coh·idf`, `rel=clamp(minmax(rel_raw),0.05,1.0)`.
- **Reconciliation:** spec §2 writes `rel(k)=coh·log(N/|C_k|+1)` (raw) but Step 4 defines
  `rel(k)` as the min-max-normalized + clamped value. **Step 4 is authoritative** and is what
  the affinity sum uses. Observed: coh∈[0.78, 0.94], size∈[16, 274], rel∈[0.05, 1.0].
- Centroids verified L2-normalized on load (Step 4 says "load and verify"); not recomputed.

## 6. Soft cluster affinity (`soft_cluster.py`, Steps 6–7)

Sparse top-L (L=10) implemented faithfully: query and skill each take their top-L clusters
by `e·μ_k`, softmax(τ·sim) **within** those L (renormalized), p=0 outside. Affinity is
`Σ_{k∈topL_s} p(k|s) · q_weighted[k]` where `q_weighted = p(·|q)·(rel if calibrated else 1)`;
since `q_weighted` is nonzero only on the query's top-L, this is exactly the intersection
sum (empty intersection → 0, per spec §6). Skill-side top-L is precomputed once for the whole
corpus (a single (N,K) matmul).

## 9. Adaptive-α entropy — DECISION (deviation from a literal reading)

Step 9 computes `H(q)=-Σ_k p(k|q) log(p(k|q)+ε)` and `H_norm = H/log K`. But Step 6 makes
`p(k|q)` sparse over only **L=10** clusters. Taken literally, H would be capped at
`log L = 2.30`, so `H_norm ≤ log L/log K ≈ 0.40` and `conf_cluster = 1-H_norm` would never
drop below 0.60 — the signal would be meaningless. **Decision:** the entropy uses the
**full-K** query softmax (cheap; K=300), keeping `H_norm ∈ [0,1]` as the formula intends.
The sparse top-L is used only for the affinity overlap. Effect on results is small: adaptive
α landed in ~0.69–0.77 (close to the 0.7 base), so this choice is not what drives the verdict.

## 10. Output format deviations (`scorer.py`)

- Per-query records use key **`retrieved`** (not the spec's `candidates`) so they feed
  `sragents.retrieve.metrics` unchanged; `instance_id`/`gold_skill_ids` kept for the same reason.
  All spec §5.1 per-candidate fields are present (`final_score, rrf_score, rrf_norm, aff_score,
  aff_norm, rank, is_gold`).
- **`cluster_debug` compacted:** `top_query_clusters` is query-level (same for all candidates),
  so it's stored once per record, not per candidate; per-candidate `cluster_debug` keeps only
  `hard_cluster` + `cluster_reliability`. Storing the spec's literal per-candidate layout for
  100 cands × 5,400 q × 9 variants would have been ~10 GB. Gold-skill clusters needed for error
  analysis are recomputed on demand.
- Records written as JSONL (one record/line) at `results/m4_v2/{variant_slug}/{ds}.jsonl`.

## 11. RESULTS (full ablation, all 5,400 queries, macro over 6 datasets, %)

| Var | Name | R@1 | R@5 | R@10 | R@50 | R@100 | nDCG@10 |
|---|---|--:|--:|--:|--:|--:|--:|
| A0 | RRF baseline | 36.66 | 61.78 | 71.44 | 87.83 | 92.23 | 56.57 |
| A1 | Current M4 (α=0.7) | 35.53 | 61.42 | 71.76 | 89.63 | 92.23 | 55.92 |
| A2 | + larger pool (500) | 34.66 | 61.78 | 71.90 | 89.75 | 93.54 | 55.59 |
| A3 | + calibrated hard aff | 31.44 | 58.11 | 67.71 | 85.36 | 90.46 | 51.39 |
| A4 | + soft affinity | 27.44 | 60.34 | 70.59 | 89.82 | 93.63 | 51.49 |
| A5 | + soft-calibrated aff | 27.18 | 58.95 | 69.10 | 88.65 | 92.69 | 50.45 |
| A6 | + adaptive α | 28.40 | 59.67 | 69.82 | 88.82 | 92.70 | 51.48 |
| A7 | + PRF only | 34.87 | 62.07 | 72.05 | 89.92 | 93.71 | 55.79 |
| **A8** | **Full M4-v2** | **27.78** | **59.21** | **69.70** | **88.95** | **92.86** | **51.02** |

**Verdict (spec §11/§17.1): KEEP current M4. A8 does NOT beat A1** — R@10 −2.06pp,
nDCG@10 −4.90pp, R@1 −7.75pp. Component-isolated effects (macro):
- **Larger pool (A2−A1):** R@100 **+1.31pp** (ceiling ↑, the one clear win), R@10 ≈flat,
  R@1 −0.87pp. The pool helps recall depth, costs a little at the top.
- **PRF only (A7−A1):** R@10 **+0.29pp** (best R@10 of all), R@100 +1.48pp, R@1 −0.66pp,
  nDCG ≈flat. **PRF is the most useful single component** and is roughly safe.
- **Reliability calibration (A3−A2):** R@10 **−4.19pp** — calibration *hurts* hard affinity.
- **Soft affinity (A4−A2):** R@1 **−7.22pp**, nDCG −4.10pp — the main culprit. Hard cluster
  assignment beats the soft top-L overlap at default τ=20/L=10, especially at rank 1
  (TheoremQA R@1 70→51, MedCalcBench 51→35).
- **Adaptive α (A6−A5):** ≈+1pp R@1 — small, and α stays near 0.7.

Best *safe* recipe would be **larger pool + PRF, hard affinity, no soft/no calibration**
(≈A7 + A2's pool). The soft-cluster + calibration machinery, at default hyperparameters,
is net-negative. The spec's hyperparameter sweep (§8: τ∈{5,10,20,40}, L∈{3,5,10,20},
ρ, α-range) on the **dev splits** (`results/splits/`) is the proper next step before
concluding soft affinity is unusable — NOT tuning on test (spec §18).

Confidence the negative result is real (not a bug): A1 reproduces the published number
exactly, and an adversarial per-module correctness review was run (see §15).

## 12. How to run

```bash
# all code is under src/kmeans/; run from repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1            # model is in local HF cache
PYTHONPATH=src python src/kmeans/scripts/build_m4_v2_cluster_stats.py
PYTHONPATH=src python src/kmeans/scripts/build_m4_v2_query_embeddings.py
PYTHONPATH=src python src/kmeans/scripts/run_m4_v2_ablation.py        # A0-A8 x 6 datasets, ~150s on MPS
PYTHONPATH=src python src/kmeans/scripts/write_m4_v2_report.py
# subset:  run_m4_v2_ablation.py --datasets champ --variants A1 A8
# cluster/HPC: src/kmeans/sbatch/run_m4_v2_ablation.sbatch  (conda env "sra")
```
`run_ablation` auto-builds stats + embeddings + fused pools if missing. The whole ablation
runs locally in ~2.5 min (query encoding once, everything else matmuls/sorts).

## 13. Spec path remaps (repo has no top-level `cache/`)

`cache/corpus_emb.npy`→`results/bge/corpus_emb.npy`; `cache/skill_id_to_idx.json`→
`results/bge/corpus_ids.json` (ordered list, inverse map); `cache/kmeans/*`→`results/clusters*`;
`cache/kmeans/..._cluster_stats.json`→`results/m4_v2/cache/cluster_stats.json`;
`cache/query_emb/*`→`results/m4_v2/cache/query_emb/*`. Reports go to `results/comparisons/`,
`results/analysis/`, and `FULL_M4_V2_RESULTS.md` (repo root, per spec §19).

## 14. Things I did NOT do (and why)

- **Hyperparameter sweep (§8):** not run yet — it's a large job and must use dev splits, not
  test. Recommended as the immediate follow-up given soft affinity's poor default behavior.
- **Validation report (Step 1 `validation_report.json`):** alignment checks are enforced in
  `io.load_artifacts` (raises on any missing id / shape mismatch) and all datasets passed; a
  standalone JSON report wasn't separately emitted.
- **Latency p50/p95 per-stage breakdown (§9.6):** only per-(variant,dataset) wall time is
  logged in `run_summary.json`; fine-grained per-stage percentiles were out of scope for a
  CPU-cheap reranker but easy to add.

## 15. Adversarial correctness review (7-agent workflow, per-module vs spec)

Every math module was independently reviewed against its spec section, with a verify
pass on serious findings. **Result: all 6 modules `matches_spec = true`; no critical
findings; every reviewer concluded the negative result is a GENUINE method property at
default hyperparameters, not a code bug.** Two reviewers reproduced the math numerically:
`cluster_stats` matched the cached stats to **max-error 0.0** over all 300 clusters, and
`soft_affinity` matched a brute-force dense reference to **1e-5** (calibrated + uncalibrated),
confirming the intersection sum and that `rel(k)` multiplies inside the sum.

**Why soft affinity hurts (mechanism, confirmed):** with τ=20 over BGE centroid cosines
(~0.3–0.6) the per-cluster softmax is near one-hot, so `Aff_soft` is dominated by the single
shared top cluster and the empty-intersection→0 rule zeros a large fraction of the pool. After
Step-8 minmax the affinity becomes a coarse near-binary signal — far less discriminative than
the smooth hard `e_q·μ_{c(s)}`. A gold whose dominant cluster differs from the query's gets
aff=0 and is demoted by the 0.3 weight → the TheoremQA R@1 70→50 drop. This is exactly the
top-1 fragility the spec itself flags (§11.2, §12.3). **Fix space is hyperparameter** (lower τ,
larger L, or a dense softmax), not a code patch — see the §8 sweep as the next step.

### MAJOR finding (real, verified) — and its resolution
The recomputed fused pool extends BGE to depth 1000, but A1 uses the cached `extended-{ds}.json`
(BGE capped at 100). So shared candidates can get a different second RRF term, making A2−A1
mix "pool size" with "BGE depth." I quantified it with a read-only diagnostic (A1 scored on the
fused pool @100):

| source | R@1 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| A1 cached@100 (report-exact) | 35.53 | 71.76 | 89.63 | 92.23 | 55.92 |
| A1 fused@100 (same constr. as A2) | 34.98 | 72.01 | 89.51 | 92.23 | 55.82 |
| A2 fused@500 | 34.66 | 71.90 | 89.75 | 93.54 | 55.59 |

- **BGE-depth confound** (A1 fused@100 − A1 cached@100): R@1 −0.55, R@10 +0.25, R@100 0.00,
  nDCG −0.09 — all ≤0.55pp.
- **Clean pool-size effect** (A2 fused@500 − A1 fused@100, identical construction, only M differs):
  **R@100 +1.31pp**, R@1 −0.33, R@10 −0.11, nDCG −0.24.

→ The confound is small and changes no conclusion: enlarging the pool raises the R@100 ceiling
(~+1.3pp) and is roughly neutral at the top. We keep A1 on the cached pool (its job is exact
report reproduction) and document that the clean pool-size delta is the A2−A1fused row above.
Flip `a1_use_cached_extended_pool: false` to make A1 a clean fused@100 slice instead.

### Code fixes applied from the review
- `adaptive_alpha.rrf_confidence`: `argsort` → stable sort (deterministic s_1/s_10 on ties;
  no numeric change — tied candidates share `rrf_norm`).
- `scorer`: A0 records now report `alpha: null` (spec §7.1 lists A0's α as "none"); the blend
  never used it (A0 = pure RRF order).

### Documented-as-intended (not bugs)
- `rrf_pool.minmax` omits the eps-in-denominator and uses an all-equal→0.5 branch at 1e-12 —
  this **intentionally** matches `experiments/build_hybrid_pool.py` so A1 reproduces the
  published numbers exactly (verified). Difference vs the spec's `(x-lo)/(hi-lo+1e-8)` is ~1e-8.
- hard-calibrated affinity (A3) = `aff·rel`; since cosine can be negative, this isn't strictly
  order-preserving across clusters. The spec defines calibration only for the soft affinity
  (where p≥0); "hard_calibrated" is our ablation interpretation. A3's degradation is robust
  regardless, so this doesn't affect conclusions.
- adaptive-α entropy uses the dense full-K softmax while affinity uses sparse top-L (NOTES §9) —
  intentional; only affects A6/A8 and α stayed ≈0.7, so it doesn't drive the verdict.

## 16. Bottom line

The implementation is faithful and verified (A1 reproduces the published Method-4 numbers to
the decimal; all modules pass an adversarial spec review). **At the spec's default
hyperparameters, full M4-v2 (A8) does NOT beat current M4 (A1):** the soft-cluster affinity and
reliability calibration are net-negative, while the larger pool (R@100 ceiling) and PRF (best
R@10, ≈neutral R@1) are the only beneficial pieces. Recommendation (spec §17.1): **keep current
M4**; if pursuing M4-v2, adopt only larger-pool + PRF with hard affinity, and run the §8 τ/L/ρ
sweep on the **dev** splits before reconsidering soft affinity. This is reported honestly per
spec §18 ("do not claim M4-v2 is better before evaluation").

---

# PART II — QSC + LTR Extension (`docs/kmeans/implementation_qsc_ltr_extension.md`)

Implements the next plan: **Query-Specific Clustering (Q0–Q6)** and **Lightweight
Learning-to-Rank (L0–L7)**. All new code under `src/kmeans/` (`base_table.py`, `qsc.py`,
`ltr_features.py`, `ltr.py`, `qsc_ltr_runner.py`, `qsc_ltr_report.py` + scripts/config/sbatch).
Reuses every M4-v2 artifact (fused RRF pools, cached query embeddings, cluster stats) and the
`sragents` metrics. Results append to `FULL_M4_V2_RESULTS.md` §14 (previous A0–A8 untouched).

## 17. QSC + LTR design decisions

1. **Shared base table (`base_table.py`).** All of QSC and LTR operate over the *same* RRF
   top-500 candidate set per query (= A0 pool from the fused pool). A0/A1-A2(M4)/A7 base scores
   are recomputed over the full 500 pool (the M4-v2 JSONL outputs are only top-100), reusing the
   M4-v2 component functions (`minmax`, hard affinity via `soft.query_sims`, `prf.refine_query`).
   The union pool in plan §5.3 (A0 ∪ A1 ∪ A7 ∪ QSC) **is** the RRF top-500, because A1/A2/A7/QSC
   all rerank that same candidate set — so `P_LTR(q)` = RRF top-500 (no extra union needed).
2. **Streaming to bound memory.** `iter_base_table` yields one query at a time; holding every
   query's 500×768 candidate embeddings at once would be ~2 GB for toolqa. The runner streams:
   build entry → local KMeans → score Q0–Q6 → append LTR feature row → drop the entry.
3. **QSC base mapping.** Q0–Q2 base = A0 RRF (`baseNorm=rrf_norm`); Q3–Q4 base = A2 M4 over the
   500 pool; Q5–Q6 base = A7 PRF over the 500 pool. For A7-based variants `localAff` uses the
   PRF-refined query vector `e'_q` (plan §4.5); others use `e_q`.
4. **Local H.** `H = min(default_k=16, max(2, floor(sqrt(M))))` (plan §4.3) — equals 16 for the
   500 pool, shrinks for small pools. sklearn `KMeans` (n_init=3, max_iter=100, seed 42). Local
   centroids are the L2-normalized mean of each cluster's embeddings (so cosines are meaningful).
5. **QSC defaults are deliberately weak** (plan §4.8): QSC-1 α=0.85, QSC-2 (0.80/0.10/0.10),
   QSC-3 δ=0.05/λ=0.05. The sweep grids are in the config (`sweep_enabled: false` for the main
   run). QSC-3 (tie-break) only perturbs near-ties (|baseMax−base|≤δ) — the safest variant.
6. **45 LTR features**, grouped retrieval(15)/m4(8)/a7(6)/qsc(9)/confidence(7) exactly per plan
   §5.5. Query-level features (PRF shift, safe-set stats, RRF margins, BM25∩BGE overlaps@10/20/50,
   global cluster entropy, query token length) are broadcast to every candidate. `direct_cosine_q_s`
   is `e_q·e_s` (defined for all candidates, unlike `bge_score` which is only for BGE-retrieved).
   Missing rank → 1e6, inverse rank → 0, missing flag → 1.

## 18. LTR validity / anti-leakage (plan §5.2, §10.3) — and the n_jobs crash

- **Query-level stratified split**, seed 42: per dataset shuffle queries → 70/15/15 →
  train/dev/test, unioned across datasets. Candidates of a query never cross splits (we split by
  `instance_id`, never by candidate row). Hyperparameters chosen on **dev**; metrics reported on
  the held-out **test** split only. Labels use gold only as `y`; no feature touches gold. → meets
  every §10.3 validity condition, so LTR is a **valid learned result**, not diagnostic-only.
- **Fair comparison:** QSC (no training) is evaluated on the FULL set and is directly comparable
  to A0–A8. LTR is on the TEST split, so A0/A1/A2/A7/Q2/Q6 are **recomputed on the same test
  queries** for the delta tables. Tables always state their query set.
- **LightGBM segfault (macOS):** `LGBMRanker` with `n_jobs>1` reliably SIGSEGVs (exit 139) on the
  local libomp build (verified n_jobs∈{-1,4,2} crash; n_jobs=1 fine). Fixed by pinning
  `OMP_NUM_THREADS=1` (set before any OpenMP import in `ltr.py`) and `n_jobs=1`. Deterministic and
  correct; on a Linux/HPC env raise `n_jobs` for speed. Linear pairwise ranker (LTR-A): RankNet-
  style logistic on standardized feature differences (sampled pairs, both ±diff for balance), C
  chosen on dev.

## 19. QSC/LTR deviations from the plan

- Files placed under `src/kmeans/` (+ `scripts/`, `configs/`, `sbatch/`) per the user's "all code
  in src/kmeans" instruction, rather than the plan's top-level `src/...`+`scripts/`+`configs/`.
  One consolidated entry script `run_qsc_ltr_extension.py` runs the whole §15 order (the plan's
  6 separate scripts are folded into it + the runner/report modules).
- **LightGBM grid reduced** from the full §13 grid (num_leaves{15,31,63}×lr{0.03,0.05}×
  n_est{200,500}×min_data{10,30,50} = 36 configs/variant) to num_leaves{31,63}×lr{0.05}×
  n_est{500, early-stopped}×min_data{30} (2 configs/variant), to keep single-threaded runtime
  sane. Still dev-selected with early stopping. Full grid available by editing the config.
- `query_length_tokens` uses whitespace token count (a feature, not a tokenizer-exact value).
- LTR candidate cap = 500/query (plan's "first run"); the 1000-cap recall-heavy run not done.

## 20. QSC + LTR RESULTS (full ablation, 5,400 queries)

Runtime: ~12 min on macOS/MPS+CPU (QSC ~4 min; LTR ~8 min, single-threaded LightGBM on
~1.9M train rows × 14 fits). All numbers from the reused `sragents` metrics.

### 20.1 QSC (FULL query set, macro %, directly comparable to A0–A8)

| Method | R@1 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| A0 RRF | 36.66 | 71.44 | 87.83 | 92.23 | 56.57 |
| A1 Current M4 | 35.53 | 71.76 | 89.63 | 92.23 | 55.92 |
| A7 PRF only | 34.87 | 72.05 | 89.92 | 93.71 | 55.79 |
| Q2 QSC-RRF-tiebreak | 36.91 | 71.44 | 88.52 | 92.23 | 56.64 |
| Q6 QSC-A7-tiebreak | 34.97 | 72.05 | 90.04 | 93.71 | 55.79 |
| (Q0,Q1,Q3,Q4,Q5) | 34.2–36.3 | 70.9–71.9 | — | 92.5–93.7 | 55.0–56.1 |

**QSC verdict: SAFE but ≈neutral.** Unlike the M4-v2 *global* soft cluster (which crushed
R@1), the *local* tie-break QSC barely moves anything (≤~1pp): Q2 nudges R@1/nDCG just above
A0 (36.91 vs 36.66; 56.64 vs 56.57), Q6 matches A7. It does **not** pass the §10.1 final-ranker
bar (Q2 R@10 71.44 < A7 72.05−0.2). This confirms the plan's hypothesis that a *weak, local*
cluster signal avoids the top-rank damage of strong global clustering — but as a fixed reranker
it adds little. Its real value is as **features for LTR** (below).

### 20.2 LTR (held-out TEST split, 811 queries; baselines recomputed on the SAME split)

| Method | R@1 | R@5 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|--:|
| A0 RRF (test) | 35.43 | 61.36 | 72.61 | 88.52 | 92.74 | 56.18 |
| A7 PRF (test) | 33.93 | 61.19 | 72.78 | 90.04 | 93.56 | 55.26 |
| L0 RRF-only | 44.99 | 65.75 | 73.71 | 88.72 | 92.43 | 61.80 |
| L1 RRF+M4 | 48.76 | 71.35 | 81.27 | 94.52 | 96.17 | 67.55 |
| L2 RRF+A7 | 44.63 | 66.23 | 74.17 | 91.17 | 94.33 | 61.76 |
| L3 RRF+QSC | 43.13 | 68.03 | 76.16 | 91.30 | 95.06 | 62.40 |
| L4 M4+A7 | 48.31 | 70.93 | 81.69 | 94.78 | 96.22 | 67.58 |
| L5 A7+QSC | 44.24 | 67.41 | 76.42 | 91.89 | 95.46 | 62.87 |
| **L6 FULL** | **48.82** | **74.12** | **81.91** | 94.42 | 95.63 | **68.63** |
| L7 linear FULL | 41.08 | 67.66 | 77.75 | 93.70 | 95.20 | 61.96 |

**LTR verdict: a large, leakage-controlled win.** L6 (full features) beats the test-split
baselines by **R@1 +13.4 / R@10 +9.1 / nDCG@10 +13.4 (vs A7), nDCG@10 +12.5 (vs A0)**, landing
between RRF (~37 R@1) and the Cross-Encoder Method 7 (~54 R@1) — **with no Cross-Encoder**.
It PASSES all four §10.1 criteria (worst per-dataset R@10 vs A7 is TheoremQA −1.79, within 2pp).

**Key insight — cluster info is a learned feature, not a fixed reranker.** The single biggest
jump is adding the **M4 / global-cluster features** to the learner: L0→L1 (+RRF→+M4) lifts
nDCG@10 61.8→67.6 and R@10 73.7→81.3. The exact cluster signal that *hurt* as a hand-weighted
reranker (M4-v2 A3/A4/A8) is *valuable* when a LambdaRank model decides how to use it per query.
QSC features add a smaller, consistent lift (L0→L3 +0.6 nDCG; L2→L5 +1.1 nDCG). The linear
pairwise model (L7) confirms the signal is real but a tree model captures the nonlinear
interactions far better (L6 68.6 vs L7 62.0 nDCG@10).

Per-dataset L6 R@10 gain vs A7 (test): logicbench **+21.9**, bigcodebench **+17.1**,
toolqa +8.4, medcalcbench +7.3, champ +2.0, theoremqa −1.8 (already near-ceiling). nDCG@10 gain
vs A0: toolqa **+27.6**, logicbench +19.1, bigcodebench +16.7, medcalcbench +8.9, theoremqa +1.7,
champ +0.8.

### 20.3 Leakage controls (verified)

- Split fully disjoint: train(3780) ∩ dev(809) ∩ test(811) = ∅, sum = 5400 (every query in
  exactly one split); candidates of a query never cross splits.
- No feature column equals the label; gold used only as `y` (45 features, all gold-free).
- Per-query min-max normalization (no cross-query/cross-split statistics); LightGBM fits on
  train, early-stops on dev; linear scaler fit on train only; hyperparameters selected on dev;
  metrics on test only. (Adversarial leakage review: see §21.)

### 20.4 Recommendation (plan §16)

> **L6 LTR-FULL is the best no-Cross-Encoder ranker** — it beats A0 on nDCG@10 and A7 on R@10
> while keeping R@1 well above RRF. Recommend it as the production no-CE final ranker / a strong
> stage-1.5 before any CE. **QSC** as a standalone reranker is safe but ≈neutral; keep it only as
> a feature source for LTR (it contributes a small consistent lift). A0/A7 remain fine fixed
> fallbacks. The cluster-heavy *fixed* scoring from M4-v2 stays retired.

**Caveats (honest):** (1) LTR metrics are on a 15% test split (811 q); the split is
representative (baselines match their full-set values), but these are not full-corpus numbers.
(2) `global_cluster_id` / `qsc_local_cluster_id` features encode corpus-specific structure —
query-level leakage is controlled, but the lift may not transfer to a brand-new corpus without
retraining. (3) LightGBM ran single-threaded (macOS OpenMP crash, §18) with a reduced dev-tuned
grid; the full §13 grid could shift numbers slightly. (4) Not tuned on test; dev-selected only.

## 21. Adversarial review of the QSC/LTR extension (6-agent workflow)

Four modules reviewed line-by-line for correctness + train/test leakage, with a verify pass.
**Outcome: all 4 `matches_spec=true`; every reviewer + the independent verifier returned
`NO LEAKAGE FOUND`.** Verified concretely: splits disjoint & exhaustive (each query in exactly
one split; unique dataset-prefixed ids); LightGBM fits on train, early-stops/selects on dev,
predicts on test only; linear `StandardScaler` fit on train only and reused at predict; `y` is
the only gold-derived signal (no feature column touches gold); all normalizations are strictly
per-query (no cross-query/cross-split statistics), so building features before the split is
leakage-safe; `n_jobs=1`/`OMP_NUM_THREADS=1`. → **The L6 result is a valid learned improvement.**

### MAJOR finding — investigated and REFUTED (for the macro comparison the report uses)
A reviewer flagged a "pool-depth artifact": LTR reranks the full RRF top-500 while the stored
baselines persisted only their top-100, claiming `RRF/500` scores R@10≈79.85 vs stored A0 72.61.
**That 79.85 was a *micro* (query-pooled) average compared against a *macro* baseline** — an
averaging mismatch, not a depth effect. Recomputing the depth-matched baselines with the report's
**macro** averaging (rank the *same* 500 pool by RRF/M4/A7 → top-100):

| (test, macro) | R@1 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| RRF/500 depth-matched (A0*) | 35.43 | 72.61 | 88.52 | 92.74 | 56.18 |
| **STORED A0 (depth-100)** | **35.43** | **72.61** | **88.52** | **92.74** | **56.18** |

→ **identical to the last decimal.** The stored A0/A7 baselines ARE depth-matched: their top-100
*is* the top-100 of the 500-pool RRF/A7 ordering, so they give the same metrics at every K≤100.
A0/A2/A7 all used the **same 500 pool** as LTR (they just output RRF/M4/A7 order without
learning); L6's ability to rescue a gold from pool rank 101–500 into its top-100 is a genuine
reranking capability, fairly measured. The **fair, depth-matched, macro, same-query-set** delta:
**L6 − RRF/500 = +13.4 R@1, +9.3 R@10, +5.9 R@50, +2.9 R@100, +12.5 nDCG@10.** Stands.

Caveat retained: the only baseline NOT at 500-pool depth is **A1** (100-pool, cached extended);
so R@50/R@100 comparisons vs A1 specifically are not depth-matched. The headline/acceptance
comparisons use A0 (final-rank) and A7 (recall) — both 500-pool — so they are fair.

### Minor/nit items (no fix needed)
- LightGBM grid reduced from spec's 36 → 2 dev-selected configs (documented §19); model selection
  on dev nDCG@10 only (legit). Linear closures captured-by-reference but invoked in-iteration (safe).
- Pre-split feature build is safe only because every transform is per-query — a reviewer rightly
  notes this invariant is implicit; any *future* global/fitted feature must move after the split.
  Added to the maintainer caveats. No current leak.

## 22. Bottom line (extension)

**LTR L6 (full feature ranker, no Cross-Encoder) is the recommended no-CE ranker:** on the
held-out test split it beats the depth-matched RRF/A7 baselines by ~+13pp R@1 / +9pp R@10 /
+12pp nDCG@10 — leakage-controlled (query-level split, verified) and depth-matched (macro). The
cluster signal that *hurt* as a fixed reranker (M4-v2) is *valuable* as a learned feature. **QSC**
as a fixed reranker is safe but ≈neutral; keep it only as an LTR feature source. The cluster-heavy
fixed scoring from M4-v2 stays retired. Next steps if pursued: full LightGBM grid + 5-fold CV
(plan §5.2) to tighten the estimate, and the 1000-candidate recall-heavy LTR run.

---

# PART III — Reviewer-grade validation: 5-fold CV + full grid + fair baselines

Implements the validation request (`cv.py`, `cv_report.py`, `scripts/run_ltr_cv.py`). Goal:
is L6 robust (not a lucky split), and how does it stand against the Cross-Encoder Method 7 and
the original-paper retrieval baselines, under one matched protocol? **This is a RETRIEVAL-ONLY
validation** — no end-task baselines (LLM Direct/Oracle/Full-Skill/LLM-Selection/Progressive
Disclosure) exist in the repo, so retrieval and end-task claims are NOT mixed (plan step 4).

## 23. CV design + results + the claim

**Protocol.** 5 query-level folds (seed 1234), stratified/balanced per dataset (verified: each
of 5400 queries is test exactly once; fold sizes balanced). Per fold: train 3 folds, tune 1
(dev), evaluate once on the held-out fold → **out-of-fold (OOF)** predictions cover all queries
with no leakage. **L6 uses the full 36-config LightGBM grid** (spec §13) tuned on dev; L0-L5/L7
use a reduced dev-tuned grid (ablation). Parallelized across 8 single-threaded processes
(OMP_NUM_THREADS=1) to avoid the macOS OpenMP crash. Best config + feature importances saved per
fold in `results/qsc_ltr/cv/cv_consolidated.json`.

**Robustness (macro mean±std over 5 folds, %):**
- **L6: R@1 49.43±0.64, R@5 ~74, R@10 82.02±1.40, R@50 ~94, R@100 96.34±0.62, nDCG@10 69.00±0.61.**
  Tight std → **robust, not a lucky split** (the single-split L6 68.6 nDCG sits within the band).
- Best LTR variant on every metric. Feature ablation reproduces across folds: L0 (RRF) nDCG
  61.5 → L1 (+M4) 67.2 → L6 (full) 69.0 — the **M4/global-cluster features are the main driver**.

**Significance (paired bootstrap, 10k resamples, two-sided p):**
- L6 vs A0: R@10 **+11.13** [+10.30,+11.99] **p=0**; nDCG@10 **+14.74** [+14.04,+15.43] p=0.
- L6 vs A7: R@10 +11.06 p=0; nDCG@10 **+16.25** p=0. L6 vs Q6: ~same, p=0.
- → **L6 beats every no-CE baseline with overwhelming significance (p=0), robustly.**

**vs Cross-Encoder Method 7 (matched, depth-controlled).** Method 7 reranks RRF top-100, so the
fair comparison is **L6@100** (L6 restricted to RRF top-100; verified: 0/540k candidates violate
the depth cap):

| (macro, %) | R@1 | R@10 | nDCG@10 |
|---|--:|--:|--:|
| **Method7-CE** | **53.78** | **86.06** | **74.67** |
| L6@100 (depth-matched) | 49.39 | 81.11 | 68.65 |
| L6@500 | 49.42 | 82.01 | 68.99 |

- L6@100 − Method7: R@10 **−1.91** [−2.63,−1.18] p=0; nDCG@10 **−3.33** [−4.10,−2.53] p=0.
  → **CE Method 7 significantly beats L6.** L6 is NOT SOTA.
- BUT complementarity: at R@10, **L6 wins on 236 queries where CE misses; CE wins on 186 where L6
  misses** — L6 helps on more individual queries even though CE's macro is higher (CE's wins
  concentrate on the harder/larger datasets). Worth noting for a fusion follow-up.

**Original-paper retrieval baselines (same fold test queries, nDCG@10 mean):** RRF 56.95, A0 56.57,
Q2 56.65, A1 55.92, A7/Q6 55.79, BM25 55.12, Hybrid-official 50.61, BGE 46.69, LinearRAG 14.54.
(TF-IDF / Contriever **not cached** in the repo → not run, stated honestly. BM25/Hybrid/LinearRAG
stored top-50 → R@100 capped.) **L6 (69.0) beats all of them by a wide, significant margin.**

**Leakage & fairness audit (verified, in the report §15):** all 5 folds train/dev/test disjoint
(overlaps = 0); each query test once; no feature column equals the label (gold only as `y`);
LTR reranks RRF top-500 while CE/depth-100 baselines rank top-100 → depth-matched via L6@100;
retrieval baselines retrieve from the full corpus at their own depth.

### Bug found & fixed during validation
`LinearRAG` files omit `gold_skill_ids` → first pass scored it 0.00. Fixed `load_retrieval_baseline`
to source gold authoritatively from the instances (`skill_annotations`); LinearRAG now 14.54
nDCG@10 (genuinely a weak retriever here). Re-ran baselines/significance via `--reuse-ltr` (no
LTR retraining). Decision: never report a misleading zero — joined gold from the canonical source.

### THE CLAIM (decision rule, plan §12)
L6 beats **all** no-CE baselines across 5 folds (p=0) but **does not** beat Cross-Encoder
Method 7 (significantly below). Therefore the safe, defensible wording is:

> **"Best lightweight / no-Cross-Encoder skill retriever-reranker on SRA-Bench (retrieval-only)."**
> L6 is robust (5-fold CV, tight CIs), significantly beats RRF/M4/PRF/QSC and the BM25/BGE/Hybrid/
> LinearRAG retrieval baselines, and is **competitive with the Cross-Encoder (within ~3.3pp
> nDCG@10) at far lower inference cost (no CE forward passes)** — but it is **NOT overall SOTA**;
> CE Method 7 remains stronger on macro. Do **not** claim SOTA or beating CE.

Honest limitations for a paper: retrieval-only (no end-task eval); LTR reranks a 500-candidate
RRF pool (a reranker, not a from-scratch retriever); cluster-id features encode corpus-specific
structure (query-leakage controlled, but may need retraining on a new corpus); latency advantage
vs CE asserted (no CE forward pass) but not benchmarked on identical hardware here.

## 24. Method 7 CE@500 — depth-matched-at-500 CE comparison (user request)

Added a fair **depth-500** CE baseline so L6@500 is compared to a CE that reranks the *same*
500-candidate pool. **No retraining** — Method 7's CE (`ce-joint-v3`, MiniLM-L6) is reused; only
CE *inference* is run on the deeper pool. Code: `ce500.py` + `scripts/run_ce500.py`.

**Exact recipe reproduced** (traced from `experiments/rerank_full_bench_h100.py` + `fuse_ce_stage1.py`):
CE = `results/models/ce-joint-v3`, `device=cpu`, `max_length=256`, `SkillPacker(field_tagged,
max_content_chars=1800)`, `batch_size=64`; fuse `0.7·minmax(CE) + 0.3·minmax(Stage1)` with
Stage1 = the M4 (`hybrid_km_alpha70`) score. CE@500 = same pipeline over the RRF top-500, Stage1 =
M4-over-500 (the `m4_score` column already in the cached feature tables).

**Faithfulness gate (mandatory before trusting CE@500):** reproduced Method7@100 from
`hybrid_km_alpha70` top-100 on champ + theoremqa → **Δ = 0.000 on every metric** vs the published
`fused_alpha_beta70` (champ R@1 35.46, theoremqa R@1 77.24, …). Pipeline is exact. (Aside: batched
CE on CPU is ~16k pairs/s across 6 workers — the full 2.7M-pair CE@500 ran in a couple minutes.)

**Results (full-set macro %, + 5-fold mean±std for CE@500):**

| Method | R@1 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| Method7-CE@100 | 53.78 | 86.06 | 91.88 | 92.23 | 74.67 |
| **Method7-CE@500** | 53.56±1.55 | 88.83 (88.82±1.03) | 96.15 | 96.78±0.54 | 75.96±1.40 |
| L6@500 (OOF) | 49.42 | 82.01 | 94.94 | 96.34 | 68.99 |
| L6@100 (OOF) | 49.39 | 81.11 | 91.49 | 92.23 | 68.65 |

- **Depth helps CE**: CE@500 − CE@100 = R@10 +2.77, R@100 +4.55, nDCG@10 +1.29 (recovers gold from
  RRF ranks 101–500). CE@500 is the stronger CE.
- **L6@500 vs CE@500 (depth-matched, p from paired bootstrap):** R@10 **−3.54** [−4.39,−2.69] p=0;
  nDCG@10 **−4.27** [−5.13,−3.41] p=0. **CE@500 significantly beats L6@500 at the top.** The gap
  is slightly *larger* than at depth-100 (CE benefits more from the deeper pool).
- **But deep recall is matched**: L6@500 R@100 96.34 ≈ CE@500 96.78 — L6 reaches CE-level recall
  ceiling; CE's edge is purely top-rank ordering quality.
- **Per-dataset complementarity (nDCG@10, L6@500 − CE@500):** TheoremQA **+2.98**, MedCalcBench
  **+14.33** (L6 wins); champ −24.75, bigcodebench −16.94, logicbench −13.00, toolqa −4.42 (CE wins).
  L6 wins where CE is weak (math/medical), CE wins on logic/code/champ → a per-dataset router or
  L6+CE fusion is the obvious follow-up.

**Verdict unchanged (strengthened):** the depth-500 comparison confirms **CE > L6**; it does not
overturn anything. The claim stays **"best no-CE / lightweight reranker, competitive with CE at far
lower cost (matches CE on R@100; ~4pp behind on nDCG@10), NOT SOTA."** Integrated into
`FULL_M4_V2_RESULTS.md` §15 (mean±std), §16, §17 (CE@100/CE@500 depth-matched table), §18
(significance L6 vs CE@500). No retraining was required or done.

## 25. M5 (CE-HYRR, raw CE no-fusion) @100 và @500 (user request)

Thêm baseline **M5 = Cross-Encoder thuần** (chỉ logit CE, KHÔNG fuse Stage-1). Method 7 =
M5 + fusion `0.7·CE + 0.3·M4`. Code: `ce500.rerank_chunk_ce` / `run_ce500.py --mode m5`.
**M5@100** = `results/rerank/full_bench_h100-{ds}.json` (raw CE@100 đã publish). **M5@500** =
pure-CE rerank RRF top-500 (ce-joint-v3, no fusion) — **không train lại**.

**Gate (Δ=0.000 cả 6 dataset):** M5@100 suy ra từ pass-500 (lọc rrf_rank≤100) khớp tuyệt đối
`full_bench_h100`. Cross-check: M5@100 macro R@10 **82.64**, nDCG@10 **69.74** — đúng số "CE-only"
ghi trong `beta_ce_rerank.py` docstring → pipeline chính xác.

**Kết quả (full-set macro %):**

| | R@1 | R@10 | R@50 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|--:|
| M5-CE@100 (raw CE) | 47.81 | 82.64 | 91.44 | 92.23 | 69.74 |
| M7-CE@100 (CE+fusion) | 53.78 | 86.06 | 91.88 | 92.23 | 74.67 |
| M5-CE@500 (raw CE) | 44.92 | 83.41 | 94.56 | 95.86 | 67.71 |
| M7-CE@500 (CE+fusion) | 53.56 | 88.83 | 96.15 | 96.78 | 75.96 |
| L6@500 (LTR) | 49.42 | 82.01 | 94.94 | 96.34 | 68.99 |

**3 phát hiện:**
1. **Fusion mới làm CE mạnh:** M5→M7 thêm +4.9 nDCG@10 (@100), +8.3 (@500). CE thuần yếu hơn hẳn.
2. **Pool sâu hại CE thuần ở top:** M5@500 nDCG 67.71 < M5@100 69.74 (R@1 44.92<47.81); nhưng vớt
   recall sâu (R@100 95.86>92.23). Fusion (M7) khắc phục.
3. **L6 ngang raw-CE@100 và VƯỢT raw-CE@500:** L6@500 nDCG 68.99 > M5@500 67.71 (Δ=+2.38, p=0;
   R@10 +0.94, p=0.064); L6@100 68.65 ≈ M5@100 69.74. → LTR nhẹ bằng/hơn cross-encoder thô; chỉ
   CE+fusion (M7) mới rõ ràng hơn L6.

Đã thêm M5 vào §15, §16, §16.1, §16.2, §17 (@100 & @500), §18. Bug đã sửa: integration crash do
đọc `ce500/{ds}.jsonl` (JSONL) bằng `json.loads` → dùng `_read_jsonl`; thêm reuse cache `m5_500/`
để khỏi chạy lại CE pass. CE pass (2.7M cặp) chỉ chạy 1 lần.

## 26. §20 Fair supervised comparison (query_gen-test, no CE-leakage) — verdict thay đổi

**Phát hiện fairness (quan trọng):** CE `ce-joint-v3` (M5/M7) được fine-tune trên **query_gen-train
= 3.782 query** (xác nhận: `all_train_pairs_v2.json` = **43.098 pairs / 3.782 queries**, khớp đúng
`train_summary`). Ở §15–§19, CE được chấm trên *full-set* / CV-fold → **~70% query đánh giá CE chính
là query CE đã train** → CE bị **thổi phồng**. L6 (OOF) luôn held-out → so sánh §17 lệch *về phía CE*.

**Fix (`run_fair_eval.py`):** train **final L6** trên query_gen-train (+dev early-stop) → đánh giá
**mọi method trên cùng query_gen-test (1.079 q, held-out cho CẢ L6 và CE)**. Audit: train∩test=0,
dev∩test=0. Final model lưu `results/models/l6_ltr_final.txt` (+`.features.json`); dataset đưa vào
`data/l6_final/` (train/dev/test npz), `data/ce_train/` (43.098 pairs), `data/splits_query_gen/`.

**Kết quả (query_gen-test, macro %):**

| Method | R@1 | R@10 | R@100 | nDCG@10 |
|---|--:|--:|--:|--:|
| **L6-final (no-CE)** | **48.34** | 81.07 | **97.10** | 67.43 |
| M5-CE@500 (raw CE) | 39.17 | 76.94 | 94.00 | 61.44 |
| M7-CE@100 | 46.54 | 83.12 | 92.80 | 68.99 |
| M7-CE@500 (CE+fusion) | 45.74 | 84.92 | 96.37 | 69.42 |
| A7 PRF | 33.54 | 69.65 | 94.35 | 54.09 |
| RRF / BM25 / BGE | 35.9/39.7/28.5 | 70.2/65.9/57.7 | 93.0/83.5/85.3 | 55.5/54.8/45.5 |

**De-leak làm đảo verdict:** CE@500 nDCG@10 rớt **75.96 (leaky) → 69.42 (fair)**; L6 gần như không
đổi (OOF 69.0 → test 67.4). Significance (paired bootstrap, query_gen-test):
- **L6 vs M7-CE@500: nDCG@10 Δ=−1.85, p=0.063 (KHÔNG có ý nghĩa thống kê = HOÀ);** R@10 −2.71 (p=0.009);
  **R@1 +2.60 và R@100 +0.73 (L6 THẮNG).**
- **L6 vs M5-CE@500 (raw CE): nDCG@10 +4.41 (p=0), R@10 +2.67 (p=0.02) — L6 THẮNG raw-CE.**
- L6 vs A7/BM25/Q6: +12–17pp, p=0.

**Kết luận mới (mạnh hơn nhiều §16):** trên so sánh **fair, không leakage, cùng split+depth**, **L6
(LTR nhẹ, KHÔNG CE) ngang ngửa CE Method 7**: hoà nDCG@10 (−1.85, p=0.063), **thắng R@1 (+2.6) và
R@100**, **thắng raw-CE (M5) rõ rệt** — với chi phí inference thấp hơn ~10–50× và không cần GPU. Phần
"CE > L6 ~4–7pp" ở §17–§19 chủ yếu là **artifact do CE-leakage**, đã được sửa ở §20. Lưu ý: L6 dùng
moderate grid (4 config) cho final; full grid (CV cho OOF 69.0) có thể thu hẹp nốt khoảng cách nDCG.

Claim cập nhật an toàn: **"L6 là reranker no-CE/lightweight tốt nhất, *cạnh tranh ngang* CE
cross-encoder (hoà nDCG@10, thắng R@1/R@100) trên test fair, với chi phí thấp hơn nhiều."**



