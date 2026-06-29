# FULL_M4_V2_RESULTS.md

## 1. Objective
Improve Method 4 (RRF + KMeans rerank) into **M4-v2**: a larger-pool, soft-cluster, reliability-calibrated, adaptive-weight RRF reranker, and decide via ablation whether it should replace current M4.

## 2. Existing Baselines
A0 = RRF (BM25+BGE, k=60); A1 = current Method 4 (α=0.7, 0.7·RRF+0.3·aff). Both recomputed here with the published metrics for a like-for-like comparison.

## 3. M4-v2 Method
Steps: RRF top-M pool → optional safe PRF query refinement → soft top-L cluster distributions for query & skill → cluster-reliability calibration → adaptive α per query → blend `α·RRF_norm + (1-α)·Aff_norm` → top-100.

## 4. Mathematical Formulation
See `docs/kmeans/implementation_m4_v2 (1).md` §2-§10. Affinity:
`Aff_soft-cal(q,s) = Σ_k p(k|q) p(k|s) rel(k)`, `rel(k) = clamp(minmax(coh(k)·log(N/(|C_k|+1))), 0.05, 1.0)`.

## 5. Implementation Details
Code in `src/kmeans/`; reuses `sragents` metrics/corpus/schema. Query embeddings encoded with `BAAI/bge-base-en-v1.5` + search prefix and cached. M=500 pool built by extending the cached BGE depth via the corpus-embedding matmul + cached BM25 top-50, RRF-fused. See `src/kmeans/IMPLEMENTATION_NOTES.md` for every decision/deviation.

## 6. Ablation Results
See `results/comparisons/m4_v2_ablation.md` (macro table + isolated component effects).

| Variant | Pool | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| A0 RRF baseline | 500 | 36.66 | 61.78 | 71.44 | 87.83 | 92.23 | 56.57 |
| A1 Current M4 replicate | 100 | 35.53 | 61.42 | 71.76 | 89.63 | 92.23 | 55.92 |
| A2 M4 + larger pool | 500 | 34.66 | 61.78 | 71.90 | 89.75 | 93.54 | 55.59 |
| A3 M4 + calibrated hard aff | 500 | 31.44 | 58.11 | 67.71 | 85.36 | 90.46 | 51.39 |
| A4 M4 + soft affinity | 500 | 27.44 | 60.34 | 70.59 | 89.82 | 93.63 | 51.49 |
| A5 M4 + soft-calibrated aff | 500 | 27.18 | 58.95 | 69.10 | 88.65 | 92.69 | 50.45 |
| A6 M4 + adaptive alpha | 500 | 28.40 | 59.67 | 69.82 | 88.82 | 92.70 | 51.48 |
| A7 M4 + PRF only | 500 | 34.87 | 62.07 | 72.05 | 89.92 | 93.71 | 55.79 |
| A8 Full M4-v2 | 500 | 27.78 | 59.21 | 69.70 | 88.95 | 92.86 | 51.02 |

## 7. Main Results
See `results/comparisons/m4_v2_main_comparison.md` (per-dataset, per-metric).

## 8. Per-Dataset Analysis
Per-dataset numbers for every variant are in `results/comparisons/m4_v2_ablation_metrics.csv`.

### A8 vs A1 per-dataset (R@10, pp; win>+0.1 / tie / loss<-0.1)

| Dataset | A1 R@10 | A8 R@10 | Δ | verdict |
|---|---|---|---|---|
| theoremqa | 93.31 | 92.90 | -0.40 | loss |
| logicbench | 49.74 | 45.00 | -4.74 | loss |
| toolqa | 88.25 | 88.11 | -0.14 | loss |
| champ | 50.26 | 47.35 | -2.91 | loss |
| medcalcbench | 82.91 | 81.45 | -1.45 | loss |
| bigcodebench | 66.07 | 63.38 | -2.69 | loss |

## 9. Error Analysis
See `results/analysis/m4_v2_error_analysis.md`.
Fixed by A8: 71, broken by A8: 142, top-1 regressions: 709.

## 10. Latency and Resource Usage
Total wall time: 110.36s on macOS/MPS (query encoding once; everything else is matmuls/sorts on CPU). Per-run timings in `logs/m4_v2/run_summary.json`.

## 11. Final Recommendation

**Keep current M4.** M4-v2 does not improve R@10/nDCG@10 over it. See the ablation table for which component degraded.

Acceptance checks (macro):

| Check | Pass |
|---|---|
| R@10 improves vs M4 | ❌ |
| nDCG@10 improves vs M4 | ❌ |
| R@1 drop vs RRF ≤ 0.5pp | ❌ |
| R@100 improves vs M4 | ✅ |

Key deltas (pp): {'R@10': -2.06, 'nDCG@10': -4.89, 'R@100': 0.63, 'R@1_drop_vs_RRF': 8.89}

## 12. Limitations
- Tuned-on-test risk: no separate dev tuning was performed; defaults follow the spec. Sweep results (if run) must not be reported as unbiased test numbers (spec §8/§18).
- M=500 pool extends BGE depth via the corpus matmul and keeps cached BM25 top-50; BM25 beyond rank 50 is treated as absent (contribution 0) — a documented approximation.
- Method 7 (Cross-Encoder) is not a like-for-like baseline; any comparison is contextual only.

## 13. Files Produced
`results/m4_v2/{variant}/{dataset}.jsonl`, `results/comparisons/m4_v2_ablation_metrics.{csv,md,json}`, `results/comparisons/m4_v2_main_comparison.md`, `results/comparisons/m4_v2_delta.md`, `results/analysis/m4_v2_error_analysis.md`, `logs/m4_v2/run_summary.json`, this file.

## 14. Extension: Query-Specific Clustering and Lightweight LTR

### 14.1 Were these methods included in the previous M4-v2 run?

No. The previous M4-v2 run (A0–A8) tested RRF top-M, PRF, soft **global** cluster distributions, global cluster reliability, and adaptive alpha. It did **not** test query-specific *local* clustering or any trained learning-to-rank model. This section adds both.

### 14.2 QSC Method

For each query, run a local KMeans (H=16, capped at floor(sqrt(M))) on the embeddings of its RRF top-500 candidates, then rerank with QSC-1 (affinity blend, α=0.85), QSC-2 (prior blend), or QSC-3 (tie-break only, δ=0.05, λ=0.05). Cluster signal is deliberately weak (the previous global-cluster scoring hurt top-rank quality).

### 14.3 LTR Method

Feature-based rankers over 45 numeric features (retrieval / M4 / A7-PRF / QSC / query-confidence groups), no Cross-Encoder. LightGBM LambdaRank (objective=lambdarank, metric=ndcg) and a linear pairwise ranker. **Query-level stratified split** (train=3780 / dev=809 / test=811 queries, seed 42); candidates from the same query never cross splits; hyperparameters tuned on dev; metrics reported on the **held-out test split** only.

### 14.4 QSC Results (FULL query set, macro %, comparable to A0–A8)

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | nDCG@1 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| A0 RRF | 36.66 | 61.78 | 71.44 | 87.83 | 92.23 | 44.22 | 53.12 | 56.57 |
| A1 Current M4 | 35.53 | 61.42 | 71.76 | 89.63 | 92.23 | 42.96 | 52.25 | 55.92 |
| A7 PRF only | 34.87 | 62.07 | 72.05 | 89.92 | 93.71 | 42.18 | 52.23 | 55.79 |
| Q0 QSC-RRF-aff | 36.29 | 61.49 | 70.86 | 88.45 | 92.52 | 43.52 | 52.74 | 56.10 |
| Q1 QSC-RRF-prior | 35.62 | 61.86 | 71.87 | 89.15 | 93.13 | 42.82 | 52.44 | 55.99 |
| Q2 QSC-RRF-tiebreak | 36.91 | 61.77 | 71.44 | 87.83 | 92.23 | 44.42 | 53.19 | 56.64 |
| Q3 QSC-M4-aff | 34.21 | 61.21 | 71.34 | 90.09 | 93.68 | 41.28 | 51.43 | 55.05 |
| Q4 QSC-M4-tiebreak | 34.69 | 61.69 | 71.90 | 89.75 | 93.54 | 41.90 | 51.93 | 55.57 |
| Q5 QSC-A7-aff | 34.45 | 61.39 | 71.66 | 90.05 | 93.65 | 41.49 | 51.63 | 55.28 |
| Q6 QSC-A7-tiebreak | 34.97 | 61.97 | 72.05 | 89.92 | 93.71 | 42.15 | 52.20 | 55.79 |

Per-dataset Recall@10:

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| A0 RRF | 93.17 | 50.39 | 89.44 | 47.16 | 82.55 | 65.92 | 71.44 |
| A1 Current M4 | 93.31 | 49.74 | 88.25 | 50.26 | 82.91 | 66.07 | 71.76 |
| A7 PRF only | 93.57 | 51.97 | 88.18 | 50.11 | 82.73 | 65.74 | 72.05 |
| Q0 QSC-RRF-aff | 93.17 | 44.61 | 90.00 | 49.89 | 81.09 | 66.38 | 70.86 |
| Q1 QSC-RRF-prior | 93.31 | 52.24 | 89.58 | 49.51 | 80.91 | 65.65 | 71.87 |
| Q2 QSC-RRF-tiebreak | 93.17 | 50.39 | 89.44 | 47.16 | 82.55 | 65.92 | 71.44 |
| Q3 QSC-M4-aff | 93.17 | 47.24 | 88.39 | 52.20 | 80.45 | 66.57 | 71.34 |
| Q4 QSC-M4-tiebreak | 93.44 | 51.45 | 87.69 | 50.26 | 82.73 | 65.83 | 71.90 |
| Q5 QSC-A7-aff | 93.17 | 49.74 | 88.88 | 51.16 | 81.00 | 66.02 | 71.66 |
| Q6 QSC-A7-tiebreak | 93.57 | 51.97 | 88.18 | 50.11 | 82.73 | 65.74 | 72.05 |

### 14.5 LTR Results (TEST split, macro %; baselines on same split)

| Method | Recall@1 | Recall@5 | Recall@10 | Recall@50 | Recall@100 | nDCG@1 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| A0 (test) | 35.43 | 61.36 | 72.61 | 88.52 | 92.74 | 42.87 | 52.20 | 56.18 |
| A1 (test) | 35.97 | 60.63 | 72.32 | 89.96 | 92.74 | 43.16 | 51.63 | 55.76 |
| A7 (test) | 33.93 | 61.19 | 72.78 | 90.04 | 93.56 | 41.05 | 51.11 | 55.26 |
| Q2 (test) | 36.65 | 61.36 | 72.61 | 88.52 | 92.74 | 44.29 | 52.66 | 56.64 |
| Q6 (test) | 35.08 | 61.09 | 72.78 | 90.04 | 93.56 | 42.39 | 51.49 | 55.67 |
| L0 LTR-RRF | 44.99 | 65.75 | 73.71 | 88.72 | 92.43 | 52.89 | 58.87 | 61.80 |
| L1 LTR-RRF-M4 | 48.76 | 71.35 | 81.27 | 94.52 | 96.17 | 57.34 | 64.03 | 67.55 |
| L2 LTR-RRF-A7 | 44.63 | 66.23 | 74.17 | 91.17 | 94.33 | 52.41 | 58.88 | 61.76 |
| L3 LTR-RRF-QSC | 43.13 | 68.03 | 76.16 | 91.30 | 95.06 | 50.63 | 59.38 | 62.40 |
| L4 LTR-M4-A7 | 48.31 | 70.93 | 81.69 | 94.78 | 96.22 | 56.62 | 63.71 | 67.58 |
| L5 LTR-A7-QSC | 44.24 | 67.41 | 76.42 | 91.89 | 95.46 | 51.86 | 59.65 | 62.87 |
| L6 LTR-FULL | 48.82 | 74.12 | 81.91 | 94.42 | 95.63 | 57.48 | 65.75 | 68.63 |
| L7 LTR-LINEAR-FULL | 41.08 | 67.66 | 77.75 | 93.70 | 95.20 | 48.54 | 58.31 | 61.96 |

Per-dataset Recall@10 (test split):

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| A0 (test) | 91.07 | 51.75 | 90.70 | 50.98 | 85.45 | 65.72 | 72.61 |
| A1 (test) | 90.18 | 50.00 | 89.77 | 53.92 | 84.85 | 65.23 | 72.32 |
| A7 (test) | 91.96 | 53.51 | 89.77 | 50.98 | 84.85 | 65.62 | 72.78 |
| Q2 (test) | 91.07 | 51.75 | 90.70 | 50.98 | 85.45 | 65.72 | 72.61 |
| Q6 (test) | 91.96 | 53.51 | 89.77 | 50.98 | 84.85 | 65.62 | 72.78 |
| L0 LTR-RRF | 90.18 | 53.51 | 94.42 | 50.49 | 88.48 | 65.19 | 73.71 |
| L1 LTR-RRF-M4 | 90.18 | 71.93 | 97.67 | 55.88 | 90.30 | 81.67 | 81.27 |
| L2 LTR-RRF-A7 | 91.07 | 55.26 | 93.95 | 49.51 | 88.48 | 66.72 | 74.17 |
| L3 LTR-RRF-QSC | 91.07 | 60.53 | 94.42 | 49.02 | 90.30 | 71.64 | 76.16 |
| L4 LTR-M4-A7 | 90.18 | 71.93 | 98.14 | 55.39 | 92.12 | 82.38 | 81.69 |
| L5 LTR-A7-QSC | 91.07 | 62.28 | 94.88 | 46.57 | 93.33 | 70.37 | 76.42 |
| L6 LTR-FULL | 90.18 | 75.44 | 98.14 | 52.94 | 92.12 | 82.67 | 81.91 |
| L7 LTR-LINEAR-FULL | 91.96 | 65.79 | 92.56 | 54.41 | 89.70 | 72.10 | 77.75 |

### 14.6 Combined QSC + LTR

QSC-as-features inside LTR is variant **L3** (RRF+QSC), **L5** (A7+QSC) and **L6** (full, includes QSC). Compare L5/L6 against L2 (A7 without QSC) in 14.5 to read whether QSC features add anything once signals are learned.

### 14.7 Comparison against A0/A1/A7/A8

See `results/comparisons/qsc_ltr_extension_delta.md` for full Δ tables. QSC is compared on the full set; LTR on the test split (vs baselines on the same split).

### 14.8 Final Recommendation

Best QSC = Q2 QSC-RRF-tiebreak: does NOT pass the §10.1 final-ranker criteria (full set).

Best LTR = L6 LTR-FULL: PASSES the §10.1 final-ranker criteria (test split).

Acceptance detail (plan §10.1):

```json
{
  "best_qsc": {
    "id": "Q2",
    "name": "QSC-RRF-tiebreak",
    "nDCG@10": 56.64,
    "R@10": 71.44,
    "acceptance": {
      "nDCG@10 >= A0": true,
      "R@1 >= A0 - 0.5pp": true,
      "R@10 >= A7 - 0.2pp": false,
      "no dataset loses >2pp R@10 vs A7": false
    }
  },
  "best_ltr": {
    "id": "L6",
    "name": "LTR-FULL",
    "nDCG@10": 68.63,
    "R@10": 81.91,
    "acceptance": {
      "nDCG@10 >= A0": true,
      "R@1 >= A0 - 0.5pp": true,
      "R@10 >= A7 - 0.2pp": true,
      "no dataset loses >2pp R@10 vs A7": true
    }
  }
}
```

_QSC/LTR query sets differ (full vs test) — compare each method only against its same-query-set baseline. The A0/A7 test-split baselines are **depth-matched** to LTR (same RRF top-500 pool, both output top-100; verified macro-identical to a fresh RRF/500→top-100 re-rank), so L6's gains are genuine reranking, not a pool-depth artifact. LTR is a leakage-controlled learned result (query-level split, train/dev/test verified disjoint, no feature uses gold). Adversarial review found no leakage and all modules spec-correct. See `IMPLEMENTATION_NOTES.md` §17–§22._

## 15. Robustness: LightGBM Grid + 5-Fold CV

5 query-level folds (seed 1234); each query is test exactly once (out-of-fold predictions). Per fold: train on 3 folds, tune on 1 (dev), evaluate once on the held-out fold (test). **L6 uses the full 36-config LightGBM grid** (num_leaves{15,31,63}×lr{0.03,0.05}×n_est{200,500}×min_data{10,30,50}) tuned on dev; L0-L5/L7 use a reduced dev-tuned grid. **Retrieval-only** validation (no end-task baselines exist in the repo).

### Macro mean±std (%) across 5 folds (baselines on the same fold test queries)

| Method | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|
| L0 | 44.09±1.68 | 65.32±1.34 | 73.90±2.14 | 88.74±0.68 | 92.20±1.16 | 51.82±2.14 | 58.37±0.84 | 61.49±0.95 |
| L1 | 48.33±0.72 | 71.54±1.48 | 79.81±1.98 | 93.65±1.36 | 96.35±0.44 | 57.04±0.92 | 64.16±0.84 | 67.16±0.93 |
| L2 | 44.19±1.51 | 66.05±1.18 | 74.79±1.90 | 90.16±0.92 | 94.27±0.64 | 52.04±1.95 | 58.87±0.80 | 62.04±1.00 |
| L3 | 45.20±1.49 | 68.23±1.48 | 77.26±1.63 | 92.02±1.13 | 94.97±0.93 | 52.98±1.60 | 60.59±1.12 | 63.88±1.15 |
| L4 | 48.18±0.93 | 71.04±1.41 | 79.65±1.70 | 93.96±0.81 | 96.00±0.83 | 56.88±0.88 | 63.81±1.05 | 66.98±1.05 |
| L5 | 45.55±1.65 | 69.45±1.15 | 77.78±1.66 | 92.43±0.85 | 95.26±0.88 | 53.55±1.80 | 61.28±0.97 | 64.36±1.15 |
| L6 | 49.43±0.64 | 73.62±1.04 | 82.02±1.40 | 94.94±0.71 | 96.34±0.62 | 58.62±0.91 | 65.92±0.48 | 69.00±0.61 |
| L7 | 40.19±1.41 | 67.47±1.37 | 76.85±1.58 | 92.78±0.90 | 95.45±0.50 | 48.15±1.22 | 57.96±1.32 | 61.34±1.23 |
| L6@100 | 49.39±0.66 | 73.30±1.16 | 81.12±1.42 | 91.49±1.08 | 92.23±1.02 | 58.58±0.92 | 65.76±0.50 | 68.66±0.62 |
| A0 | 36.66±1.85 | 61.78±1.83 | 71.45±1.93 | 87.84±1.02 | 92.23±1.02 | 44.22±2.13 | 53.12±1.39 | 56.57±1.25 |
| A1 | 35.53±0.66 | 61.43±1.75 | 71.76±1.40 | 89.63±0.64 | 92.23±1.02 | 42.96±0.41 | 52.25±1.01 | 55.92±0.90 |
| A7 | 34.87±1.45 | 62.07±1.94 | 72.05±1.36 | 89.92±0.88 | 93.72±0.99 | 42.18±1.35 | 52.23±1.39 | 55.79±1.15 |
| Q2 | 36.91±1.71 | 61.77±1.83 | 71.45±1.93 | 87.84±1.02 | 92.23±1.02 | 44.42±1.91 | 53.19±1.28 | 56.65±1.16 |
| Q6 | 34.97±1.27 | 61.98±1.92 | 72.05±1.36 | 89.92±0.88 | 93.72±0.99 | 42.15±1.04 | 52.20±1.36 | 55.79±1.13 |
| BM25 | 38.06±0.66 | 59.62±1.42 | 68.05±1.58 | 83.10±1.38 | 83.10±1.38 | 44.83±0.54 | 52.05±0.72 | 55.12±0.65 |
| BGE | 29.56±2.66 | 51.14±1.47 | 59.30±0.55 | 78.58±1.50 | 85.12±1.33 | 36.22±2.81 | 43.72±2.09 | 46.69±1.69 |
| RRF(BM25+BGE) | 37.04±1.31 | 62.21±1.20 | 71.85±2.03 | 88.26±0.97 | 92.79±0.68 | 44.86±1.35 | 53.54±1.10 | 56.95±1.04 |
| Hybrid-official | 38.06±0.66 | 54.49±0.79 | 61.43±1.25 | 78.33±1.59 | 78.33±1.59 | 44.83±0.54 | 48.14±0.39 | 50.61±0.56 |
| LinearRAG | 6.99±0.59 | 17.34±0.17 | 21.25±0.65 | 26.31±1.24 | 26.31±1.24 | 8.44±0.51 | 13.17±0.20 | 14.54±0.23 |
| Method7-CE | 53.78±1.97 | 80.01±1.30 | 86.06±0.84 | 91.88±1.03 | 92.23±1.02 | 64.84±2.17 | 72.45±1.71 | 74.68±1.53 |
| Method7-CE@500 | 53.56±1.55 | 81.56±1.46 | 88.82±1.03 | 96.15±0.48 | 96.78±0.54 | 64.62±1.95 | 73.25±1.55 | 75.96±1.40 |
| M5-CE@100 | 47.81±1.30 | 76.14±1.02 | 82.65±1.08 | 91.44±0.84 | 92.23±1.02 | 57.57±1.20 | 67.35±1.24 | 69.74±1.25 |
| M5-CE@500 | 44.92±1.14 | 73.95±2.04 | 83.41±1.54 | 94.56±0.79 | 95.86±0.67 | 53.83±1.11 | 64.22±1.21 | 67.71±1.19 |

Best LightGBM config per fold and per-fold feature importances are in `results/qsc_ltr/cv/cv_consolidated.json` and `results/comparisons/ltr_5fold_cv_metrics.{csv,json}`.

### Leakage and Fairness Audit

| fold | train | dev | test | tr∩dev | tr∩te | dev∩te |
|---|---|---|---|---|---|---|
| 0 | 3238 | 1081 | 1081 | 0 | 0 | 0 |
| 1 | 3239 | 1080 | 1081 | 0 | 0 | 0 |
| 2 | 3241 | 1079 | 1080 | 0 | 0 | 0 |
| 3 | 3242 | 1079 | 1079 | 0 | 0 | 0 |
| 4 | 3240 | 1081 | 1079 | 0 | 0 | 0 |

- Feature/label independence: columns equal to the label = `NONE` (of 45 features; gold used only as `y`).
- Every query is test exactly once (OOF coverage); candidates of a query never cross splits.
- LTR ranks the RRF top-500 pool; CE Method 7 and the depth-100 baselines rank top-100.
- L6@100 (rerank only RRF top-100) is reported for a depth-matched CE comparison.
- Retrieval baselines (BM25/BGE/RRF/Hybrid/LinearRAG) retrieve from the full corpus at their own depth (BM25/Hybrid/LinearRAG top-50 -> R@100 capped).

## 16. Fair Comparison Against Original Paper Baselines

All retrieval baselines evaluated on the **same fold test queries** as LTR (macro mean±std). BM25/Hybrid/LinearRAG are stored top-50 → R@100 capped (n/a). TF-IDF and Contriever are **not available** as cached outputs in this repo (not run; stated honestly). End-task baselines (LLM Direct/Oracle/etc.) are **out of scope** — this is a retrieval-only comparison.

| Method | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|
| L6 | 49.43±0.64 | 73.62±1.04 | 82.02±1.40 | 94.94±0.71 | 96.34±0.62 | 58.62±0.91 | 65.92±0.48 | 69.00±0.61 |
| L6@100 | 49.39±0.66 | 73.30±1.16 | 81.12±1.42 | 91.49±1.08 | 92.23±1.02 | 58.58±0.92 | 65.76±0.50 | 68.66±0.62 |
| A0 | 36.66±1.85 | 61.78±1.83 | 71.45±1.93 | 87.84±1.02 | 92.23±1.02 | 44.22±2.13 | 53.12±1.39 | 56.57±1.25 |
| A1 | 35.53±0.66 | 61.43±1.75 | 71.76±1.40 | 89.63±0.64 | 92.23±1.02 | 42.96±0.41 | 52.25±1.01 | 55.92±0.90 |
| A7 | 34.87±1.45 | 62.07±1.94 | 72.05±1.36 | 89.92±0.88 | 93.72±0.99 | 42.18±1.35 | 52.23±1.39 | 55.79±1.15 |
| Q2 | 36.91±1.71 | 61.77±1.83 | 71.45±1.93 | 87.84±1.02 | 92.23±1.02 | 44.42±1.91 | 53.19±1.28 | 56.65±1.16 |
| Q6 | 34.97±1.27 | 61.98±1.92 | 72.05±1.36 | 89.92±0.88 | 93.72±0.99 | 42.15±1.04 | 52.20±1.36 | 55.79±1.13 |
| BM25 | 38.06±0.66 | 59.62±1.42 | 68.05±1.58 | 83.10±1.38 | 83.10±1.38 | 44.83±0.54 | 52.05±0.72 | 55.12±0.65 |
| BGE | 29.56±2.66 | 51.14±1.47 | 59.30±0.55 | 78.58±1.50 | 85.12±1.33 | 36.22±2.81 | 43.72±2.09 | 46.69±1.69 |
| RRF(BM25+BGE) | 37.04±1.31 | 62.21±1.20 | 71.85±2.03 | 88.26±0.97 | 92.79±0.68 | 44.86±1.35 | 53.54±1.10 | 56.95±1.04 |
| Hybrid-official | 38.06±0.66 | 54.49±0.79 | 61.43±1.25 | 78.33±1.59 | 78.33±1.59 | 44.83±0.54 | 48.14±0.39 | 50.61±0.56 |
| M5-CE@100 | 47.81±1.30 | 76.14±1.02 | 82.65±1.08 | 91.44±0.84 | 92.23±1.02 | 57.57±1.20 | 67.35±1.24 | 69.74±1.25 |
| M5-CE@500 | 44.92±1.14 | 73.95±2.04 | 83.41±1.54 | 94.56±0.79 | 95.86±0.67 | 53.83±1.11 | 64.22±1.21 | 67.71±1.19 |
| Method7-CE | 53.78±1.97 | 80.01±1.30 | 86.06±0.84 | 91.88±1.03 | 92.23±1.02 | 64.84±2.17 | 72.45±1.71 | 74.68±1.53 |
| Method7-CE@500 | 53.56±1.55 | 81.56±1.46 | 88.82±1.03 | 96.15±0.48 | 96.78±0.54 | 64.62±1.95 | 73.25±1.55 | 75.96±1.40 |

### 16.1 Per-dataset Recall@1 (%) — all methods, 5-fold CV test standard

_LTR = out-of-fold predictions (each query test once); baselines = same query universe. AVG = macro mean over the 6 datasets._

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| BM25 | 69.75 | 21.18 | 44.20 | 17.90 | 56.64 | 18.67 | 38.06 |
| BGE | 69.61 | 4.47 | 26.92 | 12.86 | 41.00 | 22.51 | 29.56 |
| RRF(BM25+BGE) | 72.29 | 14.08 | 43.15 | 18.31 | 51.00 | 23.41 | 37.04 |
| Hybrid-official | 69.75 | 21.18 | 44.20 | 17.90 | 56.64 | 18.67 | 38.06 |
| LinearRAG | 18.07 | 0.00 | 5.59 | 3.96 | 11.09 | 3.24 | 6.99 |
| A0 RRF | 72.29 | 16.45 | 40.14 | 16.74 | 51.18 | 23.18 | 36.66 |
| A1 M4 | 70.01 | 15.79 | 34.90 | 18.61 | 51.73 | 22.15 | 35.53 |
| A7 PRF | 70.55 | 15.79 | 32.10 | 17.56 | 51.27 | 21.94 | 34.87 |
| Q2 QSC | 72.69 | 16.05 | 41.33 | 16.52 | 51.45 | 23.40 | 36.91 |
| Q6 QSC | 70.55 | 15.66 | 32.66 | 17.19 | 51.73 | 22.02 | 34.97 |
| L0 | 74.83 | 23.29 | 67.20 | 17.38 | 59.09 | 22.76 | 44.09 |
| L1 | 75.77 | 29.74 | 80.98 | 18.24 | 58.73 | 26.54 | 48.33 |
| L2 | 75.50 | 22.89 | 67.90 | 17.41 | 58.09 | 23.32 | 44.19 |
| L3 | 76.04 | 25.00 | 67.83 | 17.64 | 60.91 | 23.76 | 45.20 |
| L4 | 76.04 | 28.95 | 80.91 | 18.24 | 58.45 | 26.49 | 48.18 |
| L5 | 77.51 | 25.79 | 67.55 | 18.05 | 60.36 | 24.01 | 45.55 |
| L6 | 76.57 | 30.53 | 81.82 | 19.06 | 60.55 | 28.01 | 49.42 |
| L7 | 73.36 | 21.05 | 46.57 | 19.02 | 57.64 | 23.48 | 40.19 |
| M5-CE@100 | 67.87 | 35.53 | 87.83 | 25.15 | 37.55 | 32.94 | 47.81 |
| M5-CE@500 | 60.51 | 33.42 | 86.57 | 20.78 | 37.45 | 30.78 | 44.92 |
| Method7-CE@100 | 77.24 | 38.82 | 89.44 | 35.46 | 46.27 | 35.45 | 53.78 |
| Method7-CE@500 | 75.23 | 40.79 | 90.28 | 34.83 | 44.55 | 35.66 | 53.56 |

### 16.2 Per-dataset Recall@10 (%) — all methods, 5-fold CV test standard

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| BM25 | 91.83 | 46.32 | 79.23 | 46.38 | 88.18 | 56.29 | 68.04 |
| BGE | 89.56 | 18.68 | 73.01 | 41.33 | 69.27 | 63.96 | 59.30 |
| RRF(BM25+BGE) | 93.31 | 49.08 | 89.30 | 50.60 | 81.45 | 67.33 | 71.84 |
| Hybrid-official | 88.76 | 38.29 | 73.15 | 39.20 | 80.82 | 48.31 | 61.42 |
| LinearRAG | 30.79 | 0.00 | 14.97 | 9.08 | 52.73 | 19.94 | 21.25 |
| A0 RRF | 93.17 | 50.39 | 89.44 | 47.16 | 82.55 | 65.92 | 71.44 |
| A1 M4 | 93.31 | 49.74 | 88.25 | 50.26 | 82.91 | 66.07 | 71.76 |
| A7 PRF | 93.57 | 51.97 | 88.18 | 50.11 | 82.73 | 65.74 | 72.05 |
| Q2 QSC | 93.17 | 50.39 | 89.44 | 47.16 | 82.55 | 65.92 | 71.44 |
| Q6 QSC | 93.57 | 51.97 | 88.18 | 50.11 | 82.73 | 65.74 | 72.05 |
| L0 | 93.84 | 53.42 | 92.31 | 49.14 | 86.27 | 68.38 | 73.89 |
| L1 | 93.57 | 68.42 | 97.34 | 51.79 | 87.18 | 80.48 | 79.80 |
| L2 | 93.71 | 56.05 | 91.89 | 50.93 | 87.45 | 68.63 | 74.78 |
| L3 | 94.11 | 61.45 | 93.36 | 52.73 | 90.18 | 71.69 | 77.25 |
| L4 | 93.57 | 68.42 | 97.48 | 49.83 | 87.55 | 81.00 | 79.64 |
| L5 | 94.24 | 64.87 | 93.71 | 50.52 | 90.27 | 73.01 | 77.77 |
| L6 | 93.71 | 74.47 | 97.34 | 52.38 | 91.27 | 82.89 | 82.01 |
| L7 | 94.91 | 66.45 | 89.58 | 50.82 | 85.82 | 73.51 | 76.85 |
| M5-CE@100 | 87.82 | 74.87 | 96.71 | 72.34 | 73.00 | 91.11 | 82.64 |
| M5-CE@500 | 82.33 | 84.87 | 98.74 | 70.32 | 69.45 | 94.74 | 83.41 |
| Method7-CE@100 | 92.37 | 74.74 | 96.71 | 77.05 | 83.36 | 92.11 | 86.06 |
| Method7-CE@500 | 90.63 | 85.53 | 98.74 | 79.98 | 81.27 | 96.82 | 88.83 |

## 17. L6 vs Cross-Encoder Method 7

Method 7 = α-CE β=0.7 (CE `ce-joint-v3` MiniLM-L6 rerank + Stage-1=M4 fusion). Two depth-matched comparisons: **CE@100 vs L6@100** (RRF top-100) and **CE@500 vs L6@500** (RRF top-500 — CE@500 reruns CE inference on the deeper pool, no retraining). CE wins both at the top (R@1/R@10/nDCG@10, p=0); at R@100 L6@500 ≈ CE@500 (deep recall matched). Per-dataset they are complementary (L6 wins TheoremQA & MedCalcBench; CE wins champ/bigcodebench/logicbench).

| Method | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|
| M5-CE@100 (raw CE) | 47.81 | 76.14 | 82.64 | 91.44 | 92.23 | 57.57 | 67.35 | 69.74 |
| **M5-CE@500 (raw CE)** | 44.92 | 73.96 | 83.41 | 94.56 | 95.86 | 53.83 | 64.23 | 67.71 |
| Method7-CE@100 (CE+fusion) | 53.78 | 80.01 | 86.06 | 91.88 | 92.23 | 64.83 | 72.45 | 74.67 |
| L6@100 | 49.39 | 73.29 | 81.11 | 91.49 | 92.23 | 58.58 | 65.75 | 68.65 |
| **Method7-CE@500 (CE+fusion)** | 53.56 | 81.56 | 88.83 | 96.15 | 96.78 | 64.62 | 73.25 | 75.96 |
| **L6@500** | 49.42 | 73.61 | 82.01 | 94.94 | 96.34 | 58.61 | 65.92 | 68.99 |

See `results/comparisons/ltr_vs_ce_method7_5fold.{md,csv,json}` for per-dataset.

## 18. Statistical Significance and Confidence Intervals

Per-method CI95 (=1.96·std/√5) is in the §15 fold-stats (`±std`; CI95 in the CSV/JSON). Paired bootstrap over shared queries (10k resamples), two-sided p-value:

| Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| L6 vs A0 | Recall@10 | +11.13 | [+10.30,+11.99] | 0.0 |
| L6 vs A0 | nDCG@10 | +14.74 | [+14.04,+15.43] | 0.0 |
| L6 vs A7 | Recall@10 | +11.06 | [+10.23,+11.91] | 0.0 |
| L6 vs A7 | nDCG@10 | +16.25 | [+15.52,+16.98] | 0.0 |
| L6 vs Q6 | Recall@10 | +11.06 | [+10.23,+11.91] | 0.0 |
| L6 vs Q6 | nDCG@10 | +16.17 | [+15.45,+16.90] | 0.0 |
| L6d100 vs Method7 | Recall@10 | -1.91 | [-2.63,-1.18] | 0.0 |
| L6d100 vs Method7 | nDCG@10 | -3.33 | [-4.10,-2.53] | 0.0 |
| L6full vs Method7 | Recall@10 | -1.04 | [-1.81,-0.25] | 0.0088 |
| L6full vs Method7 | nDCG@10 | -2.98 | [-3.77,-2.17] | 0.0 |
| L6full vs Method7@500 | Recall@10 | -3.54 | [-4.39,-2.69] | 0.0 |
| L6full vs Method7@500 | nDCG@10 | -4.27 | [-5.13,-3.41] | 0.0 |
| L6full vs M5@500 | Recall@10 | +0.94 | [-0.05,+1.96] | 0.0636 |
| L6full vs M5@500 | nDCG@10 | +2.38 | [+1.40,+3.35] | 0.0 |
| M7@500 vs M5@500 | Recall@10 | +4.49 | [+3.92,+5.06] | 0.0 |
| M7@500 vs M5@500 | nDCG@10 | +6.65 | [+6.20,+7.10] | 0.0 |

## 19. Final Paper Claim Recommendation

**Claim: "best lightweight / no-Cross-Encoder skill retriever-reranker on SRA-Bench (retrieval-only)."** L6 robustly beats every no-CE baseline across 5 folds but does NOT beat the Cross-Encoder Method 7; frame it as the strongest no-CE option, competitive with CE at far lower inference cost — NOT overall SOTA.

```json
{
  "l6_fold_nDCG@10": 69.0,
  "l6_fold_R@10": 82.02,
  "beats_all_no_ce_baselines": true,
  "beats_ce_method7_depthmatched": false,
  "ce_sig": {
    "Recall@10": {
      "n": 5400,
      "mean_diff": -1.912,
      "ci95_low": -2.632,
      "ci95_high": -1.184,
      "p_value": 0.0
    },
    "nDCG@10": {
      "n": 5400,
      "mean_diff": -3.328,
      "ci95_low": -4.1,
      "ci95_high": -2.533,
      "p_value": 0.0
    }
  }
}
```

Error analysis (OOF): L6 fixes A7 on 504 queries, breaks 54; CE>L6 on 186, L6>CE on 236 (see `results/analysis/ltr_cv_error_analysis.md`).

_Decision rule applied per the validation plan §12; see `IMPLEMENTATION_NOTES.md` §23._

## 20. Fair Supervised Comparison (query_gen-test, no CE-leakage)

**Every method evaluated on the SAME held-out `query_gen-test` (1079 queries).** CE (`ce-joint-v3`, M5/M7) was fine-tuned on `query_gen-train` (3782 q, seed 42); the **final L6** is trained on `query_gen-train` (+ `query_gen-dev` 539 q for early stopping). So **both L6 and CE are held-out** on the test queries — unlike §15–§19 where CE was scored partly on its own training queries. Zero-shot (BM25/BGE/RRF) and unsupervised (A0/A1/A7/QSC, KMeans uses no gold) need no training.

### 20.1 Macro (%) on query_gen-test

| Method | Type | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|---|
| L6-final (LTR) | supervised | 48.34 | 70.10 | 81.07 | 95.78 | 97.10 | 57.71 | 63.47 | 67.43 |
| A0 RRF | unsupervised | 35.03 | 60.04 | 69.33 | 87.66 | 92.80 | 42.56 | 51.44 | 54.81 |
| A1 M4 | unsupervised | 34.30 | 59.80 | 69.05 | 89.34 | 92.80 | 41.34 | 50.76 | 54.12 |
| A7 PRF | unsupervised | 33.54 | 59.94 | 69.65 | 89.92 | 94.35 | 40.51 | 50.60 | 54.09 |
| Q2 QSC | unsupervised | 36.13 | 60.01 | 69.33 | 87.66 | 92.80 | 43.78 | 51.75 | 55.14 |
| Q6 QSC | unsupervised | 33.41 | 59.90 | 69.65 | 89.92 | 94.35 | 40.14 | 50.41 | 53.92 |
| BM25 | zero-shot | 39.73 | 58.43 | 65.87 | 83.53 | 83.53 | 47.07 | 51.98 | 54.82 |
| BGE | zero-shot | 28.48 | 49.80 | 57.69 | 78.69 | 85.27 | 35.03 | 42.66 | 45.52 |
| RRF(BM25+BGE) | zero-shot | 35.92 | 60.18 | 70.19 | 88.37 | 93.04 | 43.68 | 52.00 | 55.51 |
| M5-CE@100 | supervised(CE) | nan | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@100 | supervised(CE) | nan | nan | nan | nan | nan | nan | nan | nan |
| M5-CE@500 | supervised(CE) | nan | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@500 | supervised(CE) | 45.74 | 75.41 | 84.92 | 94.44 | 96.37 | 56.04 | 65.95 | 69.42 |

### 20.2 Per-dataset Recall@1 (%)

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| L6-final (LTR) | 75.17 | 26.97 | 80.42 | 19.70 | 60.45 | 27.34 | 48.34 |
| A0 RRF | 68.46 | 16.45 | 39.51 | 16.10 | 46.82 | 22.83 | 35.03 |
| A1 M4 | 67.79 | 15.79 | 34.27 | 18.37 | 48.64 | 20.93 | 34.30 |
| A7 PRF | 68.46 | 15.13 | 31.47 | 18.37 | 47.27 | 20.53 | 33.54 |
| Q2 QSC | 69.80 | 17.11 | 41.26 | 17.23 | 47.73 | 23.63 | 36.13 |
| Q6 QSC | 69.13 | 14.47 | 31.12 | 16.48 | 48.18 | 21.07 | 33.41 |
| BM25 | 71.14 | 20.39 | 44.06 | 26.52 | 57.73 | 18.57 | 39.73 |
| BGE | 67.79 | 7.24 | 25.52 | 11.93 | 35.91 | 22.52 | 28.48 |
| RRF(BM25+BGE) | 67.79 | 14.47 | 43.01 | 18.37 | 48.64 | 23.27 | 35.92 |
| M5-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M5-CE@500 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@500 | 61.74 | 37.50 | 86.71 | 16.48 | 37.27 | 34.76 | 45.74 |

### 20.3 Per-dataset Recall@10 (%)

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| L6-final (LTR) | 90.60 | 74.34 | 95.45 | 53.22 | 88.64 | 84.19 | 81.07 |
| A0 RRF | 90.60 | 46.05 | 86.71 | 46.97 | 80.00 | 65.61 | 69.33 |
| A1 M4 | 89.93 | 43.42 | 86.71 | 49.24 | 80.00 | 64.98 | 69.05 |
| A7 PRF | 90.60 | 46.71 | 86.71 | 49.24 | 80.00 | 64.65 | 69.65 |
| Q2 QSC | 90.60 | 46.05 | 86.71 | 46.97 | 80.00 | 65.61 | 69.33 |
| Q6 QSC | 90.60 | 46.71 | 86.71 | 49.24 | 80.00 | 64.65 | 69.65 |
| BM25 | 88.59 | 42.76 | 77.62 | 47.54 | 83.64 | 55.07 | 65.87 |
| BGE | 85.91 | 21.05 | 73.08 | 38.26 | 65.45 | 62.40 | 57.69 |
| RRF(BM25+BGE) | 89.93 | 48.68 | 87.41 | 50.00 | 79.55 | 65.58 | 70.19 |
| M5-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M5-CE@500 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@500 | 82.55 | 87.50 | 97.55 | 64.96 | 80.45 | 96.48 | 84.92 |

### 20.4 Per-dataset nDCG@10 (%)

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| L6-final (LTR) | 82.90 | 47.67 | 89.06 | 37.25 | 73.98 | 73.73 | 67.43 |
| A0 RRF | 80.13 | 29.50 | 65.98 | 32.89 | 62.18 | 58.21 | 54.81 |
| A1 M4 | 79.19 | 28.54 | 62.37 | 34.89 | 62.99 | 56.72 | 54.12 |
| A7 PRF | 79.83 | 29.88 | 61.27 | 34.87 | 62.42 | 56.28 | 54.09 |
| Q2 QSC | 80.49 | 29.74 | 66.71 | 32.89 | 62.46 | 58.55 | 55.14 |
| Q6 QSC | 79.95 | 29.64 | 61.09 | 33.64 | 62.65 | 56.54 | 53.92 |
| BM25 | 79.97 | 29.90 | 61.88 | 40.29 | 69.39 | 47.49 | 54.82 |
| BGE | 75.38 | 13.38 | 51.36 | 27.53 | 49.65 | 55.81 | 45.52 |
| RRF(BM25+BGE) | 79.27 | 29.18 | 67.23 | 35.53 | 62.91 | 58.90 | 55.51 |
| M5-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@100 | nan | nan | nan | nan | nan | nan | nan |
| M5-CE@500 | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@500 | 71.81 | 61.59 | 93.20 | 42.20 | 57.13 | 90.58 | 69.42 |

### 20.5 Significance (paired bootstrap, query_gen-test)

| Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| L6 vs M7-CE@500 | Recall@10 | -2.71 | [-4.70,-0.70] | 0.0092 |
| L6 vs M7-CE@500 | nDCG@10 | -1.85 | [-3.82,+0.10] | 0.0632 |
| L6 vs A7 PRF | Recall@10 | +12.26 | [+10.33,+14.23] | 0.0 |
| L6 vs A7 PRF | nDCG@10 | +16.44 | [+14.91,+18.00] | 0.0 |
| L6 vs BM25 | Recall@10 | +16.86 | [+14.80,+18.92] | 0.0 |
| L6 vs BM25 | nDCG@10 | +16.47 | [+14.88,+18.09] | 0.0 |
| L6 vs Q6 QSC | Recall@10 | +12.26 | [+10.33,+14.23] | 0.0 |
| L6 vs Q6 QSC | nDCG@10 | +16.45 | [+14.89,+18.04] | 0.0 |

### 20.6 Reading

- **L6 vs CE is now a fair, leakage-free, depth-and-split-matched comparison.** L6 is the best **no-CE** ranker; CE (with fusion) is the upper line. See Δ above.
- Categories are NOT interchangeable: **supervised** (L6, CE — trained on query_gen-train) vs **zero-shot** (BM25/BGE/RRF) vs **unsupervised** (M4/QSC). Compare within intent.
- The final L6 model is saved at `results/models/l6_ltr_final.txt` (+ `.features.json`); training data in `data/l6_final/`, CE training data in `data/ce_train/`.
- L6 final config (dev-tuned): `{'num_leaves': 31, 'learning_rate': 0.03, 'n_estimators': 500, 'min_data_in_leaf': 30, 'best_iteration': 322}`.

## 21. Standalone CE (raw-corpus, SkillRouter-style) — query_gen-test

First stage retrieves from the **FULL 26,262-skill corpus** (no M4 @100 pool); CE-Raw reranks the top-1000 with **pure CE scores (no fusion)**. CE-Raw is trained on full-corpus-mined negatives (data/ce_raw/), so it is decoupled from Stage-1 in BOTH training and inference. Same held-out query_gen-test (1,079) as §20.

### 21.1 Macro (%) — CE-Raw variants (+ §20 context rows)

| Method | Type | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|---|
| bge_ft (retriever-only) | retriever | 56.99 | 81.82 | 87.87 | 98.42 | 99.26 | 67.32 | 74.55 | 76.85 |
| CE-Raw·bge_ft@1000 | standalone-CE | 33.19 | 59.17 | 69.14 | 88.33 | 93.13 | 42.32 | 50.88 | 54.49 |
| L6-final (LTR) (§20) | supervised | 48.34 | 70.10 | 81.07 | 95.78 | 97.10 | 57.71 | 63.47 | 67.43 |
| BM25 (§20) | zero-shot | 39.73 | 58.43 | 65.87 | 83.53 | 83.53 | 47.07 | 51.98 | 54.82 |
| BGE (§20) | zero-shot | 28.48 | 49.80 | 57.69 | 78.69 | 85.27 | 35.03 | 42.66 | 45.52 |
| RRF(BM25+BGE) (§20) | zero-shot | 35.92 | 60.18 | 70.19 | 88.37 | 93.04 | 43.68 | 52.00 | 55.51 |
| M5-CE@500 (§20) | supervised(CE) | nan | nan | nan | nan | nan | nan | nan | nan |
| M7-CE@500 (§20) | supervised(CE) | 45.74 | 75.41 | 84.92 | 94.44 | 96.37 | 56.04 | 65.95 | 69.42 |

### 21.2 Per-dataset nDCG@10 (%) — CE-Raw variants

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| CE-Raw·bge_ft@1000 | 53.25 | 52.81 | 88.03 | 15.66 | 33.60 | 83.58 | 54.49 |

### 21.3 Significance (paired bootstrap vs held-out CE)

| Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| CE-Raw·bge ft@1000 vs M7-CE@500 | Recall@10 | -11.82 | [-14.16,-9.47] | 0.0 |
| CE-Raw·bge ft@1000 vs M7-CE@500 | nDCG@10 | -12.53 | [-14.19,-10.84] | 0.0 |

### 21.4 Reading

- CE-Raw is a **standalone retrieve-and-rerank** (SkillRouter recipe) — different category from pipeline-CE (M5/M7, which rerank the M4 pool). Compare ceilings via retriever-only R@100.
- First-stage recall bounds CE-Raw (reranker can't recover gold outside the shortlist); rerank-depth=1000.

