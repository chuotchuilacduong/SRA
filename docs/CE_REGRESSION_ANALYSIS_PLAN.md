# CE v3 Regression Analysis — Mathematical Improvement Plan

## 1. Hiện tượng quan sát (full bench, Recall@10)

| Dataset | Stage 1 (RRF+KMeans) | Stage 2 (CE v3 H100) | Δ |
|---|---:|---:|---:|
| TheoremQA | 93.31 | 87.82 | **−5.49** ❌ |
| LogicBench | 49.87 | 74.87 | +25.00 ✓ |
| ToolQA | 88.25 | 96.71 | +8.46 ✓ |
| CHAMP | 50.26 | 72.34 | +22.08 ✓ |
| MedCalcBench | 82.91 | 73.00 | **−9.91** ❌ |
| BigCodeBench | 66.10 | 91.11 | +25.01 ✓ |
| **AVG** | 71.78 | 82.64 | +10.86 ✓ |

CE rerank hurts **TheoremQA** và **MedCalcBench** dù stage 1 đã rất cao.

---

## 2. Phân tích nguyên nhân (mathematical, không bịa)

### 2.1 — Distribution của gold rank

Khi stage 1 đã đặt gold ở top-10 với xác suất cao (93%, 83%), CE chỉ có thể:
- **(a) Giữ gold ở top-10** → không có change → metric tương đương.
- **(b) Đẩy gold xuống dưới top-10** → metric giảm.
- **(c) Đẩy thêm gold (khác) lên top-10** → metric tăng (chỉ áp dụng multi-label).

TheoremQA + MedCalcBench là **single-label** datasets. Trường hợp (c) không xảy ra.
Tỷ lệ (b)/(a) chính là `Δ_loss / Δ_gain`. Nếu CE noisy → tỷ lệ (b) > 0 → metric giảm.

### 2.2 — Domain mismatch trong CE training

CE v3 train trên **43k pairs** từ 6 datasets, balanced sampling (mỗi dataset ~16-25% weight). Đối với TheoremQA + MedCalcBench, share này nhỏ hơn signal cần để model "thuộc" precision của math/medical:
- TheoremQA: 5,764 train pairs (524 queries × 11) ~ 13% total.
- MedCalcBench: 8,470 pairs ~ 20%.
- LogicBench/BigCodeBench: chiếm 14-24% nhưng có **rich syntactic patterns** (logic operators, code keywords) → CE học dễ hơn nhiều.

**Hệ quả**: CE features bias về syntactic/structural patterns → math formulas + medical terminology bị under-weighted → confused candidates trong same domain.

### 2.3 — Score calibration drift

BCE/listwise loss chỉ tối ưu ordering trong batch query. **Cross-query CE score không calibrated** — score=2.0 có ý nghĩa khác nhau cho 2 queries khác nhau.

Hậu quả: model có thể "over-confident" trên non-gold của TheoremQA (do feature overlap với train queries thắng được), và "under-confident" trên gold.

### 2.4 — Pool ceiling effect (mathematical bound)

Stage 1 RRF+KMeans R@100 = R@100 của RRF (alpha-blend chỉ reorder pool, không thêm candidates).
- TheoremQA R@100 = 98.13%, R@10 = 93.31% → gap chỉ 4.82pp gold ở ranks 11-100.
- MedCalcBench R@100 = 99.64%, R@10 = 82.73% → gap 16.91pp gold ở ranks 11-100.

CE phải "lôi" gold từ rank 11-100 lên top-10 ĐỒNG THỜI không đẩy gold đang ở top-10 xuống. Khi CE noisy:
- TheoremQA: ít gold ở ranks 11-100 (chỉ 4.82pp) → ít "lôi lên" được → mất mát "đẩy xuống" lấn át gain.
- MedCalcBench: nhiều gold ở ranks 11-100 (16.91pp) → tiềm năng tăng cao, **nhưng** CE phải đúng. Hiện không đúng → loss còn lớn hơn.

---

## 3. Mathematical improvements (KHÔNG mix, KHÔNG thay thế, KHÔNG bịa)

Mục tiêu: dùng **toán học thuần** trên output sẵn có (CE rank + Stage 1 rank) để giảm noise CE.

### 3.1 — Linear score blend (Method α-CE)

Cho mỗi candidate trong CE-reranked pool top-100:
```
ce_norm    = (CE_score - min_CE) / (max_CE - min_CE)     ∈ [0,1]
stage1_norm = (rrf_score - min_RRF) / (max_RRF - min_RRF) ∈ [0,1]
final = β · ce_norm + (1-β) · stage1_norm
```

- β = 1.0: pure CE (current setup).
- β = 0.0: pure stage 1 (no CE benefit).
- 0 < β < 1: convex combination, mathematical guarantee that final ∈ [0,1].

**Tại sao hợp lý**: stage 1 ≈ retrieval baseline (relevance dựa trên BM25/dense semantic match — robust signal cho math/medical). CE ≈ learned classifier có thể noisy. Blending = soft voting between independent estimators.

### 3.2 — Rank-RRF fusion (Method R-RRF)

Cho mỗi candidate:
```
final_score = 1/(60 + rank_CE) + 1/(60 + rank_stage1)
```

- Skill ở top của BOTH → score cao.
- Skill chỉ top của CE OR stage 1 → score trung bình.
- Skill ngoài top → score thấp.

**Tại sao hợp lý**:
- RRF (Cormack et al. 2009) là well-studied technique, không cần normalize score scale.
- Không cần hyper-parameter tuning trên dev set (k_rrf=60 robust default).
- Mathematical property: monotone & symmetric trong 2 ranks.

### 3.3 — Reciprocal-rank-weighted blend (Method RRW)

Generalize RRF nhưng có weight:
```
final = w_CE / (k + rank_CE) + w_S1 / (k + rank_stage1)
```

Với w_CE + w_S1 = 1. Reduces to RRF when w_CE = w_S1 = 1 (no normalization).

---

## 4. Plan thực nghiệm

### Bước 1: Diagnostic — tính rank distribution của gold

Cho mỗi dataset, tính:
- P(gold rank = r) ở stage 1
- P(gold rank = r) ở stage 2
- Confusion: queries có gold tăng vs giảm rank.

### Bước 2: Implement 2 fusion methods

Post-hoc, không re-run CE:
- α-CE với β ∈ {0.3, 0.5, 0.7, 0.85, 1.0}.
- R-RRF (no parameters).

### Bước 3: Evaluate

Trên FULL bench 5400 queries, compute R@1, R@10, nDCG@1, nDCG@10.

### Bước 4: Compare

Identify best β; compare với original CE + RRF baselines.

### Bước 5: Verify on each dataset

Đảm bảo improvement KHÔNG đến với cost giảm trên datasets khác.
