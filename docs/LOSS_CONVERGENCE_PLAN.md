# Plan — So sánh hội tụ 3 hàm loss cho CE HYRR

## 1. Mục tiêu

So sánh **hội tụ** và **chất lượng cuối cùng** của 3 hàm loss cho fine-tune
Cross-Encoder HYRR, từ đó chọn:
- **Hàm loss tốt nhất**
- **Số epoch tối ưu để hội tụ**
- **Learning rate tối ưu**

Đầu ra: 2 biểu đồ (train loss & dev metric) + bảng số liệu + markdown report.

---

## 2. Ba hàm loss so sánh

### (a) Pointwise — Binary Cross-Entropy (BCE)

Cho mỗi pair `(q, s, y)` độc lập:
$$
\mathcal{L}_{\text{BCE}}(q, s, y) = -[y \log \sigma(g_\theta(q,s)) + (1-y)\log(1-\sigma(g_\theta(q,s)))]
$$

- `σ` = sigmoid; `y ∈ {0, 1}`.
- Mỗi pair là 1 phân loại nhị phân độc lập.

### (b) Listwise — Softmax cross-entropy per query

Cho query có K candidates `{s_1,...,s_K}` với `n_pos` positive:
$$
\mathcal{L}_{\text{list}}(q) = -\sum_{i=1}^K \frac{y_i}{n_{\text{pos}}} \cdot \log \frac{e^{g_\theta(q, s_i)}}{\sum_j e^{g_\theta(q, s_j)}}
$$

- So sánh tương đối trong cùng 1 query group.
- Cần grouped batch (đã có `QueryGroupedBatchSampler`).

### (c) InfoNCE — Contrastive với temperature

Cho query có 1+ positives:
$$
\mathcal{L}_{\text{InfoNCE}}(q) = -\log \frac{\sum_{i: y_i=1} e^{g_\theta(q, s_i)/\tau}}{\sum_{j=1}^{K} e^{g_\theta(q, s_j)/\tau}}
$$

- `τ = 0.05` (default temperature).
- Tương tự listwise nhưng có temperature điều khiển độ "sharp" của softmax.
- Khi `τ = 1`, InfoNCE ≡ listwise (với one-hot target).

---

## 3. Setup chung

| Item | Value |
|---|---|
| Model base | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Train data | `results/train/all_train_pairs_v2.json` (43,098 pairs từ 6 datasets) |
| Dev pool | 100 random queries (rút từ `joint_dev_pool.json`, smaller for speed) |
| Batch sampler | QueryGroupedBatchSampler, 3 queries/batch ≈ 33 pairs |
| Max length | 256 |
| Optim | AdamW, weight_decay=0.01, warmup 10% |
| Device | CPU |
| Seed | 42 |

---

## 4. Phase 1 — Loss comparison (cố định lr=2e-5)

Train 3 model song song (nối tiếp) với cùng setup, khác loss:

| Run | Loss | Epochs | Output |
|---|---|---:|---|
| L1 | BCE | 3 | `results/models/ce-loss-bce/` |
| L2 | Listwise softmax | 3 | `results/models/ce-loss-listwise/` |
| L3 | InfoNCE (τ=0.05) | 3 | `results/models/ce-loss-infonce/` |

**Logging mỗi run**:
- Train loss mỗi 25 steps → `train_loss_history.json`
- Dev metrics mỗi epoch (R@1, R@10, nDCG@1, nDCG@10, MRR@10) → trong `train_summary.json`
- Wall-clock mỗi step

**Estimate**: ~42 min/epoch × 3 epochs × 3 losses = **~6.3 hours CPU**.

---

## 5. Phase 2 — LR sweep (best loss only)

Sau Phase 1, dùng loss tốt nhất, sweep LR:

| Run | LR | Epochs |
|---|---:|---:|
| LR1 | 5e-6 | 2 |
| LR2 | 2e-5 | (reuse từ Phase 1) |
| LR3 | 5e-5 | 2 |

**Estimate**: 2 runs × 2 epochs × ~42 min = **~2.8 hours**.

---

## 6. Plots cần vẽ

### Plot 1 — Train loss convergence

- X-axis: step
- Y-axis: training loss (log scale có thể)
- 3 lines: BCE / Listwise / InfoNCE
- Vertical markers ở epoch boundary

### Plot 2 — Dev metric per epoch

- X-axis: epoch
- Y-axis: nDCG@10 (%)
- 3 lines tương ứng

### Plot 3 — LR sweep (Phase 2)

- Bar plot: lr × final dev nDCG@10
- Hoặc: line plot lr → dev nDCG@10 sau epoch 2

---

## 7. Tools

- Loss: `losses.py` (BCE và listwise đã có); cần **thêm InfoNCE**.
- Trainer: `train_cross_encoder.py` (cần thêm log train_loss vào history).
- Plotting: matplotlib (cài sẵn).

---

## 8. Output files

```
results/
├── models/
│   ├── ce-loss-bce/                    # Phase 1, BCE
│   ├── ce-loss-listwise/               # Phase 1, listwise
│   ├── ce-loss-infonce/                # Phase 1, InfoNCE
│   ├── ce-lr-5e-6/                     # Phase 2, lr=5e-6 (best loss)
│   └── ce-lr-5e-5/                     # Phase 2, lr=5e-5
├── plots/
│   ├── loss_convergence.png
│   ├── dev_metric_epoch.png
│   └── lr_sweep.png
├── comparisons/
│   ├── loss_comparison.json
│   ├── lr_sweep.json
│   └── LOSS_LR_REPORT.md               # final report
```

---

## 9. Success criteria

1. **Vẽ được 3 đường convergence rõ ràng** trên 1 plot duy nhất.
2. **Xác định loss tốt nhất** theo dev nDCG@10 sau 3 epochs.
3. **Xác định epoch tối ưu**: epoch mà dev nDCG@10 plateau (Δ < 0.5pp giữa epochs liên tiếp).
4. **Xác định LR tối ưu** trong {5e-6, 2e-5, 5e-5}.
5. **Markdown report** đầy đủ: số liệu + bảng + hình + analytical notes.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Training quá lâu (>6h/run) | Giảm xuống 50% data, dùng 1 epoch để bootstrap |
| InfoNCE divergence (loss → ∞) | Clip gradient, lower lr, monitor first 100 steps |
| Plot không informative | Add smoothing (rolling mean) cho train loss; per-epoch markers |
| OOM trên CPU batch | Đã safe vì batch ~33 pairs × 256 tokens |
