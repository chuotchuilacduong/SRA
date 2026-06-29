# CE-Raw / SkillRouter — Quy trình chi tiết (training → evaluation)

Tài liệu này ghi lại **toàn bộ quy trình** đã thực hiện để tạo ra kết quả **§21
(Standalone CE, raw-corpus, SkillRouter-style)** trong `FULL_M4_V2_RESULTS.md`:
từ dựng dữ liệu → train cross-encoder → fine-tune retriever → chạy evaluation trên
held-out `query_gen-test` (1,079 query). Mọi bước đều tái lập được bằng các lệnh ở §11.

> **Phạm vi thí nghiệm.** CE-Raw là một pipeline **retrieve-and-rerank độc lập**, *tách
> hoàn toàn* khỏi M4 Stage-1 (@100 pool) ở **cả training lẫn inference** — đối lập với M5/M7
> (`ce-joint-v3`) vốn rerank pool @100/@500 của M4. Mục tiêu: đo **chi phí của việc bỏ tầng-1
> M4 mạnh** và xem một CE kiểu SkillRouter có cạnh tranh được trên kho 26K skill hay không.

---

## 0. Bức tranh tổng thể

```
Phase 1  (CPU)   ce_raw dataset  ── đào hard-negatives từ FULL corpus (26,262 skill)
                                    + SkillRouter false-negative filter
   │
   ├── Phase 2 (GPU/MPS)  CE-Raw      = cross-encoder MiniLM-L6, listwise, full-text
   │                                    → results/models/ce-raw-v1
   │
   └── Phase B (GPU/MPS)  retriever   = bi-encoder BGE fine-tune, InfoNCE + hard-neg
                                        → results/models/sr-emb-bge-v1   (= "bge_ft")
   │
Phase 3+4 (GPU/MPS)  evaluation       first-stage (bge_base | rrf | bge_ft) lấy top-D từ
                                       FULL corpus → CE-Raw rerank top-1000 (no fusion)
                                       → eval trên query_gen-test → ghi §21
```

Hai thành phần **được train** (CE-Raw và retriever `bge_ft`) đều học trên `query_gen-train`
(3,782 query) và đánh giá trên `query_gen-test` (1,079 query) → **fair, không rò rỉ (leakage-free)**,
cùng tập test với §20.

---

## 1. So sánh với CE hiện có (M5/M7 = `ce-joint-v3`)

| | M5 / M7 (đã có) | **CE-Raw (tài liệu này)** |
|---|---|---|
| Negatives khi train | đào từ M4 **stage-1 `extended@100`** | đào từ **FULL 26,262-skill corpus** (bm25/bge/cluster/random) + false-neg filter |
| Candidates khi infer | rerank pool **M4 @100/@500** | retrieve top-D từ **full corpus** (BGE / RRF / BGE fine-tune) |
| Fusion tầng-1 | M7 = `0.7·CE + 0.3·M4`; M5 = raw CE | **không** (pure CE) |
| Retriever | M4/RRF (cố định) | base BGE, RRF, **hoặc bi-encoder fine-tune (Phase B)** |

Ý tưởng kiến trúc tham khảo từ **SkillRouter**: (1) bi-encoder InfoNCE + hard-negative mining
(semantic / lexical / taxonomy / random) + false-negative filtering; (2) lấy top-K; (3)
cross-encoder listwise rerank trên **full skill text**.

---

## 2. Dữ liệu đầu vào (prerequisites, phải có sẵn)

```
data/bench/corpus/corpus.json                 # 26,262 skill (name/description/content)
data/bench/instances/*.json                   # query + gold_skill_ids theo dataset
results/splits/{ds}-query_gen.json            # train/dev/test theo từng dataset
results/retrieval_bm25/{ds}-bm25.json         # BM25 top-k đã cache
results/retrieval_dense/{ds}-dense-bge.json   # BGE dense top-k đã cache
results/bge/corpus_emb.npy + corpus_ids.json  # embedding toàn corpus (BGE base), để retrieve full
results/clusters.json                         # skill_id → cluster (KMeans K=300), cho cluster-hard neg
results/m4_v2/cache/query_emb/{ds}.npy + {ds}_ids.json   # query embedding đã cache
# (tùy chọn, cho §21.3 significance + §20 context):
results/qsc_ltr/ce500/{ds}.jsonl              # M7-CE@500 records
results/qsc_ltr/m5_500/{ds}.jsonl             # M5-CE@500 records
results/comparisons/fair_supervised_query_gen.json   # §20 (mọi method, per-dataset)
```

6 dataset: `theoremqa, logicbench, toolqa, champ, medcalcbench, bigcodebench`.
`query_gen` split: **train 3,782 / dev 539 / test 1,079 = 5,400**.

---

## 3. Môi trường chạy

Python deps: `torch`, `transformers`, `sentence-transformers>=2.2`, **`datasets`**
(bắt buộc cho `.fit()` của ST ở Phase B — `pip install datasets`), `numpy`, `scipy`, `pyyaml`.
Base checkpoint tải từ HuggingFace lần đầu: `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`BAAI/bge-base-en-v1.5` (giữ HF **online** cho lần chạy đầu).

**Chạy local trên Mac (Apple-GPU / MPS).** Trainer/reranker/eval mặc định CUDA-hoặc-CPU
(tắt MPS vì lịch sử không ổn định). Đặt **`SRA_ALLOW_MPS=1`** để bật MPS — đây là khác biệt
*duy nhất* so với lệnh trên H100, và là no-op trên máy CUDA (CUDA luôn thắng).

```bash
export PYTHONPATH=src SRA_ALLOW_MPS=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
```

Thời gian quan sát trên một máy M-series (~64 GB unified): Phase 2 ≈ 25 phút, Phase B ≈ 18 phút,
deep eval (3 retriever @1000) ≈ 2 giờ. Trên CUDA (H100) nhanh hơn nhiều và dùng được batch lớn.
`fp16` chỉ bật với CUDA; MPS chạy `fp32`.

> Lưu ý: trainer (`train_cross_encoder.py`) và các script eval/retriever đều có cổng
> `SRA_ALLOW_MPS`. Reranker khi eval nằm ở `src/sragents/retrieve/cross_rerank.py` —
> file này **bị gitignore** trong repo, nên cổng MPS của nó là patch local; H100/CUDA không
> ảnh hưởng (CUDA không bao giờ vào nhánh MPS).

---

## 4. Phase 1 — Dựng dataset raw (CPU, ~2–5 phút)

**Script:** `src/kmeans/scripts/ceraw_build_dataset.py`

```bash
python src/kmeans/scripts/ceraw_build_dataset.py    # mix 4 bm25 / 3 bge / 2 cluster / 1 random
```

**Logic đào negative (cho mỗi query trong `query_gen-train`):**

1. **Positive** = các `gold_skill_ids` của query.
2. **Hard negatives** lấy từ **FULL corpus** theo 4 nguồn, tỉ lệ **4 : 3 : 2 : 1**:
   - `bm25_hard`  — top BM25 (lexical) không phải gold;
   - `bge_hard`   — top BGE dense (semantic) không phải gold;
   - `cluster_hard` — skill cùng cluster KMeans với gold (taxonomy-style), không phải gold;
   - `random`     — skill ngẫu nhiên trong corpus.
3. **False-negative filtering** (lọc các "negative" thực ra gần như là gold, kiểu SkillRouter):
   loại bỏ negative nếu trùng tên với gold (name-dedup), hoặc trùng nội dung
   (**trigram Jaccard > 0.6**), hoặc gần về embedding (**cosine > 0.92**) với bất kỳ gold nào.

**Outputs → `data/ce_raw/`:**

| File | Nội dung |
|---|---|
| `ce_pairs_train.json` | cặp `(question, skill, label)` cho CE — train |
| `dev_pool.json` | full-corpus **BGE@100** trên `query_gen-dev` (để chọn checkpoint theo dev-nDCG@10) |
| `instances_all.json` | instances để map `instance_id → query` |
| `de_triples_train.jsonl` | `(query, positive_id, negative_ids)` cho Phase B retriever |
| `stats.json` | thống kê |

**Thống kê thực tế (`stats.json`):**

```
train_queries = 3,782 | dev_queries = 539 | pairs = 43,098
source_counts: positive 5,278 | bm25_hard 15,128 | bge_hard 11,346 | cluster_hard 7,564 | random 3,782
false_negatives_removed = 296 | mix = 4:3:2:1 | filter = {jaccard 0.6, cosine 0.92}
```

---

## 5. Phase 2 — Train CE-Raw (cross-encoder) — GPU/MPS, ~25 phút

**Script:** `src/kmeans/scripts/ceraw_train_ce.py` (driver mỏng gọi
`sragents.train.train_cross_encoder` — đúng code đã tạo `ce-joint-v3`, nên **chỉ dữ liệu khác**,
recipe giống hệt). **Config:** `src/kmeans/configs/ceraw_listwise.yaml`.

```bash
python src/kmeans/scripts/ceraw_train_ce.py \
    --config src/kmeans/configs/ceraw_listwise.yaml \
    --out results/models/ce-raw-v1
```

**Recipe (giống ce-joint-v3):**

```yaml
model:   { base: cross-encoder/ms-marco-MiniLM-L-6-v2, max_length: 256 }
trainer: { batch_size: 33, lr: 2.0e-5, epochs: 3, weight_decay: 0.01, warmup_ratio: 0.1,
           seed: 42, group_by_query: true, queries_per_batch: 3, balanced_by_dataset: true,
           early_stop_metric: dev_ndcg@10, early_stop_patience: 99, fp16_if_cuda: true }
loss: listwise
packing: { mode: field_tagged, include_tools: false, max_content_chars: 1800 }
```

- **Listwise loss**: mỗi batch chứa *toàn bộ* pairs của `queries_per_batch=3` query (positive + hard
  negatives cùng query trong một forward) → softmax listwise.
- **Full-text packing** (`field_tagged`, max 1,800 ký tự nội dung) — toàn văn skill quan trọng (SkillRouter).
- Chọn **checkpoint có dev-nDCG@10 cao nhất** trên `dev_pool.json`. Lần chạy ghi nhận:
  **best epoch = 2, dev-nDCG@10 ≈ 0.6216** (loss train 1.48 → 0.85 → 0.65).
- Device: CUDA (H100) > MPS (nếu `SRA_ALLOW_MPS=1`) > CPU; MPS chạy fp32.

**Output:** `results/models/ce-raw-v1/` (`model.safetensors`, tokenizer, `train_config.json`,
`train_summary.json`).

---

## 6. Phase B — Fine-tune retriever (full SkillRouter) — GPU/MPS, ~18 phút

**Script:** `src/kmeans/scripts/ceraw_train_retriever.py`. Fine-tune bi-encoder
`BAAI/bge-base-en-v1.5` bằng **MultipleNegativesRankingLoss** (InfoNCE in-batch + hard negatives
đã đào ở Phase 1, đọc từ `de_triples_train.jsonl`).

```bash
python src/kmeans/scripts/ceraw_train_retriever.py \
    --epochs 2 --batch-size 16 --max-negs 4 --max-seq-length 256 \
    --out results/models/sr-emb-bge-v1
```

- Mỗi anchor = `[query, positive, neg_1..neg_k]`; loss kéo query↔positive lại gần, đẩy ra
  negatives + các sample khác trong batch.
- **`--max-seq-length 256` là bắt buộc trên MPS**: bge-base ở seq mặc định 512 làm
  **OOM** bộ nhớ unified (~63 GB ở step 0). Cap seq=256 + batch=16 + max-negs=4 chạy ổn định.
  Trên CUDA có thể dùng batch 64, seq mặc định.
- 3,782 anchors, 2 epoch. Output `results/models/sr-emb-bge-v1/` — đây chính là retriever
  **`bge_ft`** dùng ở eval (`--retrievers bge_ft`). `max_seq_length=256` được lưu trong
  `sentence_bert_config.json`.

---

## 7. Phase 3+4 — Evaluation + ghi §21 — GPU/MPS, ~2 giờ

**Script:** `src/kmeans/scripts/ceraw_eval.py`

```bash
python src/kmeans/scripts/ceraw_eval.py \
    --retrievers bge_base,rrf,bge_ft \
    --rerank-depth 1000 \
    --ce-model results/models/ce-raw-v1
```

**First stage — lấy candidate từ FULL corpus (không cắt @100):**

- `bge_base` — dùng query-embedding đã cache `@` `corpus_emb` (BGE gốc), lấy top-D toàn corpus.
- `rrf` — **RRF(k=60)** hợp nhất BM25 (lexical) + BGE (dense) trên toàn corpus.
- `bge_ft` — encode lại toàn corpus bằng retriever fine-tune (cache `sr-emb-bge-v1/corpus_emb.npy`),
  rồi retrieve top-D.

**Second stage — CE-Raw rerank:** với mỗi query, lấy **top-1000** của first stage, cho
`results/models/ce-raw-v1` chấm điểm thuần CE (**no fusion**), sắp xếp lại, cắt `output_top_k`.

**Eval:** `evaluate.eval_variant` trên đúng **`query_gen-test` (1,079)**; in thêm dòng
`bge_base|rrf|bge_ft (retriever-only)` (chỉ first stage, chưa rerank) làm **trần** để thấy
giới hạn recall. Significance: **paired bootstrap** so CE-Raw·rrf vs M7/M5-CE@500 (held-out).

**Ghi báo cáo:** append **§21** vào `FULL_M4_V2_RESULTS.md` + lưu
`results/comparisons/ceraw_query_gen.{json,md}`.

> Lưu ý vận hành: `ceraw_eval.py` cắt báo cáo từ `## 21.` trở đi khi ghi lại — nên **§21 phải
> được ghi trước §22**. (Để chạy nhanh kiểm tra wiring, dùng `--retrievers bge_base
> --rerank-depth 50` ~4 phút.)

---

## 8. (Tùy chọn) Phase 5b — §22 so sánh hợp nhất với MỌI method

**Script:** `src/kmeans/scripts/unified_compare.py` — gộp
`fair_supervised_query_gen.json` (§20) + `ceraw_query_gen.json` (§21) thành một bảng công bằng,
CE-Raw vs mọi method **trên cùng `query_gen-test`**, kèm 4 bảng per-dataset (Recall@1, Recall@10,
nDCG@1, nDCG@10). Không re-eval, không GPU.

```bash
python src/kmeans/scripts/unified_compare.py    # append §22
```

---

## 9. Kết quả §21 (query_gen-test, depth 1000)

### 21.1 Macro (%) — CE-Raw variants (+ context)

| Method | Type | R@1 | R@5 | R@10 | R@50 | R@100 | nD@1 | nD@5 | nD@10 |
|---|---|---|---|---|---|---|---|---|---|
| bge_base (retriever-only) | retriever | 28.38 | 49.54 | 57.48 | 77.77 | 84.63 | 34.58 | 42.17 | 45.12 |
| CE-Raw·bge_base@1000 | standalone-CE | 34.10 | 59.82 | 71.04 | 87.41 | 92.00 | 43.49 | 51.93 | 55.85 |
| rrf (retriever-only) | retriever | 35.83 | 59.96 | 69.33 | 87.64 | 92.80 | 43.26 | 51.67 | 55.08 |
| CE-Raw·rrf@1000 | standalone-CE | 34.79 | 61.32 | 72.85 | 89.46 | 94.16 | 44.18 | 53.03 | **57.07** |
| bge_ft (retriever-only) | retriever | 56.99 | 81.82 | 87.87 | 98.42 | **99.26** | 67.32 | 74.55 | **76.85** |
| CE-Raw·bge_ft@1000 | standalone-CE | 33.19 | 59.17 | 69.14 | 88.33 | 93.13 | 42.32 | 50.88 | 54.49 |

### 21.2 Per-dataset nDCG@10 (%)

| Method | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | AVG |
|---|---|---|---|---|---|---|---|
| CE-Raw·bge_base@1000 | 57.00 | 47.13 | 88.17 | 18.11 | 37.69 | 87.03 | 55.85 |
| CE-Raw·rrf@1000 | 57.19 | 53.81 | 89.22 | 17.27 | 37.83 | 87.09 | 57.07 |
| CE-Raw·bge_ft@1000 | 53.25 | 52.81 | 88.03 | 15.66 | 33.60 | 83.58 | 54.49 |

### 21.3 Significance (paired bootstrap vs held-out CE)

| Comparison | Metric | Δ (pp) | 95% CI | p |
|---|---|---|---|---|
| CE-Raw·rrf@1000 vs M7-CE@500 | Recall@10 | −8.28 | [−10.13, −6.43] | 0.0 |
| CE-Raw·rrf@1000 vs M7-CE@500 | nDCG@10 | −9.86 | [−11.28, −8.47] | 0.0 |

> Các con số ở §9 là từ bản ghi hiện tại trong `FULL_M4_V2_RESULTS.md` §21. Hai thành phần
> *được train* (CE-Raw, `bge_ft`) có dao động nhỏ giữa các lần chạy/máy (seed, MPS vs CUDA);
> các dòng `retriever-only` cho `bge_base`/`rrf` là tất định (đọc từ embedding đã cache).

---

## 10. Phân tích / nhận xét chính

1. **CE-Raw (standalone) kém pipeline-CE (M5/M7).** CE-Raw·rrf (tốt nhất trong nhóm CE-Raw)
   thua **M7-CE@500 ~ −9.86 pp nDCG@10**, p≈0. Đây chính là **chi phí của việc bỏ tầng-1 M4 mạnh**
   — câu hỏi cốt lõi của thí nghiệm.
2. **First-stage recall chặn trên CE-Raw.** Reranker không thể cứu gold nằm ngoài shortlist:
   R@100 của `bge_base`≈84.6, `rrf`≈92.8, `bge_ft`≈99.3. Vì vậy `CE-Raw·rrf` công bằng hơn
   `CE-Raw·bge_base`.
3. **Retriever fine-tune (Phase B) là phần thắng lớn nhất.** `bge_ft (retriever-only)` đạt
   **R@100 ≈ 99.3%** và **nDCG@10 = 76.85** — *cao nhất toàn bộ so sánh* (hơn cả M7-CE@500 và
   L6-final), dù chỉ là retriever một tầng. Đây là "encoder stage" kiểu SkillRouter.
4. **Nhưng CE-Raw rerank LÀM XẤU `bge_ft`:** nDCG@10 **76.85 → 54.49**. Lý do: CE-Raw được train
   trên negatives đào từ bm25/bge/cluster, nên khi rerank trong pool *khó/khác phân phối* của
   `bge_ft` thì xếp hạng sai, đẩy gold tốt xuống. ⇒ Muốn CE có ích cho `bge_ft`, phải **đào lại
   hard-negatives từ chính `bge_ft`** (chạy lại Phase 1 nguồn từ `sr-emb-bge-v1`) rồi train lại CE.
5. **Per-dataset:** CE-Raw mạnh ở `toolqa` (~88–89) và `bigcodebench` (~84–87), yếu ở `champ`
   (~16–18; chỉ 44 test query) và `medcalcbench` (~34–38) — đúng các dataset mà recall họ BGE thấp.

---

## 11. Lệnh tái lập đầy đủ (end-to-end)

```bash
# 0) môi trường (local Mac/MPS; trên H100 bỏ SRA_ALLOW_MPS và dùng batch lớn hơn)
cd <repo>
export PYTHONPATH=src SRA_ALLOW_MPS=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
pip install datasets            # cần cho Phase B (.fit của sentence-transformers)

# 1) dựng dataset raw (CPU)
python src/kmeans/scripts/ceraw_build_dataset.py

# 2) train CE-Raw (cross-encoder)
python src/kmeans/scripts/ceraw_train_ce.py \
    --config src/kmeans/configs/ceraw_listwise.yaml --out results/models/ce-raw-v1

# B) fine-tune retriever (bge_ft)  — cờ memory-safe cho MPS
python src/kmeans/scripts/ceraw_train_retriever.py \
    --epochs 2 --batch-size 16 --max-negs 4 --max-seq-length 256 \
    --out results/models/sr-emb-bge-v1

# 3+4) evaluation + ghi §21
python src/kmeans/scripts/ceraw_eval.py \
    --retrievers bge_base,rrf,bge_ft --rerank-depth 1000 \
    --ce-model results/models/ce-raw-v1

# 5b) (tùy chọn) §22 so sánh hợp nhất với mọi method
python src/kmeans/scripts/unified_compare.py
```

**Trên H100 (CUDA):** bỏ `SRA_ALLOW_MPS`/`PYTORCH_ENABLE_MPS_FALLBACK`; Phase B dùng
`--batch-size 64` và bỏ `--max-seq-length` (mặc định 512). Mọi script còn lại giữ nguyên.

---

## 12. Bản đồ file / artifacts

| File | Vai trò | Device |
|---|---|---|
| `src/kmeans/scripts/ceraw_build_dataset.py` | Phase 1 — dựng dataset raw | CPU |
| `src/kmeans/configs/ceraw_listwise.yaml` | Phase 2 — config CE | — |
| `src/kmeans/scripts/ceraw_train_ce.py` | Phase 2 — train CE-Raw | GPU/MPS |
| `src/kmeans/scripts/ceraw_train_retriever.py` | Phase B — fine-tune retriever | GPU/MPS |
| `src/kmeans/scripts/ceraw_eval.py` | Phase 3+4 — eval + §21 | GPU/MPS |
| `src/kmeans/scripts/unified_compare.py` | Phase 5b — gộp §20+§21 → §22 | CPU |
| `src/kmeans/CE_RAW_SKILLROUTER_RUNBOOK.md` | runbook (H100 + local) | — |
| `data/ce_raw/` | dataset Phase 1 (pairs, dev_pool, triples, stats) | — |
| `results/models/ce-raw-v1/` | CE-Raw checkpoint | — |
| `results/models/sr-emb-bge-v1/` | retriever fine-tune (`bge_ft`) | — |
| `results/comparisons/ceraw_query_gen.{json,md}` | bảng §21 + significance | — |

(`data/` và `results/` bị gitignore — chỉ code + báo cáo được track; mọi artifact tái tạo bằng §11.)

---

## 13. Hạn chế & hướng cải tiến

- **CE lệch phân phối với `bge_ft`** (xem §10.4): cần một vòng *hard-negative mining từ chính
  retriever đã fine-tune* rồi train lại CE (self-distillation kiểu SkillRouter) — kỳ vọng nâng
  `CE-Raw·bge_ft` lên trên cả retriever-only.
- **Depth = 1000** là lựa chọn "không cắt @100" thực dụng; `--rerank-depth full` (rerank cả corpus,
  ~28M cặp CE) chỉ để đo trần, rất nặng.
- CE-Raw vẫn là *category khác* với M5/M7 (standalone vs rerank-pool-M4) — khi so sánh phải đối
  chiếu trần qua `retriever-only R@100`.
