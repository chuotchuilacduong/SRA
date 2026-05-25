# Hướng dẫn chạy RocketQAv2

RocketQAv2 (Ren et al., EMNLP 2021) huấn luyện **đồng thời** dual-encoder (retriever) và cross-encoder (reranker) bằng dynamic listwise distillation:

```
L = KL(p̃_DE ‖ p̃_CE)  +  CrossEntropy(s_CE, pos_idx)
```

Sau huấn luyện, hai model được dùng thay thế cho BM25 + CE-rerank thông thường.

---

## Yêu cầu

- Conda env `linearag311` đã activated
- File BM25 đã có sẵn trong `results/retrieval_test/` (dùng làm hard-negative mining)
- GPU khuyến nghị ≥ 8 GB VRAM; nếu thiếu VRAM dùng `--fp16` và `--encode-sub-batch 2`

---

## Bước 1 — Huấn luyện

### 1a. Chạy nhanh để kiểm tra (model nhỏ, 1 dataset)

```bash
sragents train-rocketqav2 \
  --de-model-path google/bert_uncased_L-2_H-128_A-2 \
  --ce-model-path cross-encoder/ms-marco-MiniLM-L-2-v2 \
  --instances data/bench/instances/champ.json \
  --bm25-results results/retrieval_test/champ-bm25-1000.json \
  --corpus data/bench/corpus/corpus_1000.json \
  --output-dir import/rocketqav2_test \
  --epochs 1 --batch-size 4 --n-hard-neg 7 \
  --encode-sub-batch 2 --fp16 --log-every 5
```

Kết quả lưu vào:
```
import/rocketqav2_test/
  dual_encoder/     ← model retriever đã fine-tune
  cross_encoder/    ← model reranker đã fine-tune
  train_meta.json   ← siêu tham số đã dùng
```

### 1b. Huấn luyện đầy đủ (model thật, 1 dataset)

```bash
sragents train-rocketqav2 \
  --de-model-path BAAI/bge-base-en-v1.5 \
  --ce-model-path BAAI/bge-reranker-base \
  --instances data/bench/instances/champ.json \
  --bm25-results results/retrieval_test/champ-bm25-1000.json \
  --corpus data/bench/corpus/corpus_1000.json \
  --output-dir import/rocketqav2_champ \
  --epochs 3 --batch-size 8 --n-hard-neg 15 \
  --encode-sub-batch 4 --fp16 --log-every 50
```

### 1c. Huấn luyện trên tất cả 6 dataset (loop)

```bash
DATASETS=(bigcodebench champ logicbench medcalcbench theoremqa toolqa)

for DS in "${DATASETS[@]}"; do
  echo "=== Training on $DS ==="
  sragents train-rocketqav2 \
    --de-model-path BAAI/bge-base-en-v1.5 \
    --ce-model-path BAAI/bge-reranker-base \
    --instances data/bench/instances/${DS}.json \
    --bm25-results results/retrieval_test/${DS}-bm25-1000.json \
    --corpus data/bench/corpus/corpus_1000.json \
    --output-dir import/rocketqav2_${DS} \
    --epochs 3 --batch-size 8 --n-hard-neg 15 \
    --encode-sub-batch 4 --fp16
done
```

---

## Bước 2 — Retrieval với Dual-Encoder đã huấn luyện

Chạy dense retrieval dùng model DE vừa train (ví dụ cho `champ`):

```bash
sragents retrieve \
  --retriever bge \
  --retriever-arg model_path=import/rocketqav2_champ/dual_encoder \
  --corpus data/bench/corpus/corpus_1000.json \
  --instances data/bench/instances/champ.json \
  --output results/retrieval_test/champ-rqv2_de-1000.json \
  --top-k 50
```

> Lệnh này tự in Recall@K và nDCG@K ngay sau khi chạy xong.

Loop tất cả dataset:

```bash
DATASETS=(bigcodebench champ logicbench medcalcbench theoremqa toolqa)

for DS in "${DATASETS[@]}"; do
  sragents retrieve \
    --retriever bge \
    --retriever-arg model_path=import/rocketqav2_${DS}/dual_encoder \
    --corpus data/bench/corpus/corpus_1000.json \
    --instances data/bench/instances/${DS}.json \
    --output results/retrieval_test/${DS}-rqv2_de-1000.json \
    --top-k 50
done
```

---

## Bước 3 — Reranking với Cross-Encoder đã huấn luyện

```bash
sragents cross-encoder-rerank \
  --input results/retrieval_test/champ-rqv2_de-1000.json \
  --model-path import/rocketqav2_champ/cross_encoder \
  --instances data/bench/instances/champ.json \
  --corpus data/bench/corpus/corpus_1000.json \
  --output results/retrieval_test/champ-rqv2_rerank-1000.json \
  --top-k 50
```

Loop tất cả dataset:

```bash
DATASETS=(bigcodebench champ logicbench medcalcbench theoremqa toolqa)

for DS in "${DATASETS[@]}"; do
  sragents cross-encoder-rerank \
    --input results/retrieval_test/${DS}-rqv2_de-1000.json \
    --model-path import/rocketqav2_${DS}/cross_encoder \
    --instances data/bench/instances/${DS}.json \
    --corpus data/bench/corpus/corpus_1000.json \
    --output results/retrieval_test/${DS}-rqv2_rerank-1000.json \
    --top-k 50
done
```

---

## Bước 4 — So sánh retrieval metrics

Sau khi có file kết quả, dùng script có sẵn:

```bash
# So sánh tất cả method trong results/retrieval_test/
python compare_metrics.py

# Xem bảng đầy đủ (theo dataset + trung bình)
python show_results.py
```

Các file `*-rqv2_de-1000.json` và `*-rqv2_rerank-1000.json` sẽ tự xuất hiện trong bảng so sánh.

---

## Bước 5 (Tuỳ chọn) — Inference + Task Evaluation

Nếu muốn đánh giá accuracy trên benchmark (không chỉ retrieval metrics):

### 5a. Inference

```bash
sragents infer \
  --instances data/bench/instances/champ.json \
  --output results/infer_test/champ-rqv2_rerank.jsonl \
  --model <tên-model-LLM> \
  --provider topk \
  --provider-arg source=results/retrieval_test/champ-rqv2_rerank-1000.json \
  --provider-arg k=5 \
  --engine direct \
  --api-base $OPENAI_API_BASE
```

### 5b. Evaluate

```bash
sragents evaluate \
  --input results/infer_test/champ-rqv2_rerank.jsonl \
  --instances data/bench/instances/champ.json \
  --output results/eval_test/champ-rqv2_rerank.json
```

---

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `CUDA out of memory` | Thêm `--encode-sub-batch 2 --fp16 --batch-size 4` |
| `ValueError: Couldn't instantiate backend tokenizer` | Đổi model sang `google/bert_uncased_L-2_H-128_A-2` (test) hoặc `BAAI/bge-base-en-v1.5` (full) |
| `KeyError: instance_id` trong BM25 lookup | Kiểm tra `--bm25-results` đúng dataset (champ BM25 cho champ instances) |
| Model download chậm | `export HF_TOKEN=<token>` trước khi chạy |

---

## Tóm tắt pipeline

```
BM25 results (hard-neg mining)
        │
        ▼
sragents train-rocketqav2   →  dual_encoder/  +  cross_encoder/
                                    │                   │
                                    ▼                   │
                            sragents retrieve           │
                            (--retriever bge)           │
                                    │                   │
                                    ▼                   ▼
                            rqv2_de-1000.json  →  sragents cross-encoder-rerank
                                                         │
                                                         ▼
                                                  rqv2_rerank-1000.json
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                    python compare_metrics.py   sragents infer
                                    python show_results.py           │
                                    (retrieval metrics)              ▼
                                                             sragents evaluate
                                                             (task accuracy)
```
