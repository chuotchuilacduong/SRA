# SR-Agents + LinearRAG

Benchmark **Skill-Retrieval Augmented Agents** trên [SRA-Bench](https://huggingface.co/datasets/WeihangSu/SRA-Bench), tích hợp thêm **LinearRAG** (graph-based retrieval) làm retriever mới bên cạnh các baseline BM25, TF-IDF, BGE, Contriever.

```
26,262 skills  ×  6 datasets  ×  5,400 queries
```

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Chuẩn bị dữ liệu](#3-chuẩn-bị-dữ-liệu)
4. [Cấu hình LLM backend](#4-cấu-hình-llm-backend)
5. [Chạy pipeline (test nhanh — RAM thấp)](#5-chạy-pipeline-test-nhanh--ram-thấp)
6. [Chạy pipeline (toàn bộ corpus)](#6-chạy-pipeline-toàn-bộ-corpus)
7. [Xem kết quả](#7-xem-kết-quả)
8. [Cấu trúc thư mục](#8-cấu-trúc-thư-mục)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Yêu cầu hệ thống

| | Yêu cầu |
|---|---|
| **OS** | Linux / WSL2 Ubuntu |
| **Python** | 3.10 – 3.12 (khuyến nghị 3.11) |
| **RAM** | ≥ 8 GB cho test (1k corpus); ≥ 16 GB cho full corpus |
| **Disk** | ≥ 10 GB (corpus + embedding cache) |
| **GPU** | Không bắt buộc cho retrieval; khuyến nghị cho local LLM |

---

## 2. Cài đặt môi trường

### 2a. Tạo conda environment

```bash
conda create -n linearag311 python=3.11 -y
conda activate linearag311
```

### 2b. Cài dependencies

```bash
cd /path/to/LinearRAG/SR-Agents

pip install -r requirements.txt

# Spacy NER model (bắt buộc cho LinearRAG)
python -m spacy download en_core_web_sm
```

---

## 3. Chuẩn bị dữ liệu

### Tải SRA-Bench từ HuggingFace

```bash
cd /path/to/LinearRAG/SR-Agents

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="WeihangSu/SRA-Bench",
    repo_type="dataset",
    local_dir="SRA-Bench",
)
PY
```

### Tạo symlink data

```bash
mkdir -p data
ln -s "$(pwd)/SRA-Bench" data/bench

# Kiểm tra
ls data/bench/corpus/corpus.json
ls data/bench/instances/
```

Sau bước này cấu trúc trông như sau:

```
SR-Agents/
├── data/
│   └── bench -> SRA-Bench/        # symlink
└── SRA-Bench/
    ├── corpus/corpus.json          # 26,262 skills
    └── instances/
        ├── theoremqa.json          # 747 queries
        ├── logicbench.json         # 760 queries
        ├── toolqa.json             # 1,430 queries
        ├── champ.json              # 223 queries
        ├── medcalcbench.json       # 1,100 queries
        └── bigcodebench.json       # 1,140 queries
```

---

## 4. Cấu hình LLM backend

Tạo file `.env` trong thư mục `SR-Agents/`. Pipeline tự load khi khởi động.

### Option A — Local model (mặc định trong script test)

Serve model bằng vLLM:

```bash
pip install vllm

vllm serve Qwen/Qwen3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
```

Không cần API key trong `.env`. Script mặc định dùng `http://localhost:8000/v1`.

### Option B — Timely API

```bash
# SR-Agents/.env
TIMELY_API_KEY=your_timely_key_here
```

Timely được tự động chọn khi `TIMELY_API_KEY` có mặt và không truyền `--api-base`.

### Option C — OpenAI

```bash
# SR-Agents/.env
OPENAI_API_KEY=sk-...
```

---

## 5. Chạy pipeline (test nhanh — RAM thấp)

Dùng cho máy RAM ≤ 8 GB hoặc khi muốn kiểm tra nhanh. Tự động giới hạn **1,000 skills** và **1,000 queries**.

### Bước 1 — Retrieval

```bash
conda activate linearag311
cd /path/to/LinearRAG/SR-Agents

bash run_retrieve_linearrag_test.sh
```

Mặc định chạy dataset `theoremqa`. Tuỳ chỉnh:

```bash
# Dataset khác
bash run_retrieve_linearrag_test.sh champ

# Nhiều datasets
bash run_retrieve_linearrag_test.sh champ logicbench

# Custom size
SUBSET_SIZE=500 MAX_INSTANCES=500 bash run_retrieve_linearrag_test.sh
```

Output lưu tại: `results/retrieval_test/{dataset}-linearrag-1000.json`

> **Lần đầu:** mất ~5–15 phút (NER + embedding 1k skills).
> **Lần sau:** dùng lại cache, xong trong vài giây.

---

### Bước 2+3 — Inference + Evaluation

Đảm bảo vLLM đang chạy (xem [mục 4](#4-cấu-hình-llm-backend)), rồi:

```bash
bash run_infer_eval_test.sh
```

Mặc định dùng:
- Model: `Qwen/Qwen3-8B-Instruct` tại `http://localhost:8000/v1`
- Engine: `progressive_disclosure`
- Dataset: `theoremqa`

Tuỳ chỉnh:

```bash
# Dataset khác
bash run_infer_eval_test.sh champ

# Dùng Timely API thay local
API_BASE="" MODEL=gpt-4o-mini bash run_infer_eval_test.sh

# Model khác, subset nhỏ hơn
MODEL=meta-llama/Llama-3.1-8B-Instruct SUBSET_SIZE=500 bash run_infer_eval_test.sh champ

# Nhiều datasets
bash run_infer_eval_test.sh champ logicbench theoremqa
```

Kết quả cuối:

```
=== Done. Results ===
  theoremqa: 142/223 (0.6368)
```

---

## 6. Chạy pipeline (toàn bộ corpus)

Dùng cho máy RAM ≥ 16 GB. Chạy toàn bộ 26,262 skills × 6 datasets.

### Bước 1 — Retrieval (tất cả 6 datasets)

```bash
bash run_retrieve_linearrag.sh
```

Hoặc từng dataset:

```bash
bash run_retrieve_linearrag.sh champ theoremqa logicbench
```

> **Thời gian:** Dataset đầu tiên mất **30–90 phút** (NER toàn corpus). Các dataset sau dùng cache NER, chỉ mất **2–5 phút/dataset**.

Output: `results/retrieval/{dataset}-linearrag.json`

### Bước 2+3 — Inference + Evaluation (qua experiment runner)

```bash
# LinearRAG top-1
sragents experiment --exp retrieval_comparison \
    --model Qwen/Qwen3-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --methods linearrag_top1 \
    --workspace results

# Progressive Disclosure (top-50 catalog)
sragents experiment --exp main \
    --model Qwen/Qwen3-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --methods progressive_disclosure \
    --workspace results
```

Chạy một dataset cụ thể:

```bash
sragents experiment --exp retrieval_comparison \
    --model Qwen/Qwen3-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --methods linearrag_top1 \
    --dataset champ \
    --workspace results
```

**Experiments có sẵn** (`sragents list experiments`):

| Tên | Mô tả |
|---|---|
| `main` | 5 methods: llm_direct, oracle, bm25_top1, llm_select, progressive_disclosure |
| `retrieval_comparison` | So sánh 6 retriever: BM25, TF-IDF, BGE, Contriever, BM25+Rerank, LinearRAG |
| `topk_sweep` | Sweep K ∈ {1,2,4,8} với Full Injection và Progressive Disclosure |
| `distractor` | Thêm N hard-negative distractors, đánh giá noise robustness |

---

## 7. Xem kết quả

### Retrieval metrics (Stage 1)

In ngay sau khi retrieve. Xem lại từ file:

```bash
python - <<'PY'
import json
data = json.load(open("results/retrieval_test/theoremqa-linearrag-1000.json"))
print("Retriever:", data["metadata"]["retriever"])
for k, v in data["metrics"].items():
    print(f"  {k}: {v:.4f}")
PY
```

### End-to-end accuracy (Stage 3)

```bash
python - <<'PY'
import json
data = json.load(open("results/eval_test/theoremqa-linearrag-1000.json"))
m = data["metrics"]
print(f"Accuracy: {m['correct']}/{m['total']} = {m['accuracy']:.4f}")
PY
```

### So sánh nhiều retriever

```bash
python - <<'PY'
import json
from pathlib import Path

results_dir = Path("results/retrieval")
datasets    = ["champ", "theoremqa", "logicbench"]
retrievers  = ["bm25", "bge", "linearrag"]

print(f"{'Retriever':<14}", end="")
for ds in datasets:
    print(f"  {ds:>14}", end="")
print()

for ret in retrievers:
    print(f"{ret:<14}", end="")
    for ds in datasets:
        path = results_dir / f"{ds}-{ret}.json"
        if path.exists():
            m = json.loads(path.read_text())["metrics"]
            print(f"  {m.get('Recall@10', 0):.4f}        ", end="")
        else:
            print(f"  {'N/A':>10}        ", end="")
    print()
PY
```

---

## 8. Cấu trúc thư mục

```
SR-Agents/
├── .env                                 # API keys
├── pyproject.toml
├── requirements.txt
│
├── run_retrieve_linearrag.sh            # Stage 1 — full corpus (6 datasets)
├── run_retrieve_linearrag_test.sh       # Stage 1 — 1k subset (RAM thấp)
├── run_infer_eval_test.sh               # Stage 2+3 — infer + eval (1k subset)
│
├── data/
│   └── bench -> SRA-Bench/             # symlink
│
├── SRA-Bench/                           # Dataset từ HuggingFace
│   ├── corpus/corpus.json
│   └── instances/
│
├── import/                              # Cache LinearRAG
│   ├── bench_full/                      # Cache cho full corpus
│   │   ├── ner_results.json
│   │   ├── passage_embedding.parquet
│   │   ├── entity_embedding.parquet
│   │   ├── sentence_embedding.parquet
│   │   ├── LinearRAG.graphml
│   │   └── skill_id_map.json
│   └── bench_1000/                      # Cache cho 1k subset
│       └── ...
│
├── results/
│   ├── retrieval/                       # Stage 1 output (full)
│   ├── retrieval_test/                  # Stage 1 output (test)
│   ├── infer_test/                      # Stage 2 output (test)
│   └── eval_test/                       # Stage 3 output (test)
│
└── src/sragents/
    ├── retrieve/
    │   ├── linearrag.py                 # LinearRAG adapter
    │   └── _linearrag/                  # Core implementation
    │       ├── core.py                  # Graph + PPR retrieval
    │       ├── config.py
    │       ├── embedding_store.py
    │       ├── ner.py
    │       └── utils.py
    ├── llm.py                           # LLM wrapper (Timely + OpenAI-compat)
    ├── timely_client.py                 # Timely API client
    └── experiments/definitions.py      # Experiment catalog
```

---

## 9. Troubleshooting

**`Terminated` / OOM khi chạy Stage 1**

RAM không đủ. Dùng script test thay vì full:
```bash
bash run_retrieve_linearrag_test.sh    # giới hạn 1k skills
```
Hoặc tăng RAM cho WSL: tạo `C:\Users\<user>\.wslconfig`:
```ini
[wsl2]
memory=12GB
```
Rồi `wsl --shutdown`.

**`TIMELY_API_KEY not set`**

```bash
cat SR-Agents/.env    # kiểm tra file có đúng vị trí không
```

**vLLM không start được**

```bash
# Kiểm tra port đang dùng
lsof -i :8000

# Giảm memory nếu GPU VRAM thấp
vllm serve Qwen/Qwen3-8B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.85
```

**NER chạy quá chậm**

```bash
--retriever-arg max_workers=2 --retriever-arg max_chars_per_passage=2000
```

**Resume khi bị gián đoạn**

- **Stage 1:** chạy lại cùng lệnh — NER cache được giữ, chỉ build lại graph
- **Stage 2:** chạy lại cùng `--output` — các instance đã done được bỏ qua tự động
- **Stage 3:** thêm `--force` để ghi đè kết quả cũ
