# 📜 Hướng Dẫn Sử Dụng Scripts

## 🚀 Tổng Quát

Có 3 scripts chính để chạy retrieval trên SRA-Bench:

| Script | Corpus | Datasets | Use Case | RAM |
|--------|--------|----------|----------|-----|
| `run_retrieve_linearrag.sh` | Full (26k) | 1 cái | Test 1 dataset | 16+ GB |
| `run_retrieve_linearrag_all.sh` | Full (26k) | Tất cả 6 | Production | 16+ GB |
| `run_retrieve_linearrag_test_all.sh` | Subset (1k) | Tất cả 6 | Quick test | 8 GB |

---

## 📊 6 Datasets Trong Benchmark

```
bigcodebench     - 1,140 queries (programming)
champ            - 223 queries (multi-step)
logicbench       - 760 queries (logical reasoning)
medcalcbench     - 1,100 queries (medical calculations)
theoremqa        - 747 queries (theorem proving)
toolqa           - 1,430 queries (tool use)
─────────────────────────────────
TOTAL            - 5,400 queries
```

---

## 🎯 Quick Start

### **Máy RAM 8-12 GB (Recommended)**

Chạy test trên subset 1000 skills:

```bash
# Tất cả 6 datasets
bash run_retrieve_linearrag_test_all.sh

# Hoặc chọn một vài datasets
bash run_retrieve_linearrag_test_all.sh champ theoremqa

# Custom size (500 skills)
SUBSET_SIZE=500 bash run_retrieve_linearrag_test_all.sh
```

Output: `results/retrieval_test/{dataset}-linearrag-1000.json`

---

### **Máy RAM 16+ GB**

Chạy trên full corpus (26k skills):

```bash
# Tất cả 6 datasets
bash run_retrieve_linearrag_all.sh

# Hoặc chọn một vài datasets
bash run_retrieve_linearrag_all.sh champ theoremqa toolqa
```

Output: `results/retrieval/{dataset}-linearrag.json`

---

## ⚙️ Tùy Chỉnh Parameters

### **Giảm Memory Usage**

Nếu bị `Killed: 9` (OOM), thử:

```bash
# Giảm batch size
BATCH_SIZE=8 bash run_retrieve_linearrag_test_all.sh

# Giảm workers
MAX_WORKERS=1 bash run_retrieve_linearrag_test_all.sh

# Cả hai
BATCH_SIZE=8 MAX_WORKERS=1 bash run_retrieve_linearrag_test_all.sh

# Giảm passage length
MAX_CHARS=2000 bash run_retrieve_linearrag_test_all.sh
```

### **Giảm Corpus Size (Test Mode)**

```bash
# 500 skills
SUBSET_SIZE=500 MAX_INSTANCES=500 bash run_retrieve_linearrag_test_all.sh

# 2000 skills
SUBSET_SIZE=2000 bash run_retrieve_linearrag_test_all.sh
```

### **Tăng Speed (Full Mode)**

Nếu có đủ RAM & CPU:

```bash
# Parallel processing
MAX_WORKERS=4 BATCH_SIZE=64 bash run_retrieve_linearrag_all.sh
```

---

## 📈 Thời Gian Chạy (Ước Tính)

### **Test Mode (1000 skills)**

```
Machine: MacBook Pro M1, 8GB RAM, max_workers=1, batch_size=16

Dataset 1 (theoremqa):  5-10 min  ← NER + embedding
Dataset 2 (champ):      2-3 min   ← reuse NER cache
Dataset 3 (logicbench): 2-3 min
Dataset 4 (toolqa):     3-4 min
Dataset 5 (medcalc):    3-4 min
Dataset 6 (bigcode):    3-4 min
─────────────────────────────────
TOTAL:                  ~20-30 min
```

### **Full Mode (26262 skills)**

```
Dataset 1: 30-90 min   ← Full NER (CPU intensive)
Dataset 2: 5-10 min    ← Reuse NER cache
Dataset 3-6: 5-10 min each
─────────────────────────────────
TOTAL: 1-2 hours
```

---

## 🔍 Giám Sát Progress

### **Xem log realtime:**

```bash
# Terminal 1: Chạy script
bash run_retrieve_linearrag_test_all.sh

# Terminal 2: Monitor
watch -n 5 "ls -lh results/retrieval_test/*json | tail -10"
```

### **Check output files:**

```bash
# Test mode
ls -lh results/retrieval_test/*linearrag*

# Full mode
ls -lh results/retrieval/*linearrag.json
```

### **Xem statistics:**

```bash
# Xem metrics của một retrieval result
python3 - <<'PY'
import json
data = json.load(open("results/retrieval_test/theoremqa-linearrag-1000.json"))
print(f"Recall@10: {data['metrics'].get('Recall@10', 'N/A')}")
print(f"Recall@50: {data['metrics'].get('Recall@50', 'N/A')}")
print(f"Instances: {len(data['results'])}")
PY
```

---

## 🛑 Troubleshooting

### **Error: `Killed: 9` (Out of Memory)**

```bash
# Reduce batch size
BATCH_SIZE=8 MAX_WORKERS=1 bash run_retrieve_linearrag_test_all.sh
```

### **Error: `spacy` module not found**

```bash
conda activate linearag311
python -m spacy download en_core_web_sm
pip install spacy
```

### **Error: File not found (corpus.json, instances/)**

```bash
# Kiểm tra cấu trúc
ls data/bench/corpus/corpus.json
ls data/bench/instances/

# Nếu không có, download from HuggingFace
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="WeihangSu/SRA-Bench",
    repo_type="dataset",
    local_dir="data/bench",
)
PY
```

### **Error: `sragents: command not found`**

```bash
pip install -e .
```

---

## 📝 Output Format

Mỗi file output chứa:

```json
{
  "metadata": {
    "retriever": "linearrag",
    "dataset": "theoremqa",
    "retriever_args": {...}
  },
  "results": [
    {
      "instance_id": "q1",
      "retrieved": [
        {"rank": 1, "skill_id": "s123", "score": 0.95},
        {"rank": 2, "skill_id": "s456", "score": 0.91},
        ...
      ]
    }
  ],
  "metrics": {
    "Recall@10": 0.68,
    "Recall@50": 0.85,
    ...
  }
}
```

---

## 💡 Pro Tips

1. **Chạy test trước**: Dùng `run_retrieve_linearrag_test_all.sh` để kiểm tra setup trước khi chạy full
2. **Cache NER**: Lần đầu chạy dataset 1 lâu (30-90 min), nhưng các dataset sau nhanh hơn 10x vì reuse NER cache
3. **Parallel datasets**: Có thể chạy 2 script cùng lúc với `SUBSET_SIZE` khác nhau trong background
4. **Check disk space**: Embeddings + NER cache ~ 5-10 GB, dataset output ~ 100-500 MB/dataset

---

## 🔗 Related Scripts

```bash
# Inference (cần retrieval output từ trên)
bash run_infer_eval_test.sh

# Experiment runner (comprehensive pipeline)
sragents experiment --exp retrieval_comparison
```

---

## ❓ FAQ

**Q: Tôi nên chạy cái nào?**
- **Máy yếu (8GB RAM)**: `run_retrieve_linearrag_test_all.sh`
- **Máy mạnh (16+GB RAM)**: `run_retrieve_linearrag_all.sh`

**Q: Lần đầu chạy lâu, lần sau nhanh hơn không?**
- Có. NER cache được reuse. Dataset 1 mất 30-90 min, dataset 2-6 mỗi cái 2-5 min

**Q: Có thể skip datasets nào không?**
- Có: `bash run_retrieve_linearrag_all.sh champ theoremqa` (chỉ chạy 2 cái)

**Q: Output ở đâu?**
- Test: `results/retrieval_test/{dataset}-linearrag-{SUBSET_SIZE}.json`
- Full: `results/retrieval/{dataset}-linearrag.json`
