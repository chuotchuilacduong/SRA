# 🚀 LinearRAG Retrieval Scripts - Quick Start

## Bạn muốn chạy cái gì?

### ✅ **Tôi muốn test nhanh (RAM 8-12 GB)**

```bash
bash run_retrieve_linearrag_test_all.sh
```

**Kết quả:**
- Chạy tất cả 6 datasets
- Corpus: 1,000 skills (subset)
- Thời gian: ~20-30 phút
- Output: `results/retrieval_test/`

---

### ✅ **Tôi muốn chạy full benchmark (RAM 16+ GB)**

```bash
bash run_retrieve_linearrag_all.sh
```

**Kết quả:**
- Chạy tất cả 6 datasets
- Corpus: 26,262 skills (full)
- Thời gian: ~1-2 giờ
- Output: `results/retrieval/`

---

### ✅ **Tôi chỉ muốn chạy một vài datasets**

```bash
# Test mode - 1000 skills
bash run_retrieve_linearrag_test_all.sh champ theoremqa

# Full mode - 26k skills
bash run_retrieve_linearrag_all.sh champ theoremqa toolqa
```

---

### ✅ **Tôi bị OOM (Out of Memory)**

```bash
# Reduce memory usage
BATCH_SIZE=8 MAX_WORKERS=1 bash run_retrieve_linearrag_test_all.sh
```

---

## 📊 6 Datasets

```
┌─────────────────┬──────────┬─────────────────────────────┐
│ Dataset         │ Queries  │ Category                    │
├─────────────────┼──────────┼─────────────────────────────┤
│ theoremqa       │ 747      │ Theorem proving             │
│ logicbench      │ 760      │ Logical reasoning           │
│ toolqa          │ 1,430    │ Tool use                    │
│ champ           │ 223      │ Multi-step reasoning        │
│ medcalcbench    │ 1,100    │ Medical calculations        │
│ bigcodebench    │ 1,140    │ Programming tasks           │
├─────────────────┼──────────┼─────────────────────────────┤
│ TOTAL           │ 5,400    │                             │
└─────────────────┴──────────┴─────────────────────────────┘
```

---

## 🎯 Cách Sử Dụng Chi Tiết

### Script 1: `run_retrieve_linearrag_test_all.sh` (Khuyến nghị để bắt đầu)

**Mục đích:** Chạy nhanh với corpus nhỏ (1000 skills)

```bash
# Basic - chạy tất cả 6 datasets
bash run_retrieve_linearrag_test_all.sh

# Chạy 2-3 datasets
bash run_retrieve_linearrag_test_all.sh champ theoremqa

# Custom subset size
SUBSET_SIZE=500 bash run_retrieve_linearrag_test_all.sh
SUBSET_SIZE=2000 bash run_retrieve_linearrag_test_all.sh
```

**Environment Variables:**
```bash
SUBSET_SIZE=1000        # Corpus size (default: 1000)
MAX_INSTANCES=1000      # Max queries per dataset
BATCH_SIZE=16           # Embedding batch size (reduce if OOM)
MAX_WORKERS=1           # NER parallel workers
MAX_CHARS=4000          # Max passage length
```

**Output:** `results/retrieval_test/{dataset}-linearrag-1000.json`

---

### Script 2: `run_retrieve_linearrag_all.sh` (Production)

**Mục đích:** Chạy full benchmark với corpus đầy đủ (26k skills)

```bash
# Basic - chạy tất cả 6 datasets
bash run_retrieve_linearrag_all.sh

# Chạy một số datasets
bash run_retrieve_linearrag_all.sh champ logicbench toolqa

# Reduced memory
BATCH_SIZE=16 MAX_WORKERS=1 bash run_retrieve_linearrag_all.sh
```

**Environment Variables:**
```bash
BATCH_SIZE=32           # Embedding batch size
MAX_WORKERS=2           # NER parallel workers
MAX_CHARS=5000          # Max passage length
```

**Output:** `results/retrieval/{dataset}-linearrag.json`

---

### Script 3: `run_retrieve_linearrag.sh` (Legacy)

**Mục đích:** Chạy 1 dataset với full corpus

```bash
bash run_retrieve_linearrag.sh theoremqa
bash run_retrieve_linearrag.sh champ
```

---

## ⏱️ Thời Gian Chạy

### Test Mode (1000 skills)
```
MacBook Pro M1, 8GB RAM, batch_size=16, max_workers=1

Dataset 1 (theoremqa):  5-10 min  ← NER + embedding
Dataset 2 (champ):      2-3 min   ← reuse cache
Dataset 3 (logicbench): 2-3 min
...
TOTAL (6 datasets):     ~20-30 min
```

### Full Mode (26,262 skills)
```
MacBook Pro M1, 16GB RAM, batch_size=32, max_workers=2

Dataset 1: 30-90 min  ← NER (CPU-intensive, one-time)
Dataset 2-6: 5-10 min each ← reuse NER cache
TOTAL (6 datasets): ~1-2 hours
```

---

## 💾 Cache System

**Một lần lớn lần đầu, nhanh lần sau:**

```
FIRST RUN:
  NER extraction (parse skills):  30-90 min ⏱️
  Save to cache (ner_results.json)
  
SECOND RUN onwards:
  Load cache (ner_results.json):  1 sec ⚡
  Reuse embeddings
  Only rebuild graph: 1-3 min
```

**Cache locations:**
- `import/bench_full/` — Full corpus cache (26k skills)
- `import/bench_1000/` — Test subset cache (1k skills)

---

## 🛑 Troubleshooting

### Problem: `Killed: 9` (Out of Memory)

**Solution:** Reduce memory usage

```bash
# Option 1: Use test mode (smaller corpus)
bash run_retrieve_linearrag_test_all.sh

# Option 2: Reduce batch size
BATCH_SIZE=8 MAX_WORKERS=1 bash run_retrieve_linearrag_test_all.sh

# Option 3: Reduce corpus size further
SUBSET_SIZE=500 bash run_retrieve_linearrag_test_all.sh
```

### Problem: `spacy` module not found

```bash
conda activate linearag311
pip install spacy
python -m spacy download en_core_web_sm
```

### Problem: corpus.json not found

```bash
# Download SRA-Bench dataset
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="WeihangSu/SRA-Bench",
    repo_type="dataset",
    local_dir="data/bench",
)
PY
```

### Problem: `sragents` command not found

```bash
pip install -e .
```

---

## 📈 Check Results

### View output files:

```bash
# Test mode
ls -lh results/retrieval_test/*linearrag*

# Full mode
ls -lh results/retrieval/*linearrag.json
```

### Parse metrics:

```bash
python3 - <<'PY'
import json
data = json.load(open("results/retrieval_test/theoremqa-linearrag-1000.json"))
print(f"Recall@10: {data['metrics'].get('Recall@10', 'N/A')}")
print(f"Recall@50: {data['metrics'].get('Recall@50', 'N/A')}")
PY
```

### Compare all datasets:

```bash
python3 - <<'PY'
import json
from pathlib import Path

results_dir = Path("results/retrieval_test")
for filepath in sorted(results_dir.glob("*linearrag*.json")):
    data = json.loads(filepath.read_text())
    dataset = filepath.stem.split("-")[0]
    recall_50 = data['metrics'].get('Recall@50', 'N/A')
    print(f"{dataset:15} Recall@50: {recall_50}")
PY
```

---

## 🔗 Next Steps

After retrieval completes, run inference & evaluation:

```bash
bash run_infer_eval_test.sh
```

Or use experiment runner:

```bash
sragents experiment --exp retrieval_comparison \
    --model Qwen/Qwen3-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --methods linearrag_top1
```

---

## 📚 Additional Documentation

- **SCRIPTS_GUIDE.md** — Detailed usage guide (Vietnamese)
- **SCRIPTS_SUMMARY.txt** — Quick reference
- **README.md** — Full project documentation

---

## ❓ Quick FAQ

**Q: Which script should I use?**
- **Limited RAM (8GB):** `run_retrieve_linearrag_test_all.sh`
- **Powerful machine (16+GB):** `run_retrieve_linearrag_all.sh`

**Q: Why is the first run slow?**
A: NER extraction takes time. But the result is cached for reuse on subsequent datasets.

**Q: Can I run multiple datasets in parallel?**
A: Yes, open multiple terminals and run different datasets with different `SUBSET_SIZE` values.

**Q: Where are the results?**
- Test: `results/retrieval_test/{dataset}-linearrag-{size}.json`
- Full: `results/retrieval/{dataset}-linearrag.json`

**Q: How much disk space do I need?**
A: ~10GB (embeddings + cache + results)

---

**Ready to start?** Run:

```bash
bash run_retrieve_linearrag_test_all.sh
```

Enjoy! 🚀
