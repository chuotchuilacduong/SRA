# Chạy PreFlight diagnostic trên SSH H100 (Linux) — Runbook chi tiết

> Mục tiêu: chạy lại toàn bộ diagnostic (4 dataset single-shot **+ bigcodebench** code) trên H100,
> nơi verifier code-execution chạy native (macOS không chạy được — xem `results/preflight_bigcodebench/README.md`).

Ký hiệu: `H100` = `<user>@<host>`, thư mục đích `~/SRA`. Thay cho phù hợp.

---

## 0. Kiến trúc khi chạy (cùng 1 máy H100)

```
┌─────────────────────────────────────────────┐
│ H100 box                                     │
│  [GPU]  Ollama/vLLM  serve generator :PORT   │  ← model sinh câu trả lời
│  [CPU]  python -m ...preflight  ──HTTP──►     │  ← chấm verifier tất định, tính headroom
└─────────────────────────────────────────────┘
```
Preflight là CPU (chỉ gọi HTTP + chạy verifier). GPU chỉ để serve model.

---

## 1. Đưa CODE lên H100

Code mới (`experiments/probehyrr_validation/`, sửa `metrics.py`) **chưa commit**, nên `git clone` thường sẽ THIẾU. Chọn 1 trong 2:

### Cách A — Git (khuyến nghị)
Trên **máy local**, commit + push code mới:
```bash
cd /Users/hiro/Documents/Vinuni/code/SRA
git add experiments/probehyrr_validation/ src/sragents/retrieve/metrics.py new_docs/
git commit -m "preflight headroom diagnostic + bigcodebench cách-B"
git push origin hiro00-snapshot-2026-05-23
```
Trên **H100**:
```bash
git clone git@github.com:chuotchuilacduong/SRA.git ~/SRA      # lần đầu
cd ~/SRA && git checkout hiro00-snapshot-2026-05-23 && git pull
```

### Cách B — rsync code (không cần commit)
Trên **máy local** (loại data/results/.git nặng):
```bash
rsync -avP --exclude data --exclude results --exclude .git --exclude '*.pyc' \
  /Users/hiro/Documents/Vinuni/code/SRA/  H100:~/SRA/
```

---

## 2. Tạo Python env trên H100

```bash
cd ~/SRA
conda create -n sra python=3.11 -y && conda activate sra      # hoặc python -m venv
pip install -r requirements.txt
pip install -e .                                              # cài 'sragents' editable
python -c "import sragents, numpy, sympy, tree_sitter; print('env OK')"
```
> Lưu ý: `requirements.txt` ghim `transformers==4.46.3`. **Đừng** cài `vllm` vào CHUNG env này (vllm kéo transformers khác → xung đột). Serve model bằng **Ollama** (mục 4) hoặc vLLM trong **env riêng**.

---

## 3. Đưa DỮ LIỆU lên H100

`data/` và `results/` **không nằm trong git**. Hai nguồn:

### 3a. Bench data (corpus + instances) — tải lại từ HuggingFace (nhanh hơn copy)
```bash
cd ~/SRA
huggingface-cli download WeihangSu/SRA-Bench --repo-type dataset --local-dir data/bench
# -> data/bench/corpus/corpus.json (222M) + data/bench/instances/*.json
```

### 3b. Pool + rerank + splits (~258M, KHÔNG có trên HF/git) — rsync từ local
Chạy **trên máy local**:
```bash
cd /Users/hiro/Documents/Vinuni/code/SRA
ssh H100 'mkdir -p ~/SRA/results/pool ~/SRA/results/rerank ~/SRA/results/splits'
rsync -avP results/pool/hybrid_km_alpha30-*.json            H100:~/SRA/results/pool/
rsync -avP results/rerank/test-kmeans_M30-*-ce-v3-top20.json H100:~/SRA/results/rerank/
rsync -avP results/splits/                                   H100:~/SRA/results/splits/
```
> Đây là **M4 pool** (ứng viên) + **CE-HYRR baseline** (CE v3 top-1) + splits chống leak — preflight bắt buộc cần.

---

## 4. Serve generator (chọn 1)

### Cách 1 — Ollama (KHUYẾN NGHỊ: giống hệt local → số liệu so sánh trực tiếp được)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve >/tmp/ollama.log 2>&1 &        # hoặc systemd; dùng tmux để khỏi chết khi SSH ngắt
ollama pull qwen2.5:7b
curl http://localhost:11434/v1/models       # kiểm tra
```
→ `--model qwen2.5:7b --api-base http://localhost:11434/v1` (Q4 giống local).

### Cách 2 — vLLM (env riêng; nhanh hơn, hợp model lớn — NHƯNG là generator khác → phải re-baseline)
```bash
conda create -n vllm python=3.11 -y && conda activate vllm && pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16 --max-model-len 8192 \
  >/tmp/vllm.log 2>&1 &
curl http://localhost:8000/v1/models
```
→ `--model Qwen/Qwen2.5-7B-Instruct --api-base http://localhost:8000/v1`.

> **Nguyên tắc nhất quán:** generator của diagnostic phải = generator pipeline thật. Nếu pipeline dùng model lớn, đo headroom bằng chính model đó (số sẽ khác run local 7B — đó là điều bình thường, không phải lỗi).

---

## 5. Chạy PreFlight

Tạo `.env` (gitignored) hoặc truyền `--api-base` mỗi lần:
```bash
cd ~/SRA && conda activate sra
echo 'OPENAI_API_BASE=http://localhost:11434/v1' > .env     # khớp endpoint mục 4
```

### 5a. Bốn dataset single-shot (verifier tất định)
```bash
python -m experiments.probehyrr_validation.preflight \
  --datasets logicbench medcalcbench champ theoremqa \
  --per-stratum 40 --max-tokens 1024 --workers 16 \
  --model qwen2.5:7b --api-base http://localhost:11434/v1
# -> results/preflight/headroom_metrics.json
```

### 5b. bigcodebench (Linux chạy được — đây là phần macOS không làm được)
```bash
BIGCODEBENCH_TIMEOUT_PER_TASK=30 \
python -m experiments.probehyrr_validation.bigcodebench_preflight \
  --per-stratum 20 --workers 8 --max-tokens 1024 \
  --model qwen2.5:7b --api-base http://localhost:11434/v1
# -> results/preflight_bigcodebench/headroom_metrics.json (giờ HỢP LỆ trên Linux)
```
> Script này (cách B: verify đơn luồng + fork) an toàn hơn runner thường trên Linux. Các shim Darwin tự tắt.

### 5c. (Tùy chọn) Bước 0 saturation — 0 LLM, chạy ngay
```bash
python -m experiments.probehyrr_validation.gate0_pool_audit
```

---

## 6. Đọc kết quả
```bash
cat results/preflight/headroom_metrics.json
cat results/preflight_bigcodebench/headroom_metrics.json
python - <<'PY'
import json
for p in ["results/preflight/headroom_metrics.json","results/preflight_bigcodebench/headroom_metrics.json"]:
    d=json.load(open(p)); print(p, "->", d["verdict"]["decision"])
    for ds,m in d.get("per_dataset",{}).items():
        print(f"  {ds}: no_load={m['no_load']:.2f} harmful={m['harmful_exposure']:.2f} oracle_gap={m['oracle_gap']}")
PY
```

---

## 7. (Tùy chọn, nặng) toolqa agentic
toolqa cần thực thi tool thật → **không** chạy bằng single-shot. Để đo được:
1. Tải tool-data ToolQA về `data/external/toolqa/` (flights/airbnb/dblp/agenda/scirex... từ repo gốc night-chen/ToolQA).
2. Chạy bằng engine `react` (không phải `DirectEngine`). Đây là track riêng, làm sau.

---

## 8. Mẹo vận hành SSH
- Bọc cả serve model lẫn preflight trong **`tmux`** (`tmux new -s sra`) để không chết khi rớt SSH.
- preflight có **cache** (`results/preflight*/probe_cache.jsonl`) → chạy lại gần như tức thì, không probe lại.
- Theo dõi tiến độ: `tail -f results/preflight*/run.log` hoặc `wc -l results/preflight*/probe_cache.jsonl`.

---

## Tóm tắt checklist
```
[ ] 1. code lên H100 (git push+pull  HOẶC  rsync)        ← nhớ commit experiments/ + metrics.py
[ ] 2. conda env + pip install -r requirements.txt + pip install -e .
[ ] 3a. huggingface-cli download SRA-Bench -> data/bench
[ ] 3b. rsync results/{pool,rerank,splits} (~258M)
[ ] 4. serve model (Ollama qwen2.5:7b  HOẶC  vLLM)
[ ] 5a. preflight 4 datasets
[ ] 5b. bigcodebench preflight  ← chỉ chạy được ở đây (Linux)
[ ] 6. đọc headroom_metrics.json
```
