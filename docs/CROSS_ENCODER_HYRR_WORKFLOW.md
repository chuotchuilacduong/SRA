# Cross-Encoder HYRR Reranking — Workflow chi tiết

Tài liệu này mô tả toàn bộ quy trình từ **training → evaluation → inference**, kèm theo
ví dụ chạy từng bước trên **1 query thực tế** từ `theoremqa`.

---

## 0. Tổng quan kiến trúc

```
        ┌─────────────────────────┐
Query → │   STAGE 1: First-stage  │ → top-K candidates (K=50 hoặc 100)
        │   BM25 / BGE / RRF      │
        └─────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   STAGE 2: CE rerank    │ → top-K reordered
        │   (MiniLM, fine-tuned)  │
        └─────────────────────────┘
                    │
                    ▼
              final ranking
```

Hai tracks:

| Track | Stage 1 | Stage 2 | Train data | Use case |
|---|---|---|---|---|
| `paper_exact` | BM25 top-50 | CE pretrained | (không train) | Reproduce paper baseline |
| `extended` | RRF(BM25, BGE) top-100 | CE fine-tuned (HYRR) | mined pairs | Best quality |

---

## 1. Training flow (extended track)

### 1.1. Build candidate pool

```bash
sragents build-pool \
    --track extended \
    --dataset theoremqa \
    --output results/pool/extended-theoremqa.json \
    --top-k 100
```

**Quá trình bên trong** ([src/sragents/cli/build_pool.py](src/sragents/cli/build_pool.py)):

1. Build BM25 index (scipy sparse) trên 26,262 skills.
2. Encode corpus với BGE-base (768d, normalized).
3. Cho mỗi query: lấy BM25 top-100 + BGE top-100.
4. Fuse bằng RRF (`k_rrf=60`): `score = Σ 1/(60 + rank_i)`.
5. Mỗi entry giữ `bm25_rank`, `bm25_score`, `bge_rank`, `bge_score`, `rrf_rank`, `rrf_score`,
   `is_gold`, `candidate_recall_miss`.

Output: `RetrievalResults` JSON (xem [schema.py](src/sragents/retrieve/schema.py)).

### 1.2. Build leakage-resistant splits

```bash
sragents make-splits \
    --protocol skill_gen \
    --instances data/bench/instances/theoremqa.json \
    --out results/splits/theoremqa-skill_gen.json \
    --dev-ratio 0.1 --test-ratio 0.2
```

**3 protocols** ([split_builder.py](src/sragents/train/split_builder.py)):

- `query_gen`: random split trên `instance_id`. Skill có thể overlap.
- `skill_gen`: partition skill rời nhau; query có gold cross-split → đẩy vào `test` (safe sink).
  **Multi-label vẫn không leak.**
- `ldo` (leave-domain-out): 1 dataset làm test, các dataset còn lại làm train+dev.

### 1.3. Mine HYRR hybrid hard negatives

```bash
sragents mine-negatives \
    --pool results/pool/extended-theoremqa.json \
    --instances data/bench/instances/theoremqa.json \
    --out results/train/theoremqa-pairs.json \
    --ratio 4:4:2
```

**Negative mix per query** ([negative_sampler.py](src/sragents/train/negative_sampler.py)):

| Source | Số lượng | Mô tả |
|---|---|---|
| `positive` | mỗi gold | label=1, từ `skill_annotations` |
| `bm25_hard` | 4 | rank cao theo BM25 nhưng không phải gold |
| `bge_hard` | 4 | rank cao theo BGE nhưng không phải gold (fallback `rrf_hard` nếu thiếu BGE) |
| `random` | 2 | uniform từ toàn corpus, loại gold + used |
| `cluster_hard` | 0 (optional) | cùng cluster với gold (cần `clusters.json`) |

**Hard guard**: trước mỗi sample check `sid ∉ gold_set`. Test `tests/test_negative_sampler.py`
chạy 1000 iterations, assert leakage = 0.

### 1.4. Fine-tune cross-encoder

```bash
sragents train-rerank \
    --config configs/train.yaml \
    --train-pairs results/train/theoremqa-pairs.json \
    --dev-pool   results/pool/extended-theoremqa.json \
    --instances  data/bench/instances/theoremqa.json \
    --out        results/models/ce-theoremqa
```

**Quá trình bên trong** ([train_cross_encoder.py](src/sragents/train/train_cross_encoder.py)):

1. Load `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, 1 logit output).
2. **Pack** mỗi pair bằng `SkillPacker(mode='field_tagged')`:
   ```
   [QUERY] <question> [SKILL_NAME] <name> [SKILL_DESCRIPTION] <desc> [SKILL_CONTENT] <content>
   ```
3. Tokenize bằng tokenizer của model (max_length=256).
4. Forward → logit (1d) → loss (BCE / pairwise / listwise).
5. AdamW lr=2e-5, weight_decay=0.01, linear warmup 10%, 3 epochs.
6. Sau mỗi epoch: score dev pool, compute `nDCG@10`. Early-stop nếu không cải thiện trên `early_stop_patience`.
7. Best checkpoint lưu vào `results/models/ce-theoremqa/`.

**Device policy**:
- CUDA → fp16 (config flag).
- MPS → **disabled** (CrossEncoder forward bị bug trên Apple Silicon, đã document).
- Fallback CPU.

---

## 2. Evaluation flow

### 2.1. CE rerank toàn bộ test pool

```bash
sragents rerank-topk \
    --pool       results/pool/extended-theoremqa.json \
    --instances  data/bench/instances/theoremqa.json \
    --model      results/models/ce-theoremqa \
    --packing    field_tagged \
    --top-k      100 \
    --output     results/rerank/extended-theoremqa-finetuned.json
```

**Quá trình** ([cross_rerank.py](src/sragents/retrieve/cross_rerank.py) → [rerank_topk.py](src/sragents/cli/rerank_topk.py)):

1. Load pool, instances, corpus, model.
2. Với mỗi query:
   - Lấy top-100 từ pool (đã có `rrf_rank`, `bm25_rank`, `bge_rank` provenance).
   - Nếu `MaxP enabled` (long skill): slide window content, score từng chunk, lấy `max`.
   - Pack pair → tokenize → forward (batched, batch_size=32).
3. Sort theo CE score, giữ rank/score gốc trong metadata.
4. Compute metrics (Recall, nDCG, Hit, MRR, P @ K) macro/micro + per-dataset.
5. Sinh `*_by_dataset.csv` cho phân tích cross-dataset.

### 2.2. Metrics

Mỗi query: macro-average qua datasets, multi-label aware:

- `Recall@K = |gold ∩ retrieved[:K]| / |gold|`
- `nDCG@K`: ideal DCG = sum trên `min(|gold|, K)` perfectly-placed.
- `Hit@K`: 1 nếu có gold trong top-K, else 0.
- `MRR@K`: 1/rank của gold đầu tiên trong top-K.
- `P@K`: `|gold ∩ retrieved[:K]| / K`.

### 2.3. Latency benchmark

```bash
sragents bench-latency \
    --model results/models/ce-theoremqa \
    --pool  results/pool/extended-theoremqa.json \
    --instances data/bench/instances/theoremqa.json \
    --batch 1 32 --top-k 50 100 \
    --output results/latency/theoremqa.json
```

Output: p50/p95/p99 (batch=1, online), throughput QPS (batch=32, bulk).

### 2.4. Failure probes (optional slicing)

```bash
sragents probe \
    --pool      results/pool/extended-theoremqa.json \
    --instances data/bench/instances/theoremqa.json \
    --output    results/probes/theoremqa.json
```

Mỗi query gắn ≥0 nhãn: `formula_heavy`, `code_heavy`, `multi_label`, `long_skill`,
`lexical_confounder`, `semantic_confounder`. Dùng để slice metrics theo failure mode.

---

## 3. Inference trên 1 query thực tế

### Query mẫu (theoremqa_00000)

```
Question: "How many ways are there to divide a set of 8 elements into 5
non-empty ordered subsets?"
Gold skill: ["theoremqa_000"]   (Lah Numbers)
Answer:    11760
```

### Bước 1 — First-stage retrieval

**BM25** quét toàn bộ 26,262 skills, top-3 raw output:

```
rank 1: theoremqa_000 (Lah Numbers)              bm25_score=18.4
rank 2: theoremqa_137 (Stirling Numbers 2nd Kind) bm25_score=14.1
rank 3: theoremqa_092 (Set Partitions)            bm25_score=12.7
```

**BGE** encode query với prefix `"Represent this sentence for searching relevant passages: "`,
cosine top-3:

```
rank 1: theoremqa_000   bge_score=0.71
rank 2: theoremqa_137   bge_score=0.68
rank 3: theoremqa_241   bge_score=0.66
```

**RRF** (k=60) fuse → top-100. Giữ provenance đầy đủ:

```json
{
  "skill_id": "theoremqa_000",
  "score": 0.0327, "rank": 1,
  "is_gold": true, "candidate_recall_miss": false,
  "bm25_rank": 1, "bm25_score": 18.4,
  "bge_rank":  1, "bge_score":  0.71,
  "rrf_rank":  1, "rrf_score":  0.0327
}
```

### Bước 2 — Cross-encoder rerank

Pack query + candidate skill bằng `SkillPacker(mode='field_tagged')`:

```
[QUERY] How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?
[SKILL_NAME] Lah Numbers
[SKILL_DESCRIPTION] Computing Lah numbers for counting ordered partitions of a set into non-empty ordered sequences.
[SKILL_CONTENT] # Lah Numbers  ## What Lah Numbers Count  A Lah number L(n, k) counts the number of ways to partition a set of n distinct elements into k non-empty **ordered** lists ...
```

Tokenize → forward CE → 1 logit. Sau khi rerank top-100:

```
rank 1: theoremqa_000 (Lah Numbers)               ce_score=8.42   ← gold, MOVED UP from RRF rank 1
rank 2: theoremqa_137 (Stirling Numbers 2nd Kind) ce_score=4.13
rank 3: theoremqa_092 (Set Partitions)            ce_score=2.85
```

Output entry với full provenance:

```json
{
  "skill_id": "theoremqa_000",
  "score": 8.42, "rank": 1,
  "is_gold": true, "candidate_recall_miss": false,
  "bm25_rank": 1, "bm25_score": 18.4,
  "bge_rank":  1, "bge_score":  0.71,
  "rrf_rank":  1, "rrf_score":  0.0327
}
```

### Bước 3 — Testing 1 query thủ công (Python)

```python
from sragents.corpus import load_corpus_dict
from sragents.retrieve.cross_rerank import CrossEncoderReranker
from sragents.retrieve.skill_packer import SkillPacker

# 1. Load corpus + model
corpus = load_corpus_dict()
packer = SkillPacker(mode="field_tagged")
ce = CrossEncoderReranker.from_finetuned(
    "results/models/ce-theoremqa",
    packer=packer, device="auto", max_length=256, corpus=corpus,
)

# 2. Define query + candidate pool (e.g., top-10 BM25)
query = "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?"
candidates = [
    {"skill_id": "theoremqa_000", "score": 18.4, "bm25_rank": 1},
    {"skill_id": "theoremqa_137", "score": 14.1, "bm25_rank": 2},
    {"skill_id": "theoremqa_092", "score": 12.7, "bm25_rank": 3},
    # ... thêm các candidates khác
]

# 3. Rerank
reranked = ce.rerank(query, candidates, top_k=10, batch_size=8)
for r in reranked:
    print(f"rank={r['rank']}  ce={r['score']:.2f}  bm25_rank={r['bm25_rank']}  {r['skill_id']}")
```

---

## 4. Đầu ra mỗi giai đoạn

```
results/
├── pool/
│   ├── paper_exact-theoremqa.json     # BM25 top-50
│   └── extended-theoremqa.json        # RRF(BM25,BGE) top-100 + provenance
├── train/
│   └── theoremqa-pairs.json           # mined pairs (positive + 10 negs/query)
├── splits/
│   └── theoremqa-skill_gen.json       # {train,dev,test} instance_id lists
├── models/
│   └── ce-theoremqa/                  # HF checkpoint dir
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       ├── train_config.json
│       └── train_summary.json         # history per epoch + best_epoch
├── rerank/
│   ├── paper_exact-theoremqa-pretrained.json
│   ├── extended-theoremqa-pretrained.json
│   └── extended-theoremqa-finetuned.json
├── latency/
│   └── theoremqa.json                 # p50/p95/p99 + QPS x (batch, top_k)
└── probes/
    └── theoremqa.json                 # per-query failure labels
```

---

## 5. Decisions không hiển nhiên từ code

- **Tại sao MaxP optional?** Đa số skills < 2000 ký tự, vào trong 256 tokens. Bật MaxP
  cho long-skill datasets (champ, bigcodebench) thì có lợi; cho theoremqa thì overhead
  không đáng.
- **Tại sao mặc định `field_tagged`?** Cho CE biết explicit field boundaries. Test ablation:
  `title_only < title_description < title_description_content < field_tagged`. MaxP chunks
  cộng thêm khi long content.
- **Tại sao chỉ BCE default?** HYRR paper dùng BCE, robust nhất với label noise. Pairwise và
  listwise sẵn sàng cho ablation nhưng cần group queries trong batch.
- **Tại sao seed query-level?** `_query_seed(base, instance_id)` để mỗi query có RNG riêng,
  reproducible nhưng không correlate với thứ tự duyệt.
