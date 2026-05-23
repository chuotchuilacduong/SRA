# CE v3 Pipeline — Mô tả chi tiết từng bước

Tài liệu này mô tả **toàn bộ** quy trình của phương pháp cross-encoder reranker v3:
**Listwise loss + Query-grouped batches + HYRR hybrid negatives + Cluster-hard + Balanced sampling**.

Mỗi bước có: (a) input, (b) output, (c) lệnh chạy, (d) ví dụ thực tế với 1 query.

---

## 0. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QUERY: "How many ways to                     │
│                          partition 8 elements into 5                 │
│                          non-empty ordered subsets?"                 │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌────────┐      ┌────────┐      (could add more retrievers)
   │  BM25  │      │  BGE   │
   │  top-K │      │  top-K │      Mỗi retriever score độc lập
   └───┬────┘      └───┬────┘
       │               │
       └───────┬───────┘
               ▼
       ┌──────────────┐
       │  RRF fusion  │      ← top-100 candidates (extended pool)
       └──────┬───────┘
              │
              ▼
       ┌──────────────────────────┐
       │  Cross-Encoder v3        │      ← Fine-tuned MiniLM-L6
       │  (listwise + HYRR)       │      Score mỗi (query, candidate)
       └──────────┬───────────────┘      bằng forward pass
                  │
                  ▼
            Final top-100
       (gold luôn được đẩy lên top)
```

**Hai giai đoạn**:
- **Stage 1 — First-stage retrieval**: BM25 + BGE → RRF top-100. Cheap, recall-friendly.
- **Stage 2 — Cross-encoder rerank**: CE v3 score (query, skill) pair. Expensive but precise.

---

## 0.1. Cách BIẾN ĐỔI DATA — full schema trace

Phần này trả lời chính xác: data thô đi qua **6 bước biến đổi schema** trước khi vào CE.

### Schema 0 — Raw SRA-Bench

#### Corpus (đầu vào, 26,262 skills)

```json
// data/bench/corpus/corpus.json  — JSON array
[
  {
    "skill_id": "theoremqa_000",
    "name": "Lah Numbers",
    "description": "Computing Lah numbers for counting ordered partitions...",
    "content": "# Lah Numbers ## What Lah Numbers Count ..." ,
    "tools": null              // optional, có ở 1 số skill
  },
  ...
]
```

#### Instances (đầu vào, 5,400 queries chia 6 datasets)

```json
// data/bench/instances/theoremqa.json
[
  {
    "instance_id": "theoremqa_00000",
    "dataset": "theoremqa",
    "question": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?",
    "skill_annotations": ["theoremqa_000"],          // gold (list vì 1 số dataset multi-label)
    "eval_data": {"answer": "11760", "answer_type": "integer"}
  },
  ...
]
```

### Schema 1 — Sau Stage-1 retrieval (BM25 + BGE)

Mỗi query × 100 candidates. Lưu thành `RetrievalResults` JSON:

```json
// results/retrieval_bm25/theoremqa-bm25.json
// results/bge/theoremqa-bge.json
{
  "metadata": {"retriever": "bm25", "top_k": 100, "dataset": "theoremqa", ...},
  "metrics": {"Recall@1": 0.697, "Recall@10": 0.918, ...},
  "results": [
    {
      "instance_id": "theoremqa_00000",
      "gold_skill_ids": ["theoremqa_000"],
      "retrieved": [
        {"skill_id": "theoremqa_000", "score": 18.4},
        {"skill_id": "theoremqa_137", "score": 14.1},
        ...
      ]
    },
    ...
  ]
}
```

### Schema 2 — Sau RRF fusion (extended pool)

```json
// results/pool/extended-theoremqa.json
{
  "metadata": {"retriever": "pool_extended_rrf", "top_k": 100, ...},
  "results": [
    {
      "instance_id": "theoremqa_00000",
      "gold_skill_ids": ["theoremqa_000"],
      "retrieved": [
        {
          "skill_id": "theoremqa_000",
          "score": 0.0327, "rank": 1,
          "is_gold": true,
          "candidate_recall_miss": false,
          "bm25_rank": 1, "bm25_score": 18.4,
          "bge_rank":  1, "bge_score":  0.71,
          "rrf_rank":  1, "rrf_score":  0.0327
        },
        ...
      ]
    }
  ]
}
```

**Biến đổi quan trọng**: thêm 5 trường provenance (`bm25_rank/score`, `bge_rank/score`, `is_gold`) cho **mỗi candidate** — sau này negative sampler dùng để chia hard-negs theo loại.

### Schema 3 — Sau split train/dev/test

```json
// results/splits/theoremqa-query_gen.json
{
  "train": ["theoremqa_00003", "theoremqa_00007", ...],  // 524 IDs (70%)
  "dev":   ["theoremqa_00001", "theoremqa_00012", ...],  //  74 IDs (10%)
  "test":  ["theoremqa_00000", "theoremqa_00002", ...]   // 149 IDs (20%)
}
```

**Biến đổi**: từ list dài of instances → 3 lists of `instance_id`. Lookup lại metadata ở instances file gốc.

### Schema 4 — Sau cluster assignment

```json
// results/clusters.json — 26,262 skills → 300 cluster_ids
{
  "theoremqa_000": 47,
  "theoremqa_137": 47,
  "theoremqa_092": 47,
  "theoremqa_086": 152,
  ...
}
```

**Biến đổi**: skill_id → cluster_id. Đây là lookup table dùng cho mining cluster-hard negs.

### Schema 5 — Sau HYRR negative mining (training pairs)

**Đây là format đầu vào trực tiếp của trainer.**

```json
// results/train/all_train_pairs_v2.json — 43,098 pairs
[
  // ----- pair POSITIVE -----
  {
    "instance_id": "theoremqa_00003",
    "skill_id": "theoremqa_000",          // gold
    "label": 1,
    "negative_source": "positive",
    "question": "How many ways are there to divide a set of 7 elements into 4 non-empty ordered subsets?",
    "skill": {
      "name": "Lah Numbers",
      "description": "Computing Lah numbers...",
      "content": "# Lah Numbers ## ..."
    },
    "gold_skill_ids": ["theoremqa_000"],
    "dataset": "theoremqa"
  },
  // ----- pair BM25_HARD (top BM25 nhưng không phải gold) -----
  {
    "instance_id": "theoremqa_00003",
    "skill_id": "theoremqa_137",
    "label": 0,
    "negative_source": "bm25_hard",
    "question": "...same question as above...",
    "skill": {"name": "Stirling Numbers 2nd Kind", ...},
    "gold_skill_ids": ["theoremqa_000"],
    "dataset": "theoremqa"
  },
  // ----- pair BGE_HARD -----
  { "negative_source": "bge_hard", ... },
  // ----- pair RANDOM -----
  { "negative_source": "random", ... },
  // ----- pair CLUSTER_HARD (cùng cluster 47 với theoremqa_000) -----
  { "negative_source": "cluster_hard", ... },
  ...
]
```

**Cấu trúc 1 query → 11 pairs**:

```
query "theoremqa_00003"
├── 1 positive   (label=1, source=positive)
├── 3 bm25_hard  (label=0, source=bm25_hard)
├── 3 bge_hard   (label=0, source=bge_hard)
├── 2 random     (label=0, source=random)
└── 2 cluster_hard (label=0, source=cluster_hard)
```

### Schema 6 — Tensors vào CE (sau tokenize + collate)

**Đây là input thực tế của model forward**.

```python
# Mỗi batch (1 query × 11 pairs) → tensors:
batch = {
    "input_ids":      LongTensor of shape (11, 256),    # token IDs
    "attention_mask": LongTensor of shape (11, 256),    # 1 = real token, 0 = pad
    "token_type_ids": LongTensor of shape (11, 256),    # 0 = query, 1 = skill text
    "labels":         FloatTensor of shape (11,),       # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    "query_ids":      LongTensor of shape (11,),        # [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
}
```

`query_ids` mapping `instance_id → int` để loss function biết pair nào thuộc query nào.

### Toàn bộ trace từ raw → tensor (1 ví dụ pair)

```
data/bench/instances/theoremqa.json
  instance:
    instance_id: "theoremqa_00003"
    question:    "How many ways... 7 elements into 4 non-empty ordered subsets?"
    skill_annotations: ["theoremqa_000"]
  │
  ▼
data/bench/corpus/corpus.json
  skill (cho candidate):
    skill_id:   "theoremqa_137"
    name:       "Stirling Numbers 2nd Kind"
    description: "..."
    content:    "..."
  │
  ▼
HYRR mining → pair dict (Schema 5):
    {instance_id, skill_id, label=0, negative_source="bm25_hard",
     question, skill, gold_skill_ids, dataset}
  │
  ▼
SkillPacker.pack_skill(skill) →
  "[SKILL_NAME] Stirling Numbers 2nd Kind
   [SKILL_DESCRIPTION] ...
   [SKILL_CONTENT] ..."
  │
  ▼
Tokenizer (max_length=256) →
  input_ids:      [CLS, t_q1, t_q2, ..., SEP, t_s1, t_s2, ..., SEP, PAD, PAD, ...]
  attention_mask: [1,   1,    1,    ..., 1,   1,    1,    ..., 1,   0,   0,   ...]
  token_type_ids: [0,   0,    0,    ..., 0,   1,    1,    ..., 1,   1,   1,   ...]
  │
  ▼
Stack 11 pairs vào batch → forward CE
```

---

## PART A — DATA PREPARATION (chạy 1 lần)

### Bước 1 — Build BM25 index trên 26,262 skills

**Input**: `data/bench/corpus/corpus.json` (26,262 skills với `name`, `description`, `content`)

**Quá trình**:
- Mỗi skill → text = `name + description + content`
- Tokenize (whitespace + punctuation, lowercase)
- Build sparse TF-IDF matrix → BM25 weighting (k1=1.5, b=0.75)
- Store: `vocab` (538k terms), `bm25_matrix` (26k × 538k sparse)

**Output**: BM25 retriever in-memory (rebuilt mỗi lần cần)

**Wall**: ~12s build, queries < 1s.

### Bước 2 — Encode corpus với BGE (chỉ 1 lần, cache trên đĩa)

**Input**: 26,262 skill texts.

**Quá trình**:
- Load model `BAAI/bge-base-en-v1.5` (768-dim, 110M params)
- Encode mỗi skill text → vector 768-dim, L2-normalized
- Save: `results/bge/corpus_emb.npy` (26262 × 768 float32, ~75 MB)

**Output**: `results/bge/corpus_emb.npy` + `corpus_ids.json`

**Wall**: ~12 phút CPU. Sau đó **dùng lại mãi mãi**.

### Bước 3 — Retrieve top-100 cho tất cả queries qua BM25 và BGE

Cho mỗi dataset (6 datasets):

**Input**: 
- Corpus embeddings (cached)
- Instances JSON với field `question`

**Quá trình** (BM25):
- Tokenize từng query
- Score: query_vec × bm25_matrix.T
- Lấy argsort top-100

**Quá trình** (BGE):
- Encode query với prefix `"Represent this sentence for searching relevant passages: "`
- Cosine similarity vs corpus_emb (đã normalize → dot product)
- Argsort top-100

**Output**:
- `results/retrieval_bm25/{ds}-bm25.json` (đã có sẵn)
- `results/bge/{ds}-bge.json`

Mỗi file format `RetrievalResults`:
```json
{
  "metadata": {"retriever": "bm25", "top_k": 50, ...},
  "metrics": {"Recall@10": 0.807, ...},
  "results": [
    {"instance_id": "theoremqa_00000",
     "gold_skill_ids": ["theoremqa_000"],
     "retrieved": [{"skill_id": "...", "score": 18.4}, ...]}
  ]
}
```

### Bước 4 — Fuse RRF → extended pool top-100

**Input**: BM25 top-50 + BGE top-100 cho mỗi dataset.

**Quá trình** ([build_all_pools.py](build_all_pools.py)):
- Với mỗi query, tính RRF score cho mỗi skill xuất hiện ở BM25 hoặc BGE:
  ```
  RRF_score(skill) = 1/(60 + rank_bm25) + 1/(60 + rank_bge)
  ```
  Nếu skill chỉ xuất hiện ở 1 retriever, term kia = 0.
- Sort theo RRF score, lấy top-100.
- **Quan trọng**: mỗi entry giữ provenance đầy đủ:

```json
{
  "skill_id": "theoremqa_000",
  "score": 0.0327,
  "rank": 1,
  "is_gold": true,
  "candidate_recall_miss": false,
  "bm25_rank": 1, "bm25_score": 18.4,
  "bge_rank":  1, "bge_score":  0.71,
  "rrf_rank":  1, "rrf_score":  0.0327
}
```

**Output**: `results/pool/extended-{ds}.json` cho 6 datasets.

### Bước 5 — Tách splits train/dev/test (zero leakage)

**Input**: instances cho mỗi dataset.

**Quá trình** ([strategy2_prepare.py](strategy2_prepare.py)):
- `query_gen` random split với seed=42.
- Tỉ lệ: train 70% / dev 10% / test 20%.
- Mỗi `instance_id` xuất hiện đúng 1 trong 3 splits → không leakage cấp query.

**Output**: `results/splits/{ds}-query_gen.json`:
```json
{
  "train": ["theoremqa_00000", "theoremqa_00003", ...],
  "dev":   ["theoremqa_00001", ...],
  "test":  ["theoremqa_00002", ...]
}
```

**Tổng**: 6 datasets × ~10-1430 queries = 5,400 total.
- train: 3,782 queries
- dev: 539 queries
- test: 1,079 queries

### Bước 6 — Cluster skills bằng K-means (chỉ 1 lần)

**Input**: `corpus_emb.npy` (26262 × 768)

**Quá trình** ([build_clusters.py](build_clusters.py)):
- `MiniBatchKMeans(n_clusters=300, batch_size=4096, max_iter=200)`
- Mỗi skill được gán vào 1 trong 300 clusters.

**Output**: `results/clusters.json`:
```json
{
  "theoremqa_000": 47,
  "theoremqa_137": 47,
  "theoremqa_092": 47,
  "logicbench_001": 152,
  ...
}
```

**Insight**: Skills cùng cluster = semantic neighborhood. K=300, median cluster size = 85.

**Wall**: ~9s.

### Bước 7 — Mine HYRR hybrid hard negatives (chỉ train queries)

**Input**:
- `results/pool/extended-{ds}.json` (top-100 candidates per query)
- `results/splits/{ds}-query_gen.json` (filter train queries)
- `results/clusters.json`

**Quá trình** ([strategy2_v2_prepare.py](strategy2_v2_prepare.py)) cho mỗi train query:

Cho query `q` với gold = `{g1}`:

| Source | Count | Cách chọn |
|---|---:|---|
| `positive` | 1 (hoặc nhiều nếu multi-label) | Lấy từ `gold_skill_ids` |
| `bm25_hard` | 3 | Top BM25 ranks, loại gold |
| `bge_hard` | 3 | Top BGE ranks, loại gold |
| `random` | 2 | Uniformly từ 26k corpus, loại gold + used |
| `cluster_hard` | 2 | Cùng cluster với gold, loại gold + used |
| **Tổng** | **11 pairs** | |

**Hard constraint**: trước mỗi sample check `sid ∉ gold_set` → **0 leakage gold**.

Mỗi pair output:
```json
{
  "instance_id": "theoremqa_00000",
  "skill_id": "theoremqa_137",
  "label": 0,
  "negative_source": "cluster_hard",
  "question": "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?",
  "skill": {
    "name": "Stirling Numbers of the Second Kind",
    "description": "...",
    "content": "..."
  },
  "gold_skill_ids": ["theoremqa_000"],
  "dataset": "theoremqa"
}
```

**Output**: 
- 6 files `results/train/{ds}-train_pairs_v2.json`
- 1 file gộp `results/train/all_train_pairs_v2.json` (**43,098 pairs** từ 3,782 train queries)

**Wall**: ~7s tổng.

---

## PART B — TRAINING (CE v3, 3 epochs)

### Bước 8 — Pack mỗi pair thành text input cho CE

**Input**: `train_pair` dict với `question` + `skill`.

**Quá trình** ([SkillPacker](src/sragents/retrieve/skill_packer.py), mode `field_tagged`):

```python
question = "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?"
skill = {"name": "Lah Numbers", "description": "Counting...", "content": "L(n,k) = ..."}

packed_skill = f"{NAME_TOKEN} {skill['name']} " \
               f"{DESC_TOKEN} {skill['description']} " \
               f"{CONTENT_TOKEN} {skill['content'][:1800]}"

# Final input to CE tokenizer:
input_pair = (question, packed_skill)
# Tokenizer adds [CLS] question [SEP] packed_skill [SEP], pad/truncate to 256 tokens.
```

**Output**: `(input_ids, attention_mask)` tensors, max_length=256.

### Bước 9 — Query-grouped batch sampling

**Input**: 43,098 pairs với fields `instance_id` + `dataset`.

**Quá trình** ([QueryGroupedBatchSampler](src/sragents/train/train_cross_encoder.py)):

1. **Group pairs by `instance_id`**: tạo 3,782 nhóm (mỗi nhóm = 1 query với 11 pairs).
2. **Compute weights**: mỗi query có weight = `1 / count(dataset của query)`. Sau normalize → mỗi dataset có marginal probability ≈ 1/6.
3. **Mỗi batch** = lấy `queries_per_batch=3` queries (theo weight) → concat 11+11+11 = ~33 pairs.

**Vì sao cần?**
- Listwise loss yêu cầu **softmax cross-entropy trên 1 group candidates của 1 query**.
- Nếu batch random shuffle, mỗi query chỉ có 1-2 pairs → softmax degenerate.
- Grouping ép positive + 10 negs cùng 1 query xuất hiện trong cùng forward pass.
- Balanced weighting tránh bigcodebench (10k pairs) lấn champ (1.8k pairs).

**Output**: iterator yields 1,261 batches/epoch, mỗi batch ~33 pairs từ 3 queries.

### Bước 10 — Forward pass + Listwise loss (CHI TIẾT NHẤT)

#### 10.1 — Input / Output shapes

| | Shape | Type | Mô tả |
|---|---|---|---|
| **Input batch** | | | |
| `input_ids` | `(33, 256)` | LongTensor | Token IDs cho 33 pairs (3 queries × 11 pairs). 256 = max_length. |
| `attention_mask` | `(33, 256)` | LongTensor | 1 = real token, 0 = padding. |
| `token_type_ids` | `(33, 256)` | LongTensor | Segment IDs: 0 cho query, 1 cho skill text. |
| `labels` | `(33,)` | FloatTensor | 1 cho positive pair, 0 cho negative. |
| `query_ids` | `(33,)` | LongTensor | int identifier — pair nào thuộc query nào. |
| **Forward output** | | | |
| `logits` | `(33,)` | FloatTensor | Relevance score (raw, không qua sigmoid) — model output đầu vào của loss. |
| **Loss output** | | | |
| `loss` | scalar | FloatTensor | Một số duy nhất, backprop tới toàn bộ weights. |

#### 10.2 — Forward pass

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1
)
# Architecture: BERT encoder (6 layers) + Linear(384 → 1)

# Input: (33, 256) token IDs
hidden_states = bert_encoder(input_ids, attention_mask, token_type_ids)
# hidden_states[:, 0]: [CLS] token embedding của 33 pairs, shape (33, 384)

logits = linear_head(hidden_states[:, 0]).squeeze(-1)
# logits shape: (33,) — 33 scalar relevance scores
```

**Score semantics**: logits ∈ ℝ. Cao nghĩa là relevant, thấp nghĩa là irrelevant. Chưa được calibrate.

#### 10.3 — Listwise softmax loss — công thức TOÁN

Cho 1 query group có `K` candidates (K=11 trong setup này):
- `s ∈ ℝᴷ`: logits của K candidates (từ forward).
- `y ∈ {0,1}ᴷ`: labels (1 positive, K-1 negatives, hoặc nhiều positives nếu multi-label).

**Step A — Build target distribution**:

```
target_i = y_i / Σⱼ y_j
```

Nếu chỉ có 1 positive: `target = [0, 0, ..., 1, ..., 0]` (one-hot, ở vị trí gold).
Nếu nhiều positive (multi-label): uniform trên các positions positive.

**Step B — Softmax over candidates of THIS query**:

```
p_i = exp(s_i) / Σⱼ exp(s_j)
```

Đây là softmax trên K candidates **của 1 query thôi**, không phải toàn batch.

**Step C — Cross-entropy loss**:

```
L_q = -Σᵢ target_i · log(p_i)
```

Với chỉ 1 positive ở index `g`:
```
L_q = -log(p_g) = -log( exp(s_g) / Σⱼ exp(s_j) )
       = -s_g + log( Σⱼ exp(s_j) )         # log-sum-exp form (numerically stable)
```

**Step D — Average over queries in batch**:

```
L_batch = (L_q1 + L_q2 + L_q3) / 3
```

#### 10.4 — MỤC TIÊU của hàm loss

```
Mục tiêu hình thức:
  Tìm θ (weights của CE) để  E_q[ -log(p_gold | s(q, ·; θ)) ]  → MIN
```

Diễn giải:

1. **Đẩy logit `s_gold` lên cao**, **kéo logits `s_neg` xuống thấp**, **đo bằng tương quan trong cùng query**.
   - L_q chỉ phụ thuộc vào `s_g - logsumexp(s_negs)`. Tăng `s_g` mà không tăng `s_negs` → L giảm.
   - Tăng `s_g` đồng đều với `s_negs` → L **không** đổi. Đây là khác biệt cốt lõi với BCE.

2. **Pairwise margin tự động**: Listwise gradient của candidate âm:
   ```
   ∂L_q / ∂s_neg_i = p_i = exp(s_neg_i) / Σ exp(s_j)
   ```
   Càng negative có logit cao (model lầm tưởng relevant) → gradient càng lớn → **học mạnh trên hard negatives**, gần như bỏ qua easy negatives. Đây là cơ chế tự động "weight hard negs more".

3. **Calibration KHÔNG quan trọng**: Sigmoid output của BCE buộc `s_pos ≈ +∞, s_neg ≈ -∞` để loss=0. Listwise chỉ cần `s_pos > s_neg`. → Model dễ học hơn cho retrieval (chỉ cần ordering đúng).

4. **Adapt với multi-label**: Mỗi positive nhận `target = 1/n_pos`. Loss vẫn = -log(softmax bag of positives) → phân phối đều xác suất giữa các golds.

#### 10.5 — Worked example NUMERIC

Giả sử batch chứa 1 query với 5 candidates (K=5):
```
candidates = [c1=gold, c2, c3, c4, c5]
logits  s = [2.0,     1.0,  0.5, -0.5, -1.0]
labels  y = [1,        0,    0,    0,    0]
target    = [1.0,      0,    0,    0,    0]
```

**Compute softmax**:
```
exp(s) = [e^2.0, e^1.0, e^0.5, e^-0.5, e^-1.0]
       = [7.389, 2.718, 1.649,  0.607,  0.368]
Σ     = 12.731
p      = [0.580, 0.214, 0.130,  0.048,  0.029]
```

**Loss**:
```
L_q = -log(p_gold) = -log(0.580) = 0.544
```

**Gradient interpretation**:
```
∂L/∂s_gold = p_gold - 1 = -0.420   ← push s_gold UP
∂L/∂s_2    = p_2       = +0.214   ← push s_2 DOWN
∂L/∂s_3    = p_3       = +0.130
∂L/∂s_4    = p_4       = +0.048
∂L/∂s_5    = p_5       = +0.029
```

Hardest negative (`c2`, logit 1.0) nhận gradient lớn nhất (+0.214). Easy negative (`c5`, logit -1.0) gradient nhỏ (+0.029) → ít update.

**Sau 1 step (lr=0.1)**:
```
s_new = s_old - lr * grad
      = [2.0+0.042, 1.0-0.021, 0.5-0.013, -0.5-0.005, -1.0-0.003]
      = [2.042,     0.979,     0.487,     -0.505,     -1.003]
```
→ Gold đẩy lên, hard negs kéo xuống mạnh hơn easy negs.

#### 10.6 — So sánh listwise vs BCE trên cùng example

**BCE loss** (nếu dùng v2):
```
L_bce = -Σᵢ [y_i log(σ(s_i)) + (1-y_i) log(1-σ(s_i))]
σ(s)  = [0.881, 0.731, 0.622, 0.378, 0.269]
L_bce = -[log(0.881) + log(1-0.731) + log(1-0.622) + log(1-0.378) + log(1-0.269)]
       = -[-0.127  - 1.312  - 0.973  - 0.475  - 0.314]
       = 3.201
```

**Quan sát**:
- Loss BCE = 3.20 (tổng over 5 pairs)
- Loss listwise = 0.54 (per query)

→ Scale khác. Quan trọng: **gradient direction**.

**BCE gradient `∂L_bce/∂s_2`** = `σ(s_2) - y_2` = `0.731 - 0 = 0.731`.
**Listwise gradient `∂L/∂s_2`** = `p_2` = `0.214`.

BCE push hard negative xuống mạnh **độc lập** với gold. Listwise push hard negative xuống chỉ **tương đối** với gold. Khi training data noisy (gold label không hoàn hảo), listwise robust hơn.

#### 10.7 — Code thực tế từ `src/sragents/train/losses.py`

```python
def listwise_softmax_loss(logits, labels, query_ids):
    """Per-query softmax-CE."""
    loss = logits.new_zeros(())
    count = 0
    for qid in torch.unique(query_ids):
        mask = query_ids == qid
        s = logits[mask]                # (K,)
        l = labels[mask].float()        # (K,)
        if l.sum() <= 0:
            continue                    # skip query if no positive in batch
        target = l / l.sum()            # (K,) — normalize to distribution
        log_p = F.log_softmax(s, dim=0) # (K,)
        loss = loss + -(target * log_p).sum()
        count += 1
    if count == 0:
        return bce_loss(logits, labels, query_ids)   # safe fallback
    return loss / count
```

Trick implementation:
- `F.log_softmax` thay vì `log(softmax)` → numerical stability.
- Iterate over `unique(query_ids)` chỉ trên CPU/GPU once per batch — cheap.
- Skip queries without positives (edge case).
- Fallback BCE nếu toàn batch không có query nào có positive.

#### 10.8 — Input → Output mapping (toàn batch)

```
INPUT:
  3 queries × 11 pairs = 33 pairs
  Mỗi pair: (question, packed_skill, label, query_id)
        │
        ▼
  Tokenize → (input_ids 33×256, attention_mask 33×256, ...)
        │
        ▼
  BERT encoder (6 layers, 384 hidden) → hidden_states (33, 256, 384)
        │
        ▼
  Pool [CLS] token → (33, 384)
        │
        ▼
  Linear(384 → 1) + squeeze → logits (33,)
        │
        ▼
LOSS:
  Group by query_id → 3 groups, each shape (11,)
  Softmax-CE per group → 3 scalar losses
  Mean → 1 scalar loss
        │
        ▼
OUTPUT:
  loss (scalar) → backward → AdamW step → updated weights
```

### Bước 10 (cũ, ngắn) — Forward pass + Listwise loss summary

**Input**: 1 batch ~33 pairs (3 queries × 11 pairs).

**Quá trình** ([train_cross_encoder.py](src/sragents/train/train_cross_encoder.py)):

```python
# 1. Tokenize batch
batch = tokenizer(questions, packed_skills, padding=True, truncation=True, max_length=256)

# 2. Forward through CE (MiniLM-L6, 22M params)
logits = model(**batch).logits.squeeze(-1)  # shape: (33,)

# 3. Listwise loss per query group
for qid in unique(query_ids):
    mask = (query_ids == qid)
    s = logits[mask]                # shape: (11,)
    l = labels[mask].float()        # shape: (11,)
    target = l / l.sum()            # one-hot or uniform over positives
    log_p = log_softmax(s, dim=0)
    loss_q = -(target * log_p).sum()
loss = mean(loss_q for each query)
```

**Insight tóm lược**: Softmax-CE đẩy `s_gold` cao tương đối so với `s_negs` của **cùng query đó**.
Khác BCE: BCE compute `σ(s)` rồi cross-entropy độc lập từng pair → không có "tương đối".

### Bước 11 — Backward + AdamW step

**Input**: scalar loss.

**Quá trình**:
- `loss.backward()` → gradients
- AdamW step: `lr=2e-5`, `weight_decay=0.01`
- Linear warmup 10% steps + linear decay
- Zero grad

**1 epoch = 1,261 steps**. **3 epochs = 3,783 steps total**.

### Bước 12 — Dev evaluation sau mỗi epoch

**Input**: `joint_dev_pool` (147 dev queries × 100 candidates).

**Quá trình**:
- Forward CE trên (query, candidate) pairs, score 14,700 pairs total.
- Sort theo score, compute `nDCG@10`, `Recall@K`, `MRR@K`...
- Lưu checkpoint nếu best.

**Output**:
- v3 epoch 0 dev nDCG@10 = 57.98%
- v3 epoch 1 dev nDCG@10 = 62.15%
- v3 epoch 2 dev nDCG@10 = **63.29%** (best, saved)

**Wall total training**:
- Epoch 0: ~40 min train + ~3.5 min eval
- Epoch 1: ~40 min + ~3.5 min
- Epoch 2: ~40 min + ~3.5 min
- **Total**: ~131 min CPU.

### Bước 13 — Save best checkpoint

**Output**:
```
results/models/ce-joint-v3/
├── pytorch_model.bin       # 22M params
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── train_config.json       # full hyperparams
└── train_summary.json      # per-epoch metrics
```

---

## PART C — INFERENCE (rerank 1 query)

### Bước 14 — Stage 1 retrieval cho query mới

Khi có query mới `Q`:

**Input**: `Q` = "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?"

**Quá trình**:
1. **BM25**: tokenize → score vs 26k skills → top-100.
2. **BGE**: encode `prefix + Q` → cosine vs 26k skills → top-100.
3. **RRF**: merge BM25 ranks + BGE ranks → top-100 với provenance.

**Output**: 100 candidate dicts:
```json
[
  {"skill_id": "theoremqa_000", "rrf_rank": 1, "rrf_score": 0.0327,
   "bm25_rank": 1, "bm25_score": 18.4, "bge_rank": 1, "bge_score": 0.71},
  {"skill_id": "theoremqa_137", "rrf_rank": 2, "rrf_score": 0.0312,
   "bm25_rank": 3, "bm25_score": 14.1, "bge_rank": 2, "bge_score": 0.68},
  ...
]
```

**Wall**: ~14 ms/query.

### Bước 15 — CE v3 rerank

**Input**: `Q` + 100 candidates.

**Quá trình** ([CrossEncoderReranker.rerank](src/sragents/retrieve/cross_rerank.py)):

```python
# 1. Build (query, packed_skill) pairs
packer = SkillPacker(mode="field_tagged")
pairs = [(Q, packer.pack_skill(corpus_dict[c["skill_id"]])) for c in candidates]

# 2. Tokenize batch (batch_size=64 by default)
# 3. Forward CE → 100 logits
scores = model.predict(pairs, batch_size=64)  # shape (100,)

# 4. Sort by score, attach provenance
ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])

# 5. Output: top-100 with new rank
```

**Output**: top-100 reordered:
```json
[
  {"skill_id": "theoremqa_000", "score": 8.42, "rank": 1, "is_gold": true,
   "bm25_rank": 1, "bge_rank": 1, "rrf_rank": 1},
  {"skill_id": "theoremqa_137", "score": 4.13, "rank": 2, "is_gold": false,
   "bm25_rank": 3, "bge_rank": 2, "rrf_rank": 2},
  ...
]
```

**Wall**: ~1,200 ms/query trên CPU (87× chậm hơn RRF).

---

## PART D — VÍ DỤ THỰC TẾ TRÊN 1 QUERY

### Query

```
instance_id: theoremqa_00000
question: "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?"
gold: ["theoremqa_000"]  (Lah Numbers)
answer: 11760
```

### Stage 1: BM25 top-3

```
rank 1: theoremqa_000 (Lah Numbers)              bm25_score=18.4
rank 2: theoremqa_137 (Stirling Numbers 2nd Kind) bm25_score=14.1
rank 3: theoremqa_092 (Set Partitions)            bm25_score=12.7
```

### Stage 1: BGE top-3

```
rank 1: theoremqa_000 (Lah Numbers)              bge_score=0.71
rank 2: theoremqa_137 (Stirling Numbers 2nd Kind) bge_score=0.68
rank 3: theoremqa_241 (Permutation Combinations)  bge_score=0.66
```

### Stage 1: RRF fusion top-3

```
RRF score:
  theoremqa_000: 1/61 + 1/61 = 0.0328  (BM25 r1, BGE r1)
  theoremqa_137: 1/63 + 1/62 = 0.0320  (BM25 r3, BGE r2)
  theoremqa_092: 1/63 + 1/inf = 0.0159 (BM25 r3, không trong BGE top-100)
```

→ RRF top-100 chứa gold ở rank 1. **Recall@1 = 100% trên query này.**

### Stage 2: CE v3 rerank

**Input pack** (cho candidate rank 1 = theoremqa_000):
```
[CLS] How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?
[SEP] [SKILL_NAME] Lah Numbers
      [SKILL_DESCRIPTION] Computing Lah numbers for counting ordered partitions of a set into non-empty ordered sequences.
      [SKILL_CONTENT] # Lah Numbers ## What Lah Numbers Count A Lah number L(n, k) counts the number of ways to partition a set of n distinct elements into k non-empty **ordered** lists ... L(n,k) = C(n-1, k-1) * n!/k!
[SEP]
```

→ Tokenize (truncate đến 256 tokens) → forward CE → **logit = 8.42**.

**Tương tự cho 99 candidates còn lại**. Sort theo logit:

```
rank 1 (CE): theoremqa_000  ce=8.42  (rrf_rank=1)  ✅ gold giữ ở rank 1
rank 2 (CE): theoremqa_137  ce=4.13  (rrf_rank=2)
rank 3 (CE): theoremqa_092  ce=2.85  (rrf_rank=4)
rank 4 (CE): theoremqa_086  ce=2.31  (rrf_rank=8)  ← CE pull up từ rank 8
...
```

CE v3 đã được train để recognize "Lah numbers" là **đúng câu trả lời** cho query "5 non-empty ordered subsets" — listwise loss giúp logit gold cao hơn đáng kể so với negs (8.42 vs 4.13).

### Kết quả

```json
{
  "skill_id": "theoremqa_000",
  "score": 8.42,
  "rank": 1,
  "is_gold": true,
  "candidate_recall_miss": false,
  "bm25_rank": 1, "bm25_score": 18.4,
  "bge_rank":  1, "bge_score":  0.71,
  "rrf_rank":  1, "rrf_score":  0.0328
}
```

→ Downstream LLM nhận skill `theoremqa_000` (Lah Numbers) → tính ra `L(8,5) = C(7,4) × 8!/5! = 35 × 336 = 11760` ✅.

---

## PART D.5 — INPUT → OUTPUT bảng tra cứu nhanh

Bảng dưới chứa **mọi giai đoạn** với Input, Output, shape/format cụ thể.

### Bảng I/O từng bước

| Bước | Input | Output | Shape/Format | Tool |
|---|---|---|---|---|
| **B1** BM25 index | corpus.json (26k skills) | BM25 retriever object | sparse matrix 26k×538k | `BM25Retriever.build_index` |
| **B2** BGE encode corpus | corpus.json (26k texts) | corpus_emb.npy | float32 array (26262, 768) | `SentenceTransformer.encode` |
| **B3** BM25/BGE retrieve | (query text, top_k=100) | List candidates với score | `[{skill_id, score}] × 100` | `Retriever.retrieve` |
| **B4** RRF fusion | 2 lists × 100 candidates | RRF-merged list | `[{skill_id, rrf_score, rank, *provenance}] × 100` | `multi_rrf_merge` |
| **B5** Split | instances list | train/dev/test ID lists | dict of 3 lists | `query_generalization_split` |
| **B6** K-means | corpus_emb (26k, 768) | clusters.json | `{skill_id: cluster_id}` | `MiniBatchKMeans` |
| **B7** HYRR mining | pool + splits + clusters | train pairs JSON | `[{instance_id, skill_id, label, source, ...}] × 43k` | `NegativeSampler.sample_for_query` |
| **B8** Pack | (question, skill dict) | string | `"[QUERY] q [SKILL_NAME] n [SKILL_DESCRIPTION] d [SKILL_CONTENT] c"` | `SkillPacker.pack_skill` |
| **B9** Batch sample | 43k pairs | iterator of batches | each batch: 33 pair indices | `QueryGroupedBatchSampler` |
| **B10** Tokenize | (question, packed_skill) × 33 | tokens dict | `input_ids (33, 256), attention_mask (33, 256), ...` | HF tokenizer |
| **B10** CE forward | tokens dict | logits | FloatTensor `(33,)` | `BERT + Linear(384→1)` |
| **B10** Listwise loss | (logits, labels, query_ids) | scalar loss | FloatTensor `()` | `listwise_softmax_loss` |
| **B11** Backward + step | scalar loss | updated weights | in-place on model | AdamW |
| **B12** Dev eval | dev pool (147 queries × 100 cands) | per-K metrics | dict `{Recall@1: ..., nDCG@10: ...}` | `compute_retrieval_metrics` |
| **B13** Save | model state | checkpoint dir | `pytorch_model.bin + config + tokenizer + summary` | `model.save_pretrained` |
| **B14** Inference stage 1 | (Q, corpus) | RRF top-100 | `[{skill_id, score, rank, provenance}] × 100` | BM25 + BGE + RRF |
| **B15** Inference stage 2 | (Q, top-100 candidates) | reranked top-100 | `[{skill_id, ce_score, rank, provenance, is_gold}] × 100` | `CrossEncoderReranker.rerank` |

### Một quan sát tổng quát về biến đổi

Toàn bộ pipeline có thể đọc theo 1 line duy nhất:

```
Raw JSON (corpus + instances)
  → Schema 1 (BM25 retrieved list)
    → Schema 2 (RRF-fused pool with provenance)
      → Schema 3 (split-filtered train queries)
        → Schema 4 (clustered corpus)
          → Schema 5 (HYRR pairs JSON)
            → Schema 6 (tokenized tensor batches)
              → CE forward → logits
                → Listwise loss → backward → updated CE weights
                  → Inference: rerank new query → final top-100
```

---

## PART D.6 — MỤC TIÊU LOSS một câu duy nhất

> **Mục tiêu: Cho mỗi query, đẩy logit của skill gold lên cao tương đối so với logits của tất cả candidate khác (trong cùng query đó), càng cao càng tốt, càng "đè bẹp" hard negative càng tốt — chứ không phải đẩy logit gold lên trị tuyệt đối cao.**

So với BCE/v2:

| Đặc tính | BCE (v2) | Listwise (v3) |
|---|---|---|
| Mục tiêu | `σ(s_gold) ≈ 1, σ(s_neg) ≈ 0` | `softmax(s_gold) > softmax(s_neg)` |
| Cần calibration | ✅ (phải push s đến ±∞) | ❌ (chỉ cần ordering) |
| Pair-level | Các pair độc lập | Pair-level so sánh trong cùng query |
| Hard-neg weighting | Đều nhau qua sigmoid | Tự động (gradient ∝ softmax prob) |
| Multi-label | Cần handcraft | Tự nhiên (target = 1/n_pos) |
| Phù hợp | Pointwise classification | Ranking/retrieval |

→ Đó là vì sao listwise (v3) giúp nhiều nhất cho các datasets khó (medcalc +8pp, champ +5pp, theoremqa +3pp).

---

## PART E — TÓM TẮT FLOW CODE

### Pipeline command sequence (reproducible)

```bash
# A. Data prep (1 lần)
python bge_all_datasets.py                                      # B2-3
python build_all_pools.py                                       # B4
python strategy2_prepare.py                                     # B5
python build_clusters.py                                        # B6
python strategy2_v2_prepare.py                                  # B7

# B. Training v3
sragents train-rerank --config configs/train_v3.yaml \
    --train-pairs results/train/all_train_pairs_v2.json \
    --dev-pool   results/pool/joint_dev_pool.json \
    --instances  results/joint_all_instances.json \
    --out        results/models/ce-joint-v3

# C. Eval
python rerank_test_split.py results/models/ce-joint-v3 s2-v3
python aggregate_full_bench.py

# D. Timing
python time_inference_full_bench.py --ce-model results/models/ce-joint-v3
```

### File map

```
src/sragents/
├── retrieve/
│   ├── bm25.py              # BM25 retriever
│   ├── dense.py             # BGE retriever
│   ├── fusion.py            # RRF
│   ├── cross_rerank.py      # CrossEncoderReranker
│   ├── skill_packer.py      # field_tagged packing
│   └── metrics.py           # Recall/nDCG/MRR/Hit/P @ K
└── train/
    ├── train_cross_encoder.py    # main loop + QueryGroupedBatchSampler
    ├── negative_sampler.py       # HYRR hybrid mining
    ├── losses.py                 # BCE/pairwise/listwise
    └── split_builder.py          # query_gen/skill_gen/ldo

configs/
├── data.yaml                # paths + field mapping
├── retrieval.yaml           # BM25/BGE/RRF hyperparams
├── train_v3.yaml            # CE v3 training config (listwise + grouped)
└── eval.yaml                # K cutoffs + probe thresholds
```

---

## PART F — HYPERPARAMETERS TÓM TẮT

| Hyperparameter | Value | Lý do |
|---|---|---|
| Base model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M params, CPU-fast, MS-MARCO pretrained baseline |
| max_length | 256 | đủ cho query + skill name + desc + ~1500 chars content |
| batch_size (nominal) | 33 (3 queries × 11) | đủ memory CPU, group đầy đủ pairs |
| lr | 2e-5 | conservative, không phá pretrained |
| epochs | 3 | đủ để consolidate cross-domain |
| weight_decay | 0.01 | mild regularization |
| warmup_ratio | 0.1 | tránh early divergence |
| loss | listwise softmax | per-query ranking pressure |
| negatives ratio | 3:3:2:2 (BM25:BGE:rand:cluster) | balance giữa lexical/semantic/random/semantic-cluster |
| RRF k | 60 | Cormack et al. default |
| K-means clusters | 300 | median 85 skills/cluster, đủ granularity |
| seed | 42 | reproducible |

---

## PART G — KẾT QUẢ FINAL

### Recall@10 (%) trên test split

| Dataset | RRF | CE v3 | Δ |
|---|---:|---:|---:|
| theoremqa | 90.60 | 82.55 | −8.05 |
| logicbench | 45.39 | **75.00** | **+29.61** |
| toolqa | 87.76 | **94.41** | **+6.65** |
| champ | 46.97 | **57.01** | **+10.04** |
| medcalc | 80.00 | 70.45 | −9.55 |
| bigcode | 66.16 | **90.23** | **+24.07** |
| **Macro** | **69.48** | **78.27** | **+8.79** |

CE v3 vượt RRF trên **4/6 datasets**, macro Recall@10 cao hơn RRF **+8.79pp**.

### Inference time (full bench 5,400 queries, CPU)

| Method | Total | QPS |
|---|---:|---:|
| BM25 | 20.6s | 540.0 |
| BGE | 52.5s | 102.9 |
| RRF (e2e) | 73.2s | 73.8 |
| **CE v3 (rerank only)** | **6,484.6s** | **0.83** |
| **CE v3 (e2e)** | **6,557.8s** | **0.82** |

CE v3 cho quality cao hơn nhưng **chậm ~89×** RRF. Trade-off rõ ràng cho deploy.
