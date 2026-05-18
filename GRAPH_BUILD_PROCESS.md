# Quy Trình Build Graph Cho LinearRAG/BFSRAG

## 🎯 Tổng Quan

Hệ thống build một **TRI-GRAPH heterogeneous** (3 loại nodes, 2 loại edges) từ corpus skills:

```
┌──────────────────────────────────────────────────────────┐
│                 LINEARRAG TRI-GRAPH                      │
│                                                          │
│   PASSAGE nodes  ──(weighted)──>  ENTITY nodes          │
│       (skills)                    (NER-extracted)        │
│                                       │                  │
│                                       │ (binary 1.0)     │
│                                       ▼                  │
│                                 SENTENCE nodes           │
│                              (chứa entities)             │
└──────────────────────────────────────────────────────────┘
```

**Ý tưởng cốt lõi**: Query → entities → traverse graph qua entities/sentences → arrive at relevant passages.

---

## 📊 Cấu Trúc Graph Cuối Cùng (bench_full)

| Loại | Count | Chiếm | Vai trò |
|------|------:|------:|---------|
| **Passage nodes** | 26,262 | 6.9% | Skills cần retrieve |
| **Entity nodes** | 181,095 | 47.8% | NER-extracted (concepts, terms) |
| **Sentence nodes** | 171,536 | 45.3% | Câu chứa entities |
| **TOTAL nodes** | **378,893** | 100% | |
| **Passage→Entity edges** | 544,961 | 48% | Weighted by frequency |
| **Entity→Sentence edges** | 588,521 | 52% | Binary (1.0) |
| **TOTAL edges** | **1,133,482** | 100% | |

Connected components: **86** (1 giant chứa 99.96% nodes)

---

## 🔧 7 Bước Build Graph (Chi Tiết Code)

### **Bước 0: Input** — `index(passages)`

```python
passages = [
    "0:Newton-Raphson is an iterative method...",
    "1:Birge-Vieta finds polynomial roots...",
    ...
]
# Format: "{idx}:{description}\n{content}"
```

Source code: [src/sragents/retrieve/_linearrag/core.py:521](src/sragents/retrieve/_linearrag/core.py#L521)

---

### **Bước 1: Embed Passages → Save Cache**

```python
# core.py:526
self.passage_embedding_store.insert_text(passages)
```

**Chi tiết** ([embedding_store.py:35](src/sragents/retrieve/_linearrag/embedding_store.py#L35)):
```python
def insert_text(self, text_list):
    # Compute hash_id cho mỗi passage:  md5(text) + "passage-" prefix
    nodes_dict = {compute_mdhash_id(text, prefix="passage-"): {'content': text}
                  for text in text_list}

    # Skip texts đã có trong cache (check by hash)
    missing_ids = [h for h in nodes_dict if h not in existing]
    if not missing_ids:
        return  # All cached

    # Encode với BGE model
    embeddings = self.embedding_model.encode(
        texts_to_encode,
        normalize_embeddings=True,
        batch_size=64,
    )

    # Save to Parquet
    self._upsert(missing_ids, texts, embeddings)
```

**Output**: `import/bench_full/passage_embedding.parquet` (116 MB)

---

### **Bước 2: NER Extraction**

```python
# core.py:530-535
existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = \
    self.load_existing_data(hash_id_to_passage.keys())

if len(new_passage_hash_ids) > 0:
    new_passage_hash_id_to_entities, new_sentence_to_entities = \
        self.spacy_ner.batch_ner(new_hash_id_to_passage, max_workers)
```

**Chi tiết** ([ner.py:9](src/sragents/retrieve/_linearrag/ner.py#L9)):
```python
def batch_ner(self, hash_id_to_passage, max_workers):
    # Pipeline: sentencizer + tok2vec + ner
    # (parser/tagger disabled to avoid SIGBUS on Apple Silicon)
    passage_list = list(hash_id_to_passage.values())
    docs_list = self.spacy_model.pipe(passage_list, batch_size=batch_size)

    passage_hash_id_to_entities = {}
    sentence_to_entities = defaultdict(list)

    for idx, doc in enumerate(docs_list):
        for ent in doc.ents:
            # Skip ORDINAL (first, second) và CARDINAL (123)
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            sentence_to_entities[sent_text].append(ent_text)
            ...
```

**Output**: 
- `import/bench_full/ner_results.json` (81 MB)
- Format:
  ```json
  {
    "passage_hash_id_to_entities": {
      "passage-abc123": ["Newton-Raphson", "iterative method", "polynomial"],
      ...
    },
    "sentence_to_entities": {
      "Newton-Raphson is iterative.": ["Newton-Raphson", "iterative"],
      ...
    }
  }
  ```

**Speed**: ~15 skills/sec với spaCy `en_core_web_sm` → 26k skills mất ~28 phút

---

### **Bước 3: Extract Unique Nodes**

```python
# core.py:539
entity_nodes, sentence_nodes, passage_hash_id_to_entities, \
    self.entity_to_sentence, self.sentence_to_entity = \
    self.extract_nodes_and_edges(passage_entities, sent_entities)
```

**Chi tiết** ([core.py:648](src/sragents/retrieve/_linearrag/core.py#L648)):
```python
def extract_nodes_and_edges(self, passage_entities, sent_entities):
    entity_nodes = set()
    sentence_nodes = set()
    entity_to_sentence = defaultdict(set)
    sentence_to_entity = defaultdict(set)

    # Passage → entities
    for passage_hash_id, entities in passage_entities.items():
        for entity in entities:
            entity_nodes.add(entity)
            passage_hash_id_to_entities[passage_hash_id].add(entity)

    # Sentence → entities
    for sentence, entities in sent_entities.items():
        sentence_nodes.add(sentence)
        for entity in entities:
            entity_to_sentence[entity].add(sentence)
            sentence_to_entity[sentence].add(entity)

    return entity_nodes, sentence_nodes, ...
```

**Result**: 181,095 unique entities + 171,536 unique sentences

---

### **Bước 4: Embed Entities + Sentences**

```python
# core.py:542-544
self.sentence_embedding_store.insert_text(list(sentence_nodes))
self.entity_embedding_store.insert_text(list(entity_nodes))
```

Tương tự bước 1, encode với BGE model:
- **Entity embeddings**: 181k × 768 dims → 540 MB Parquet
- **Sentence embeddings**: 171k × 768 dims → 539 MB Parquet

**Speed**: ~1500 entities/sec với MPS GPU → ~3 min total

---

### **Bước 5: Compute Edge Weights (Passage ↔ Entity)**

```python
# core.py:555
self.add_entity_to_passage_edges(passage_hash_id_to_entities)
```

**Công thức**: `weight(passage, entity) = count(entity in passage) / total_entity_count_in_passage`

**Chi tiết** ([core.py:634](src/sragents/retrieve/_linearrag/core.py#L634)):
```python
def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
    passage_to_entity_count = {}
    passage_to_all_score = defaultdict(int)

    # Bước A: Đếm occurrence của mỗi entity trong mỗi passage
    for passage_hash_id, entities in passage_hash_id_to_entities.items():
        passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
        for entity in entities:
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            count = passage.count(entity)  # substring count
            passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
            passage_to_all_score[passage_hash_id] += count

    # Bước B: Normalize weight = count / total
    for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
        score = count / passage_to_all_score[passage_hash_id]
        self.node_to_node_stats[passage_hash_id][entity_hash_id] = score
```

**Ví dụ**:
```
Passage "Newton-Raphson method..." chứa:
  - "Newton-Raphson" xuất hiện 5 lần
  - "iterative" xuất hiện 3 lần
  - "convergence" xuất hiện 2 lần
Total: 10

Edge weights:
  passage → "Newton-Raphson":  5/10 = 0.50
  passage → "iterative":       3/10 = 0.30
  passage → "convergence":     2/10 = 0.20
```

**Total edges**: 544,961 passage→entity edges

---

### **Bước 6: Build igraph (Add Nodes + Edges)**

```python
# core.py:559
self.augment_graph()
```

**Chi tiết**:

#### 6a. Add Nodes ([core.py:602](src/sragents/retrieve/_linearrag/core.py#L602)):
```python
def add_nodes(self):
    # Combine entity + passage nodes (sentence nodes added separately)
    all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

    for hash_id, text in all_hash_id_to_text.items():
        if hash_id not in existing_nodes:
            self.graph.add_vertex(name=hash_id, content=text)

    # Build lookup: hash_id -> vertex_index (for fast lookup later)
    self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs}

    # Cache passage node indices (used for retrieval)
    self.passage_node_indices = [
        self.node_name_to_vertex_idx[passage_id]
        for passage_id in passage_hash_ids
    ]
```

#### 6b. Add Edges ([core.py:621](src/sragents/retrieve/_linearrag/core.py#L621)):
```python
def add_edges(self):
    edges = []
    weights = []
    for node_hash_id, neighbors in self.node_to_node_stats.items():
        for neighbor_hash_id, weight in neighbors.items():
            edges.append((node_hash_id, neighbor_hash_id))
            weights.append(weight)
    self.graph.add_edges(edges)
    self.graph.es['weight'] = weights
```

**Note**: Sentence nodes và entity-sentence edges được build separately trong `parquet_to_graphml.py` (cho trigraph viz). Trong `LinearRAG.graphml`, chủ yếu là passage-entity edges (bipartite-ish).

---

### **Bước 7: Save Final Graph**

```python
# core.py:560-572
output_path = "import/bench_full/LinearRAG.graphml"
try:
    self.graph.write_graphml(output_path)
except Exception as e:
    # Sanitize control chars (XML 1.0 forbids 0x00-0x1F except tab/newline/CR)
    _CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    for v in self.graph.vs:
        if 'content' in v.attributes() and isinstance(v['content'], str):
            v['content'] = _CTRL_RE.sub('', v['content'])
    self.graph.write_graphml(output_path)
```

**Output**: `import/bench_full/LinearRAG.graphml` (161 MB)

---

## 📐 Sơ Đồ Tổng Hợp

```
INPUT: 26,262 skills (each with name + description + content)
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 1: Passage Embeddings                     │
│   • BGE encode passages → 26,262 × 768 vectors  │
│   • Save: passage_embedding.parquet (116 MB)    │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 2: NER Extraction (~28 min)               │
│   • spaCy NER on each passage                   │
│   • Extract: ~181k entities, ~171k sentences    │
│   • Save: ner_results.json (81 MB)              │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 3: Embed Entities + Sentences             │
│   • BGE encode 181k entities → 540 MB           │
│   • BGE encode 171k sentences → 539 MB          │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 4: Compute Edge Weights                   │
│   • For each (passage, entity):                 │
│     weight = count_in_passage / total_entities  │
│   • Result: 544,961 weighted edges              │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 5: Build igraph                           │
│   • Add 207,357 nodes (passages + entities)     │
│   • Add 544,961 weighted edges                  │
│   • Cache node_name → vertex_idx mapping        │
└─────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────┐
│ STAGE 6: Persist                                │
│   • Save LinearRAG.graphml (161 MB)             │
│   • Total cache size: ~1.4 GB                   │
└─────────────────────────────────────────────────┘
   │
   ▼
GRAPH READY for retrieval (PPR or BFS)
```

---

## 🚀 Retrieval Stage (Sử Dụng Graph)

Sau khi graph được build, retrieval flow:

```python
def retrieve(query):
    # 1. NER trên query
    seed_entities = ner.extract(query)   # e.g., ["Newton-Raphson"]

    # 2. Tìm seed entities trong graph (cosine similarity)
    seed_indices = [find_best_match(e, entity_embeddings) for e in seed_entities]

    # 3. Graph traversal:
    #    PPR (LinearRAG):  graph.personalized_pagerank(reset=seed_weights)
    #    BFS (BFSRAG):     bfs_passage_scoring(seeds, decay=0.5, max_depth=4)

    # 4. Sort passages by score
    return top_k_passages
```

**Difference**:
- PPR: random walk → smooth probabilities cho TẤT CẢ passages
- BFS: deterministic traversal → sharp scores theo distance

---

## 🛠️ Tools/Files Liên Quan

| File | Vai trò |
|------|--------|
| [src/sragents/retrieve/_linearrag/core.py:521](src/sragents/retrieve/_linearrag/core.py#L521) | `index()` — main build entry |
| [src/sragents/retrieve/_linearrag/ner.py](src/sragents/retrieve/_linearrag/ner.py) | spaCy NER wrapper |
| [src/sragents/retrieve/_linearrag/embedding_store.py](src/sragents/retrieve/_linearrag/embedding_store.py) | Parquet-based embedding cache |
| [src/sragents/retrieve/_linearrag/utils.py](src/sragents/retrieve/_linearrag/utils.py) | Hash ID computation |
| [parquet_to_graphml.py](parquet_to_graphml.py) | Reconstruct full trigraph (with sentence nodes) for viz |
| [src/sragents/retrieve/_linearrag/bfs_search.py](src/sragents/retrieve/_linearrag/bfs_search.py) | BFS scoring algorithm |
| [run_ner_only.py](run_ner_only.py) | Standalone NER với checkpointing |
| [run_embed_only.py](run_embed_only.py) | Standalone embedding với checkpointing |

---

## ⏱️ Performance Profile

Build full bench (26,262 skills) trên Mac M1 48GB:

| Stage | Time | Output Size |
|-------|------|-------------|
| 1. Passage embeddings | ~7 min | 116 MB |
| 2. NER extraction | ~28 min | 81 MB |
| 3. Entity + sentence embeddings | ~3 min | 1.08 GB |
| 4. Compute edge weights | ~30 sec | (in memory) |
| 5. Build igraph | ~10 sec | (in memory) |
| 6. Save GraphML | ~10 sec | 161 MB |
| **TOTAL (first run)** | **~40 min** | **1.4 GB** |
| **Subsequent runs (cache hit)** | **~30 sec** | (no rebuild) |

---

## 💡 Insights Quan Trọng

1. **Tri-graph là bipartite-ish**: Passage và Entity nodes connect qua weighted edges; Sentence nodes làm "bridge" giữa entities

2. **Hash-based caching**: Mỗi text có md5 hash → skip recompute nếu đã có

3. **Edge weights chỉ dùng cho passage→entity**: Sentence-entity edges là binary (1.0)

4. **NER là bottleneck speed**: spaCy chậm nhất, ~28 min cho 26k skills → cần checkpointing

5. **Graph rất sparse**: 26k passages × 7 entities/passage avg = 180k edges (vs full bipartite 1.7B)

6. **97.6% nodes đều trong giant component** → BFS/PPR đều có thể explore hầu hết graph từ bất kỳ seed nào