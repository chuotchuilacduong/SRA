# Refined research prompt — SRA-Bench Skill-Aware Cross-Encoder Reranking

Bạn là senior ML engineer. **Repo đã tồn tại** ở `src/sragents/` (cài đặt như package `sragents`, CLI `sragents <subcommand>`). Nhiệm vụ là **mở rộng** repo này để thực hiện "Skill-Aware Cross-Encoder Reranking with HYRR-style Hybrid Hard Negatives". KHÔNG tạo skeleton mới — plug vào module và CLI đã có.

## Repo đã có sẵn (đọc trước khi viết code mới)

- `sragents.retrieve.base` — `Retriever` Protocol + `@register("name")` registry
- `sragents.retrieve.schema.RetrievalResults` — output schema (JSON, không phải JSONL), với `instance_id`, `gold_skill_ids`, `retrieved=[{skill_id, score}]`, `metadata`, `metrics`
- `sragents.retrieve.bm25.BM25Retriever` — scipy-sparse BM25 (đã register `"bm25"`)
- `sragents.retrieve.dense.DenseRetriever` — đã register `"bge"` với prefix `"Represent this sentence for searching relevant passages: "`
- `sragents.retrieve.fusion.multi_rrf_merge` — N-way RRF, `k_rrf=60`
- `sragents.retrieve.cross_rerank.CrossEncoderReranker` — pretrained CE reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU-forced vì MPS không ổn định)
- `sragents.retrieve.metrics.compute_retrieval_metrics` — đã có Recall/nDCG @ 1/5/10/50; cần **bổ sung** Hit@K, MRR@K, P@K, Recall@100, nDCG@100
- `sragents.corpus.load_corpus_dict`, `skill_text` — corpus loader, hiện ghép `name + description + content`
- CLI ở [src/sragents/cli/main.py](src/sragents/cli/main.py): subcommands `retrieve | hybrid | rerank | infer | evaluate | experiment | list`. Thêm subcommand mới qua pattern `add_parser` trong cùng package.

## SRA-Bench schema (THỰC TẾ — không bịa)

**Corpus** (`data/bench/corpus/corpus.json`, **JSON array**):

```
{skill_id, name, description, content, tools?}     # KHÔNG có "payload", KHÔNG có "domain"
```

**Instance** (`data/bench/instances/{dataset}.json`, JSON array, `dataset ∈ {theoremqa, logicbench, toolqa, medcalcbench, champ, bigcodebench}`):

```
{instance_id, dataset, question, skill_annotations, eval_data}
```

Field-name mapping cho prompt gốc:

- `query_id` → `instance_id`
- `query` → `question`
- `gold_skill_ids` → `skill_annotations`
- `domain` → `dataset` (lấy từ tên file, đã có ở field `dataset` của mỗi instance)
- `payload` → **không tồn tại**; tùy chọn render `tools` (callable list) thành string nếu cần `[SKILL_PAYLOAD]`, nếu không bỏ slot này khỏi packing mặc định

CHAMP và BigCodeBench là **multi-label** thực (`skill_annotations` có >1 phần tử). TheoremQA/LogicBench/ToolQA/MedCalcBench là single. Mapping field phải đặt ở `configs/data.yaml`, không hardcode.

## Hai tracks

- **paper_exact_track**: BM25 top-50 → CE rerank → eval. Đã có scratch script `run_paper_exact_rerank.py` — refactor thành CLI/module sạch, đừng để file rời.
- **extended_track**: RRF(BM25 top-K, BGE top-K) top-100 → CE rerank (fine-tuned) → eval. Tái dùng `multi_rrf_merge`.

Mỗi candidate trong rerank output phải giữ `bm25_rank`, `bge_rank`, `rrf_rank`, `bm25_score`, `bge_score`, `rrf_score` khi có; thêm `is_gold`, `candidate_recall_miss`. Query không có gold trong pool **vẫn giữ trong eval** với `candidate_recall_miss=true`.

## Module mới cần thêm (đặt đúng namespace)

| Path | Trách nhiệm |
|---|---|
| `sragents/retrieve/skill_packer.py` | Field-tagged packing + ablation modes (`title_only`, `title_description`, `title_description_content`, `field_tagged`, `field_tagged_maxp_chunks`). Tokens: `[QUERY] [SKILL_NAME] [SKILL_DESCRIPTION] [SKILL_CONTENT] [SKILL_PAYLOAD]`. Mở rộng — không thay — `corpus.skill_text` để không phá baseline đang chạy. |
| `sragents/retrieve/chunking.py` | MaxP: giữ `query + name + description` mỗi chunk, slide `content` với stride; score mỗi chunk, lấy `max`. |
| `sragents/train/negative_sampler.py` | HYRR-style: 4 BM25-hard + 4 BGE/RRF-hard + 2 random (+ tùy chọn cluster-hard nếu có `clusters.json`). **Hard constraint**: không bao giờ sample `skill_id ∈ skill_annotations`. |
| `sragents/train/split_builder.py` | Query-generalization (random theo `instance_id`), skill-generalization (giữ tập skill train/test rời nhau — nếu một query có gold cross-split thì đưa query đó về test để tránh leak), leave-domain-out (giữ 1 dataset làm test). |
| `sragents/train/losses.py` | `BCEWithLogitsLoss` (default), pairwise margin, listwise softmax-over-candidates. |
| `sragents/train/train_cross_encoder.py` | Fine-tune từ `cross-encoder/ms-marco-MiniLM-L-6-v2`. Auto device: CUDA + fp16 nếu có, MPS bị **disable** (đã biết unstable, theo comment ở `cross_rerank.py`), fallback CPU. Early-stop trên `dev_ndcg@10`. |
| `sragents/retrieve/cross_rerank.py` | Mở rộng `CrossEncoderReranker` để nhận `packer` và `chunker` injectable, hỗ trợ `model_path` đã fine-tune. **Không** viết class mới — extend cái đang có. |
| `sragents/retrieve/metrics.py` | Thêm `Hit@{1,10}`, `MRR@{10,100}`, `P@{1,5,10}`, `Recall@100`, `nDCG@100`. Phải đúng cho multi-label (ideal DCG = sum trên `min(|gold|, k)`). Export macro + micro + per-dataset. |
| `sragents/eval/latency_benchmark.py` | batch=1 (p50/p95 online), batch=32 throughput, cho top-50 và top-100. |
| `sragents/eval/failure_probes.py` | Heuristic gắn nhãn probe (formula-heavy: regex `$...$`/`\\frac` ; code/API-heavy: triple-backtick + `import`; multi-label: `len(skill_annotations) > 1`; long-skill: `len(content) > N`; lexical/semantic confounder: top-1 BM25 ≠ gold nhưng có BGE hit / ngược lại). |

## CLI subcommands mới (giữ phong cách hiện tại)

Thêm vào `sragents/cli/`:

- `sragents build-pool --track {paper_exact,extended} --dataset ... --top-k ...`
- `sragents mine-negatives --pool ... --ratio 4:4:2 --out train_pairs.json`
- `sragents make-splits --protocol {query_gen,skill_gen,ldo} --out splits/`
- `sragents train-rerank --config configs/train.yaml`
- `sragents rerank-topk --model ... --pool ... --packing field_tagged --maxp ...`
- `sragents bench-latency --model ... --batch {1,32} --top-k {50,100}`

Subcommand `retrieve`, `rerank`, `evaluate` đang có thì tái dùng — đừng tạo trùng. Mọi script chạy được qua CLI; nếu cần `scripts/` thì là thin wrappers gọi `sragents.cli.*`.

## Output format

Match `RetrievalResults` đã có (JSON, không phải JSONL). Mở rộng `retrieved[]` entry với:

```json
{"skill_id": "...", "score": 8.123, "rank": 1,
 "is_gold": true, "candidate_recall_miss": false,
 "bm25_rank": 12, "bm25_score": 18.4,
 "bge_rank": 41, "bge_score": 0.72,
 "rrf_rank": 7,  "rrf_score": 0.031}
```

`train_pairs.json` (array, để đồng nhất với corpus):

```json
{"instance_id": "...", "skill_id": "...", "label": 0,
 "negative_source": "positive|bm25_hard|bge_hard|rrf_hard|random|cluster_hard",
 "question": "...", "skill": {"name": "...", "description": "...", "content": "...", "tools": "..."},
 "gold_skill_ids": ["..."]}
```

Metrics: `metrics.json` + `metrics_by_dataset.csv` (dataset là field `dataset` của instance, không gọi là "domain" trong code).

## Tests (`tests/` ở repo root — chưa có, tạo mới)

Pytest, không yêu cầu GPU. `tests/fixtures/mini_bench/` chứa:

- `corpus.json` ~10 skills, có 1 skill `content > 2000` ký tự để trigger MaxP
- `instances/mini.json` ~5 questions, có ≥1 multi-label

Unit tests bắt buộc:

1. `test_rrf.py` — RRF tính đúng với input đã biết đáp số.
2. `test_negative_sampler.py` — sample 1000 lần, assert không skill_id nào trong sample trùng `skill_annotations`.
3. `test_metrics_multilabel.py` — Recall/nDCG/MRR cho query có 3 gold, retrieved có 2 trong top-10.
4. `test_chunking_maxp.py` — long content → ≥2 chunks → final score = `max`.
5. `test_split_builder.py` — skill-generalization: assert intersection skill train ∩ skill test = ∅.
6. `test_pipeline_smoke.py` — chạy full pipeline trên `mini_bench` không lỗi, sinh đủ artifacts.

## Config (`configs/`)

- `configs/data.yaml` — field mapping (`instance_id`, `question`, `skill_annotations`, `dataset`) + corpus/instances paths.
- `configs/retrieval.yaml` — BM25 (`k1=1.5, b=0.75`), BGE (`BAAI/bge-base-en-v1.5`, prefix giữ nguyên như code đang dùng), RRF `k_rrf=60`, top-K mỗi nhánh.
- `configs/train.yaml` — model, max_length=256, batch=16, lr=2e-5, epochs=3, early_stop=`dev_ndcg@10`, fp16 nếu CUDA, packing mode, neg ratio, loss.
- `configs/eval.yaml` — K cutoffs, probe thresholds.

## Ràng buộc cứng

- Python 3.10+, dùng `numpy/scipy/torch/transformers/sentence-transformers/pandas/sklearn/tqdm/pyyaml`. Đã có `rank_bm25` trong `requirements.txt`? Kiểm tra trước; BM25 đang chạy bằng scipy không phải `rank_bm25` — **không** thêm `rank_bm25` mới.
- Tất cả script nhận CLI args + YAML config path.
- Seed control (numpy + torch + python random), log qua `logging`.
- Type hints, docstring ngắn (theo phong cách module hiện tại — xem `fusion.py`).
- Không drop query âm thầm. Không hardcode metric số.
- README cập nhật `README.md` đang có (đừng tạo file mới) thêm section "Cross-Encoder Reranking Track" với exact command sequence.

## Acceptance criteria

1. `pytest tests/` xanh hết, CPU-only.
2. `mini_bench` chạy hết pipeline (build-pool → mine-negatives → train → rerank → evaluate → bench-latency) không lỗi.
3. Reproduce được track paper_exact (BM25@50 + CE pretrained) ngang với số trong `results/retrieval_rerank_paper/` (nếu đã có) — coi như sanity baseline.
4. Extended track chạy được trên ≥1 dataset đầy đủ (gợi ý: `theoremqa` vì nhỏ nhất).
5. Latency report là số đo thực, không hardcode.

## Order of implementation

1. `configs/data.yaml` + một loader nhỏ ánh xạ `instance_id↔query_id` để code mới ổn định trước.
2. Mở rộng `metrics.py` (Hit/MRR/P/@100) — viết test trước.
3. `skill_packer.py` + `chunking.py` + test.
4. `build-pool` CLI (gọi BM25/BGE/RRF đã có) → sinh `pool__<track>__<dataset>.json`.
5. `split_builder.py` + `negative_sampler.py` + test.
6. `train_cross_encoder.py` + `losses.py`.
7. Extend `cross_rerank.py` để nhận model đã train + packer/chunker.
8. `rerank-topk` CLI + `evaluate` mở rộng + `metrics_by_dataset.csv`.
9. `latency_benchmark.py`, `failure_probes.py`.
10. README diff + smoke pipeline trên `mini_bench`.

## Ghi chú nghiên cứu (để vào README, không bịa số)

Đóng góp **không phải** "chỉ là CE rerank" mà là: (a) skill-aware field packing, (b) HYRR-style hybrid hard negatives, (c) leakage-resistant split protocol (đặc biệt skill-generalization với multi-label), (d) robustness eval qua probe heuristic, (e) optional MaxP cho long skill. Latency/throughput là **measured outputs**, không claim.

---

**Khác biệt chính so với prompt gốc:** dùng đúng field names SRA-Bench (`instance_id/question/skill_annotations/dataset`, không có `payload/domain`), reuse module đã có thay vì viết lại (`BM25/BGE/RRF/CE/schema/metrics/CLI`), output JSON không JSONL (theo `RetrievalResults`), thêm subcommand thay vì script rời, disable MPS không chỉ "fp16 if CUDA" (vì code hiện tại đã biết MPS lỗi), bỏ "payload" khỏi default packing, làm rõ rằng multi-label chỉ áp dụng CHAMP/BigCodeBench.
