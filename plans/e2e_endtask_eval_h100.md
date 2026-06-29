# End-to-end task-accuracy eval on H100 — Qwen3-4B & Qwen3-32B

Produces the SRA-paper-style **end-task accuracy** tables (one per model) on the held-out
**query_gen-test (1,079)**: real LLM agents solve the tasks with retrieved skills injected,
scored by task correctness (BigCodeBench by unit-test execution). Table shape:

```
Retrieval | Skill-use | theoremqa | logicbench | toolqa | champ | medcalcbench | bigcodebench | Average
```
- **Skill-use (5):** LLM Direct, Oracle Skill, Full-Skill Injection, LLM Selection, Progressive Disclosure.
- **Retrieval (6, only for the 3 retrieval-dependent strategies):** BM25, L6_final, CE-Raw·bge_base@1000, CE-Raw·rrf@1000, CE-Raw·bge_ft@1000, bge_ft (retriever-only).
- LLM Direct + Oracle Skill are retrieval-independent (`Retrieval = —`), computed once/model.
- **20 cells/model** (2 + 6×3); **Average = instance-weighted** (Σcorrect/Σtotal over the 1,079).

The whole run is driven by one experiment: **`python -m sragents.cli.main experiment --exp endtask`**
(no `sragents` console script is installed; invoke the module with `PYTHONPATH=src`). Added in
`src/sragents/experiments/definitions.py`). The runner auto-picks engines per dataset
(ToolQA→ReAct), resolves retrieval-source files, and is resume/idempotent.

**Decisions baked in:** ToolQA tool DBs **staged on H100**; Qwen3-32B served **bf16 TP2**
(≥2×80 GB H100); vLLM **--max-model-len 32768**; thinking **off** (do NOT pass `--thinking`).

---

## 0. Get the code on H100
```bash
cd /home/quynhtl/projects/SRA && git fetch origin && git checkout <branch> && git pull
export PYTHONPATH=src
```
New/changed (all tracked): `src/sragents/experiments/definitions.py` (`endtask` spec),
`src/kmeans/scripts/{run_fair_eval.py, ceraw_eval.py}` (retrieval-source exports),
`src/kmeans/scripts/make_test_instances.py`, `src/kmeans/scripts/aggregate_endtask_tables.py`.

## 1. Prerequisites on H100 (`data/`,`results/` symlinked to /mnt/data)
- `data/bench/corpus/corpus.json`, `data/bench/instances/{6}.json`, `results/splits/{6}-query_gen.json`.
- `results/retrieval_bm25/{ds}-bm25.json` (BM25 source).
- For source export: `results/m4_v2/cache/*` (L6 features), `results/bge/{corpus_emb.npy,corpus_ids.json}`,
  `results/models/ce-raw-v1` + `results/models/sr-emb-bge-v1` (else `bge_ft`/CE-Raw rows are skipped).
- **ToolQA tool DBs** staged at `data/external/toolqa/` (flights/coffee/agenda/scirex/dblp/yelp/airbnb)
  — without these, ToolQA ReAct observations are errors (acc invalid). Verify with the §6 sanity check.
- Env: `pip install -r requirements.txt` (BigCodeBench test deps) **+ `pip install vllm`** (separate env OK).
  **Unset `TIMELY_API_KEY` or always pass `--api-base`** — else the client silently routes to Timely, not vLLM.

## 2. Stage A — materialize retrieval-source files → `results/retrieval/{ds}-{source}.json`
```bash
# L6_final (also retrains/saves the final L6; writes results/retrieval/{ds}-l6_final.json)
python src/kmeans/scripts/run_fair_eval.py --grid moderate

# CE-Raw×3 + bge_ft retriever-only (CUDA: omit SRA_ALLOW_MPS). Writes:
#   {ds}-ceraw_bge_base.json {ds}-ceraw_rrf.json {ds}-ceraw_bge_ft.json {ds}-bge_ft_retriever.json
python src/kmeans/scripts/ceraw_eval.py --retrievers bge_base,rrf,bge_ft \
    --rerank-depth 1000 --ce-model results/models/ce-raw-v1

# BM25 source = the cached file under the name the runner expects
mkdir -p results/retrieval
for ds in theoremqa logicbench toolqa champ medcalcbench bigcodebench; do
  ln -sf ../retrieval_bm25/$ds-bm25.json results/retrieval/$ds-bm25.json
done
ls results/retrieval/   # expect 6 datasets × 6 sources = 36 files
```

## 3. Stage B — test-only instances (no `--split` flag exists)
```bash
python src/kmeans/scripts/make_test_instances.py   # -> data/bench/instances_test/{ds}.json, TOTAL 1079
```

## 4. Stage C — serve a model (OpenAI-compatible vLLM)
```bash
# Qwen3-4B (single H100)
vllm serve Qwen/Qwen3-4B  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90 --dtype bfloat16 &
# Qwen3-32B (tensor-parallel across 2× 80GB H100)
vllm serve Qwen/Qwen3-32B --port 8001 --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --dtype bfloat16 &
curl -s http://localhost:8000/v1/models   # health check
```
`enable_thinking=false` is applied client-side automatically for `qwen3*` models (do NOT pass `--thinking`).

## 5. Stage D — run all 20 cells per model (infer + evaluate)
```bash
# Qwen3-4B
python -m sragents.cli.main experiment --exp endtask --model Qwen/Qwen3-4B \
    --api-base http://localhost:8000/v1 \
    --instances-dir data/bench/instances_test \
    --workers 32 --eval-workers 8 --temperature 0.7 --max-tokens 4096

# Qwen3-32B
python -m sragents.cli.main experiment --exp endtask --model Qwen/Qwen3-32B \
    --api-base http://localhost:8001/v1 \
    --instances-dir data/bench/instances_test \
    --workers 32 --eval-workers 8 --temperature 0.7 --max-tokens 4096
```
Writes `results/inference/{ds}/{model_short}/{label}.jsonl` and `results/eval/{ds}/{model_short}/{label}.json`
(`model_short` = `Qwen3-4B`/`Qwen3-32B`; labels = `llm_direct`, `oracle_skill`, `{fsi|sel|pd}__{source}`).
Re-running is safe (per-instance JSONL resume; eval skips if output exists). `--eval-workers 8` keeps the
BigCodeBench subprocess pool modest (BCB eval is CPU-bound, needs no GPU — can be re-run on a CPU node).

> **BigCodeBench executes untrusted model code** (soft `reliability_guard` only — no container, network not
> blocked). On a shared box, consider running the BCB eval inside a container/VM and deciding egress policy.

## 6. Verification (do BEFORE the full sweep)
```bash
# (a) retrieval sources well-formed + cover test ids
python - <<'PY'
import json,glob
from sragents.corpus import load_corpus_dict
C=set(load_corpus_dict()); 
for f in glob.glob("results/retrieval/*-*.json"):
    d=json.load(open(f)); assert set(d)=={"results"} or "metadata" in d, f
    bad=sum(1 for r in d["results"] for x in r["retrieved"][:50] if x["skill_id"] not in C)
    print(f.split('/')[-1], "rows",len(d["results"]),"unknown_skill_ids(top50)",bad)
PY
# (b) ToolQA sanity — 5 instances, check Observations are real tool outputs (not errors)
python -m sragents.cli.main infer --instances data/bench/instances_test/toolqa.json \
    --output /tmp/tq.jsonl --model Qwen/Qwen3-4B --api-base http://localhost:8000/v1 \
    --provider topk --provider-arg source=results/retrieval/toolqa-bm25.json --provider-arg k=1 \
    --engine react --workers 4 --label probe   # then inspect /tmp/tq.jsonl transcripts
# (c) per-cell totals == dataset test sizes (149/152/286/44/220/228); Σ=1079
```
- `n/a` cells in the final table = that (model,label) eval JSON is missing (cell didn't run).
- Spot-check a `raw_output` has no `<think>` block (evaluators strip them anyway).

## 7. Stage E — build the two tables
```bash
python src/kmeans/scripts/aggregate_endtask_tables.py --models Qwen3-4B Qwen3-32B
# -> results/comparisons/endtask_{Qwen3-4B,Qwen3-32B}.{md,csv} + results/comparisons/endtask_tables.md
```
Average column = `100 · Σcorrect / Σtotal` over the datasets that ran (instance-weighted; matches the
paper's "overall over all instances"). Rows order: LLM Direct, Oracle Skill, then per retrieval method
the 3 retrieval-dependent strategies.

## 8. Risks / notes
1. **ToolQA data must be staged** — verify with §6(b) before the full ToolQA column; else that column is invalid.
2. **vLLM not in requirements** — `pip install vllm` (dedicated env recommended).
3. **Qwen3-32B needs TP≥2** on 80 GB H100s (bf16 weights ~64 GB).
4. **Context 32768** — bump `--max-model-len` to 65536 only if a long BigCodeBench prompt errors.
5. **CE-Raw/bge_ft sources** require `results/models/{ce-raw-v1,sr-emb-bge-v1}` present; missing → those rows skip.
6. Cost: 20 cells × 6 datasets × 2 models ≈ 43k instance-inferences (ReAct/PD multiply LLM *calls* per instance).
   Start with Qwen3-4B end-to-end, sanity-check the table, then run Qwen3-32B.

## Files (this plan)
| File | Role |
|---|---|
| `src/sragents/experiments/definitions.py` | `endtask` spec (20 methods; engines/sources) |
| `src/kmeans/scripts/run_fair_eval.py` | exports `results/retrieval/{ds}-l6_final.json` |
| `src/kmeans/scripts/ceraw_eval.py` | exports `{ds}-ceraw_{bge_base,rrf,bge_ft}.json` + `{ds}-bge_ft_retriever.json` |
| `src/kmeans/scripts/make_test_instances.py` | `data/bench/instances_test/{ds}.json` (1,079) |
| `src/kmeans/scripts/aggregate_endtask_tables.py` | pivots eval JSONs → the two tables |
| `src/sragents/cli/{infer,evaluate,experiment}.py`, `infer/providers/{topk,llm_select,oracle,none}.py`, `infer/engines/*` | harness (reference) |
