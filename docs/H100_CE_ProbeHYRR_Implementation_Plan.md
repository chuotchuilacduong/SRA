# CE-ProbeHYRR H100 Pipeline — Implementation Plan

> Companion to [`H100_Qwen3_8B_CE_ProbeHYRR_Dataset_Pipeline.md`](H100_Qwen3_8B_CE_ProbeHYRR_Dataset_Pipeline.md).
> This doc maps that spec onto the **existing** SRA codebase, records the design
> decisions taken, corrects several wrong assumptions in the spec, and gives a
> phased, verifiable build order.

---

## 0. Decisions locked in

| # | Decision | Choice |
|---|---|---|
| D1 | Probe prompt construction | **Reuse `sragents.prompts.build_prompt`** (content-only skill injection, per-dataset systems). NOT the spec §10 generic templates — the existing deterministic extractors are tuned to the current prompts; switching would risk the <5% parser-failure gate and break comparability with the `skill_probe_hidden` prior art. |
| D2 | Downstream training target | **Binary first, multi-head later.** `utility_pairs` carry BOTH the rich category label and a collapsed `label ∈ {0,1}`. First CE-ProbeHYRR is trained with the existing binary BCE/listwise `train_cross_encoder`. The multi-head (utility/risk/false_friend) reranker from Architecture_v2 is deferred. |
| D3 | First-build scope | **valid_4 = TheoremQA + LogicBench + MedCalc-Bench + CHAMP** (2,830 queries). |
| D4 | qid convention | **Keep existing 5-digit `instance_id` verbatim** (`logicbench_00001`, `medcalcbench_00007`). The spec's 4-digit / abbreviated `medcalc_0007` form is illustrative only and would break every join to pools/instances. |
| D5 | CHAMP multi-gold policy | gold anchor injects **all gold skill contents jointly**; `gold_rank_m4 = min(rank over gold candidates)`; `gold_in_top50 = gold_rank_m4 ≤ 50`. |
| D6 | Generator | **Qwen/Qwen3-8B**, `enable_thinking=False`, `temperature=0.0`, `top_p=1.0`, vLLM offline batch. Labels are generator-specific → PreFlight's GO verdict must be re-confirmed on a pilot before the full build. |
| D7 | Env | Generation runs in a **dedicated `vllm` conda env** (numpy≥2 is fine there). Verification/assembly run CPU-only and can use either env. Do not install vLLM into `sra`. |

---

## 1. Spec → existing code map

| Spec concept | Existing artifact | Action |
|---|---|---|
| "M4 top-50" pool | `results/pool/hybrid_km_alpha30-{ds}.json` (RRF+KMeans α-blend, top-**100**) | slice top-50, reshape JSON→JSONL, rename `bge_rank→dense_rank`, derive `gold_rank_m4` |
| Deterministic verifier | `sragents.evaluate.evaluate(raw_output, instance) → {extracted_answer, correct}` | wrap in `ProcessPoolExecutor`; add `parse_failed`, `verifier_type` |
| Probe prompts | `sragents.prompts.build_prompt(instance, skills=[content])` | reuse directly (D1) |
| Skill content | `sragents.corpus.load_corpus_dict()` → `{skill_id: {skill_id,name,description,content}}` | reuse directly |
| Counterfactual utility | `utility = v_candidate − v_no` (already in `experiments/skill_probe_hidden/run_probe.py`) | port the per-query merge/regime logic |
| Candidate loaders | `experiments/skill_probe_hidden/data_io.py` | port loader patterns |
| CE trainer | `sragents.train.train_cross_encoder` (binary, `{instance_id, question, skill(dict), label}`) | feed collapsed-binary pairs (D2) |
| Qwen thinking-off / think-strip | `sragents.llm.get_extra_body(thinking=False)`, `strip_think_tags` | reuse for chat-template + post-gen safety |
| Utility metrics | `sragents.retrieve.metrics.compute_utility_metrics` | reuse in `report_quality` |

## 1a. Spec assumptions that are WRONG (corrected here)

- **"CHAMP has no split"** — false. `results/splits/champ-query_gen.json` exists (157/22/44). All four valid_4 `query_gen` splits sum to **2,830 / 282 / 565** train/dev/test, exactly the spec's valid_4 count.
- **"results/m4 and data/splits exist"** — false. Both are absent; Stage 0 must build them (this plan does).
- **qids are 4-digit / `medcalc`** — false. On disk they are 5-digit and use the full dataset name `medcalcbench` (D4).
- **Skill fields are `id`/`title`** — false. They are `skill_id`/`name`/`description`/`content`. `name` → "Title"; `skill_id` must never be shown to the model (see `corpus.display_name`).
- **M4 is regenerable from code** — the exact `hybrid_km_alpha30` builder (cluster_id/affinity fields) is not in the repo and its BGE/kmeans cache artifacts are missing. **Accept the existing pool JSONs as canonical M4.**

---

## 2. Phased build order

### Phase 0 — CPU foundation (no GPU; runs on dev node in `sra` env) ✅ implemented
New package `src/sragents/probehyrr_h100/`:
- `config.py` — valid_4, split protocol, per-dataset `max_new_tokens`, sampling/model defaults, output paths.
- `io_utils.py` — jsonl IO, `prompt_hash`, `cache_key`, atomic append.
- `build_splits.py` — `data/splits/{train,dev,test}.jsonl` (merged valid_4 from `query_gen`) + `data/corpus/skills.jsonl`.
- `build_m4_cache.py` — `results/m4/m4_top50_{split}.jsonl` from the pools.
- `prompt_templates.py` — wraps `build_prompt`; per-dataset generation config; optional Qwen chat-template materialization.
- `build_anchor_tasks.py` — `anchor_probe_tasks_{split}.jsonl` (no_skill / gold_skill / m4_top1), deduped by cache_key.

**Phase 0 exit check:** counts line up (train ≈ 2,830 queries → ≤ 3 anchor tasks each), every `m4_top1` skill_id resolves to corpus content, `gold_rank_m4` matches the pool, dry-run prints a valid probe plan.

### Phase 1 — H100 pilot (gate)
- Dedicated `vllm` env; download `Qwen/Qwen3-8B`.
- `run_batched_generation.py` — vLLM offline `LLM.generate`, length bucketing + sampling-param chunking, append-only JSONL + cache-key resume, capture `finish_reason`/`num_output_tokens`.
- `verify_outputs.py` — `ProcessPoolExecutor` over `evaluate`; emit `parsed_answer, verifier_score, parse_failed, verifier_type`.
- Run **500 queries** → gates: parser-failure <5%, verifier-failure <2%, ambiguous <10%, harmful + false_friend + helpful all present, no OOM. **Measure real throughput + GPU memory** (spec §16 table is untrusted; confirm `max_model_len` 8192 vs 16384 against the real skill-content length distribution).

### Phase 2 — UCB rounds (H100)
- `query_regimes.py` — `v_no/v_gold/v_m4/gold_rank_m4/gold_in_top50` → regimes.
- `microbatch_ucb.py` — 11 Option-A arms, `UCB = mean + √2·√(ln t / n_arm)`, global 50/arm warm-up, reward table, `arm_stats_{global,by_dataset}.json`.
- `build_ucb_tasks.py` — per round pick arm→best candidate-in-arm; B=3, 3 rounds.
- loop: build → generate → verify → `label_builder.py` → update arm stats.

### Phase 3 — assemble (CPU)
- `build_utility_pairs.py` — join skill dicts; emit rich category + binary label; include anchor rows.
- `build_training_groups.py` — G1–G5 (consumed via shared `instance_id` grouping).
- `report_quality.py` — reuse `compute_utility_metrics`; write `dataset_quality_report.md`, `ucb_policy_report.md`.

### Phase 4 — train CE-ProbeHYRR
- Call `train_cross_encoder()` programmatically with `TrainConfig(...)` (bypasses missing `configs/*.yaml`), `loss='listwise'`, `group_by_query=True`.

---

## 3. Key schemas

**`results/m4/m4_top50_{split}.jsonl`** (one line per query):
```json
{"qid":"medcalcbench_00000","dataset":"medcalcbench","split":"train",
 "query":"...","gold_skill_ids":["medcalcbench_046"],
 "gold_rank_m4":7,"gold_in_top50":true,
 "top50":[{"skill_id":"medcalcbench_026","rank_m4":1,"m4_score":1.0,
           "bm25_rank":4,"dense_rank":2,"cluster_id":132,"is_gold":false}, ...]}
```
- `m4_score` = pool normalized `score` (rank-1 = 1.0). `dense_rank` = pool `bge_rank`.
- `bm25_rank`/`dense_rank` are `null` for candidates absent from that branch (50% / 22%); UCB arm logic imputes a worst-rank sentinel (= 51).

**Anchor task row** (`anchor_probe_tasks_{split}.jsonl`): `task_id`, `split`, `qid`, `dataset`, `probe_kind ∈ {no_skill,gold_skill,m4_top1}`, `skill_ids`, `system`, `user`, `prompt_hash`, `cache_key`, `generation_config{model,enable_thinking,temperature,top_p,max_new_tokens}`, `metadata{rank_m4,is_gold,gold_rank_m4}`.

**Prompt-text policy (vLLM):** the manifest stores `(system, user)` and a `prompt_hash`; the **generator** applies `tokenizer.apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)` at gen time. This keeps Phase 0 GPU/tokenizer-free and still allows vLLM prefix-caching on the templated string.

---

## 4. Risks / watch-items

1. **Generator swap invalidates labels** — Qwen2.5:7B → Qwen3-8B; re-run the §17 gate pilot before trusting any GO verdict.
2. **Env conflict** — vLLM needs numpy≥2; the `numpy<2` pin (spacy ABI) means generation and any spacy step live in different envs.
3. **Long skill content** — may force `max_model_len=16384`; measure prompt-length distribution on the real corpus.
4. **Weak verifiers** — TheoremQA/CHAMP are numeric/symbolic; the validation protocol flags them weaker than code/MCQ. Keep `verifier_type` so they can be down-weighted later.
5. **`bm25_rank`/`dense_rank` = null** for many candidates → sentinel rank for the `high_bm25_low_dense` / `high_dense_low_bm25` arms.
6. **Label collapse** — `ambiguous`/`unverifiable` rows dropped from binary training pairs; `safe_but_unneeded`/`weak_false_friend` → negative; mapping recorded in `label_builder.py`.
7. **Data transfer** — ~480 MB of gitignored inputs (pools + splits + corpus) must reach the H100.
```
