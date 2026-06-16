# CE-ProbeHYRR Dataset — End-to-End Runbook (v1)

> **Mục đích (VI):** Tài liệu này là quy trình vận hành đầy đủ để **tạo lại bộ dataset CE-ProbeHYRR đúng chuẩn** và **huấn luyện** mô hình từ nó. Nó thay thế phần "cách chạy" rải rác trong spec/plan, và ghi lại **lỗi v0 (truncation làm hỏng nhãn)** cùng cách khắc phục.
>
> **Companion docs:** design spec = [`H100_Qwen3_8B_CE_ProbeHYRR_Dataset_Pipeline.md`](H100_Qwen3_8B_CE_ProbeHYRR_Dataset_Pipeline.md); code map / decisions = [`H100_CE_ProbeHYRR_Implementation_Plan.md`](H100_CE_ProbeHYRR_Implementation_Plan.md). When they disagree with this runbook, **this runbook wins** for *how to run*; the spec wins for *intent*.

---

## 0. TL;DR

```text
1. (one-time) caps already raised in config.py; truncation gate added to report_quality.
2. Regenerate cleanly (GPU node):
     rm -rf results/ce_probehyrr_h100/{tasks,generations,verified,logs,state}
     bash run_probehyrr_full.sh
3. Check gates: every results/ce_probehyrr_h100/quality_report_{split}.json -> all_gates_pass: true
   (truncated_lt_10pct MUST be true now — it was false on v0).
4. Train:
     conda activate sra   # needs torch+transformers
     python -m sragents.probehyrr_h100.train_ce_probehyrr \
       --output-dir results/ce_probehyrr_h100/model_v0 --loss bce --epochs 3
```

The **final training input** is `results/ce_probehyrr_h100/utility_pairs_{train,dev,test}.jsonl`
(flat pairs, fed to the cross-encoder). `utility_train_groups_{split}.jsonl` is the
query-grouped view (regime/group-type analysis + listwise ranking).

---

## 1. Why v0 was NOT trainable (the defect this runbook fixes)

The first full build (2026-06-15) passed every §17 gate but its **labels were systematically
wrong** for 3 of the 4 datasets. Root cause:

- The spec assumed `enable_thinking=False` ⇒ short outputs ⇒ tiny `max_new_tokens` caps
  (logicbench 128 / medcalc 256 / champ 512) are enough.
- **Reality:** Qwen3-8B with `enable_thinking=False` still writes long step-by-step working
  and ignores "Return only the final answer". The caps cut off the answer **before** it is
  emitted. The deterministic verifier then grabs a wrong intermediate token — a step index
  (`"4."`), an input date (`"09/15/2023"`) — and scores it `0`.
- Because labels are **counterfactual** (`utility = v_candidate − v_no`; regimes derive from
  `v_no/v_gold/v_m4`), a truncation-biased verifier corrupts the whole label tree.

Measured correct-rate, **truncated vs. complete** (v0 train, all probes):

| dataset | % truncated | correct% when truncated | correct% when complete |
|---|---:|---:|---:|
| champ | 65% | **2.2%** | 71.2% |
| logicbench | 95% | **29.1%** | 82.7% |
| medcalcbench | 66% | **7.6%** | 57.8% |
| theoremqa | 2% | 0.0% (n=61) | 47.8% |

theoremqa (cap already 1024) is the **only clean** dataset. The §17 gates missed this because
`parse_failed` stays low (the extractor finds *a* token) and a label category is still
assigned. **Truncation was not gated.**

### Fix applied (already in the repo)
1. `config.py` → caps raised: logicbench/medcalc **1024**, champ **1536**, theoremqa 1024,
   default 1024; new constant `MAX_TRUNCATED_RATE = 0.10`.
2. `report_quality.py` → new gate `truncated_lt_10pct` folded into `all_gates_pass`.
3. New `train_ce_probehyrr.py` (Stage 10) bridges `utility_pairs` → the CE trainer.

> Open follow-up (not required to proceed): confirm `enable_thinking=False` is actually applied
> at gen time (`render_chat_template` in `run_batched_generation.py`). Raising the caps fixes the
> label defect either way, but if thinking is leaking, smaller caps + a stricter answer format
> could later restore the throughput the spec wanted.

---

## 2. Environment

| | |
|---|---|
| GPU | 1× H100-80GB. Generation stages (2, 6) need it; everything else is CPU. |
| Conda env | `sra` — `torch 2.8.0+cu128 · vllm 0.10.2 · transformers 4.56.1`. Driver is CUDA 12.8; **cu130 builds crash** (see memory `sra-vllm-env-cuda128`). |
| Package | `sragents` (src layout). Run modules as `python -m sragents.probehyrr_h100.<mod>`. |
| Allocate a node | `srun --partition=main --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 --pty bash` |
| ⚠️ /tmp | node-local & wiped with the allocation — keep all artifacts under `~/projects/SRA` (memory `sra-tmp-node-local`). |

`run_probehyrr_full.sh` activates `sra` and fails fast if `torch.cuda.is_available()` is False.

---

## 3. File layout

```text
data/splits/{train,dev,test}.jsonl          # Stage 0 — merged valid_4 queries (query_gen protocol)
data/corpus/skills.jsonl                     # Stage 0 — 26,262 skills {skill_id,title,name,description,content}
results/m4/m4_top50_{split}.jsonl            # Stage 0 — candidate pool (top-50) + gold_rank_m4 + query text
results/ce_probehyrr_h100/
  tasks/      anchor_probe_tasks_{split}.jsonl            # Stage 1   (generation_config baked in per row)
              ucb_probe_tasks_round_{r}_{split}.jsonl     # Stage 5
  generations/anchor_outputs_{split}.jsonl                # Stage 2   (raw_output, finish_reason, num_output_tokens)
              ucb_outputs_round_{r}_{split}.jsonl         # Stage 6
  verified/   anchor_verified_{split}.jsonl               # Stage 3   (parsed_answer, verifier_score, parse_failed, truncated)
              ucb_verified_round_{r}_{split}.jsonl         # Stage 7a
  query_regime_labels_{split}.jsonl          # Stage 4   (5 regime flags / query)
  logs/       ucb_candidate_probe_logs_{split}.jsonl       # Stage 7b (label_category + reward, append-only across rounds)
  state/      ucb_state_{split}.json                       # Stage 7b (arm_stats; updated once per round)
  utility_pairs_{split}.jsonl                # Stage 8   <-- TRAINER INPUT (flat pairs)
  utility_train_groups_{split}.jsonl         # Stage 9   <-- FINAL grouped view (G1–G5)
  quality_report_{split}.json                # gates
  model_v0/                                  # Stage 10  (trained CE checkpoint, written by the glue)
```

Scope = **valid_4** (TheoremQA + LogicBench + MedCalc-Bench + CHAMP), `query_gen` protocol,
**train 1983 / dev 282 / test 565 = 2830** queries. Splits are disjoint by `qid` (verified).
ToolQA / BigCodeBench are phase-2 (need tool-loop / execution sandbox).

---

## 4. Pipeline stages (what each one does)

| Stage | Module | GPU | In → Out | Notes |
|---|---|:--:|---|---|
| 0 | `build_splits`, `build_m4_cache` | | pools/corpus → `data/splits`, `data/corpus/skills.jsonl`, `results/m4/*` | idempotent; config-independent |
| 1 | `build_anchor_tasks` | | splits + m4 → `tasks/anchor_probe_tasks_*` | **bakes `generation_config` (incl. `max_new_tokens`) into every row**; overwrites file; dedups by `cache_key` |
| 2 | `run_batched_generation` | ✅ | anchor tasks → `generations/anchor_outputs_*` | vLLM offline batch; length-bucketed; no_skill / gold_skill / m4_top1 |
| 3 | `verify_outputs` | | anchor gens → `verified/anchor_verified_*` | CPU `ProcessPoolExecutor`; emits `verifier_score, parse_failed, truncated` |
| 4 | `query_regimes` | | anchor verified + m4 → `query_regime_labels_*` | `v_no/v_gold/v_m4/gold_rank_m4` → 5 flags |
| 5 | `build_ucb_tasks` | | m4 + regimes + prev-logs + state → `tasks/ucb_probe_tasks_round_{r}_*` | micro-batch UCB, 11 Option-A arms, B=3, also bakes `generation_config` |
| 6 | `run_batched_generation` | ✅ | ucb tasks → `generations/ucb_outputs_round_{r}_*` | same generator as Stage 2 |
| 7a | `verify_outputs` | | ucb gens → `verified/ucb_verified_round_{r}_*` | |
| 7b | `label_builder` | | ucb verified + anchors + m4 → append `logs/ucb_candidate_probe_logs_*`, update `state/ucb_state_*` | decision-tree `label_category` + reward; updates arm stats |
| 8 | `build_utility_pairs` | | anchors + probe-logs + m4 + skills → `utility_pairs_*` | joins skill dict; emits `label_category` + collapsed `label_binary`; **drops `ambiguous`/`unverifiable`**; includes anchor rows |
| 9 | `build_training_groups` | | utility_pairs + regimes → `utility_train_groups_*` | groups by `instance_id`; assigns `group_type` G1–G5 |
| — | `report_quality` | | logs + state + groups → `quality_report_*` | §17 gates **+ new truncation gate** |
| 10 | `train_ce_probehyrr` | ✅ | utility_pairs + m4 + corpus → `model_v0/` | glue → `train_cross_encoder()` |

Stages 5–7 loop `NUM_ROUNDS=3` (one candidate probe per query per round, budget B=3).
`run_probehyrr_full.sh` runs all anchors first, then Stages 4–9 per split.

---

## 5. Generation config (the part that was wrong)

`src/sragents/probehyrr_h100/config.py`:

```python
MODEL_ID = "Qwen/Qwen3-8B"; ENABLE_THINKING = False; TEMPERATURE = 0.0; TOP_P = 1.0
MAX_NEW_TOKENS = {"logicbench": 1024, "medcalcbench": 1024, "champ": 1536, "theoremqa": 1024}
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_TRUNCATED_RATE = 0.10     # report_quality gate
```

`MAX_MODEL_LEN` default **16384** (longest anchor prompt ≈ 9337 tok; 8192 overflows the xlong
bucket). With caps up to 1536 and prompts up to ~9.3k tok, `input_len + max_new_tokens` stays
< 16384, so the length filter in `run_batched_generation` won't silently drop tasks — but
**watch the `[gen] WARNING: skipping … task(s)`** line; if it appears, raise `MAX_MODEL_LEN`.

> Tuning trade-off: bigger caps = slower generation. After regen, if `truncated_rate` is far
> below 0.10 on a dataset you may lower its cap to recover throughput — but re-run the pilot
> gate first and never let any dataset cross `MAX_TRUNCATED_RATE`.

---

## 6. Regenerate the dataset (the correct, clean procedure)

### 6.1 Why you must delete derived artifacts first

Two footguns make a naïve re-run wrong:

1. **Caps are baked into task rows.** Each task carries its own `generation_config`. Editing
   `config.py` only changes *new* manifests — Stage 1 / Stage 5 must re-emit them. (Stage 1
   overwrites its file, so rebuilding the manifest is enough there.)
2. **`--force` appends, it does not truncate.** `run_batched_generation`/`verify_outputs`
   resume by `task_id` and **always write `append=True`**. `--force` only skips the resume
   check — it does **not** delete the old file. Re-running with `FORCE=1` over existing
   outputs therefore **duplicates rows** (old truncated + new), which then double-counts
   downstream. So do **not** rely on `FORCE=1` for a config change — delete first.

### 6.2 Command (on an allocated GPU node)

```bash
cd ~/projects/SRA

# Keep config-independent inputs (data/splits, data/corpus, results/m4).
# Delete everything derived from the OLD caps:
rm -rf results/ce_probehyrr_h100/{tasks,generations,verified,logs,state}

# Full rebuild. Stage 0 re-runs (idempotent); Stage 1 re-bakes the new caps;
# nothing to resume, so no duplication.
bash run_probehyrr_full.sh
```

Useful overrides (env vars): `SPLITS="train"` (one split), `MAX_MODEL_LEN=…`,
`GPU_MEM_UTIL=0.95`, `NUM_ROUNDS`/`BUDGET`. `SKIP_PREP=1` skips Stage 0 **and** Stage 1 —
do **not** use it here, because Stage 1 must rebuild the manifests with the new caps.

Each generation stage cold-starts vLLM (~90s) per the per-stage subprocess design — expected.

### 6.3 Optional fast pilot before the full run

```bash
SPLITS="train" LIMIT_QUERIES=500 bash run_probehyrr_full.sh
python -m sragents.probehyrr_h100.report_quality --split train
```
Proceed to the full build only if the §17 gates **and** `truncated_lt_10pct` pass.

---

## 7. Acceptance gates (must all pass before training)

`report_quality` writes `quality_report_{split}.json`. Required: `all_gates_pass: true` for
**every** split, with:

```text
parse_failure_lt_5pct        true   (extractor health)
truncated_lt_10pct           true   <-- NEW; was false on v0, the whole reason for the regen
ambiguous_lt_10pct           true
harmful_present              true
strict_false_friend_present  true
positive_present             true
```

Also sanity-check (informational, not gated): label distribution has all categories,
`group_types` covers G1–G5, `groups_with_positive` ≈ 50% (the rest are no-load /
classification groups with negatives only — expected).

---

## 8. Stage 10 — train CE-ProbeHYRR

The glue `train_ce_probehyrr.py` reads `utility_pairs_{split}.jsonl`, renames
`label_binary → label` (everything else — `instance_id, dataset, question, skill` — is
already present), builds the dev nDCG pool + corpus from the M4 cache, and calls
`train_cross_encoder()`.

```bash
conda activate sra   # needs torch + transformers (NOT available in the base env)

python -m sragents.probehyrr_h100.train_ce_probehyrr \
  --output-dir results/ce_probehyrr_h100/model_v0 \
  --loss bce --epochs 3          # binary-first (decision D2); uses all pairs
# ranking alternative:
#   --loss listwise              # auto-enables --group-by-query; ~half the groups (no-load /
#                                #   classification) have no positive and contribute 0 to listwise
```

Key facts about the trainer (important — they bit us during review):

- **A dev pool is mandatory.** The trainer saves a checkpoint *only* when a dev metric
  improves (`early_stop_metric=dev_ndcg@10`). With no dev pool it never saves. The glue always
  builds one from `m4_top50_dev.jsonl` (re-ranks the top-50, all 282 dev skill_ids resolve in
  the corpus — verified).
- Loss options: `bce` (default), `pairwise`, `listwise`. Eval is always retrieval nDCG, so you
  can train with BCE and still early-stop on nDCG@10.
- Class balance: ~20% positive (1547/7803 train pairs) — expected for hard-negative ranking.
- Output: `model_v0/` with `train_config.json`, `train_summary.json`, and the best checkpoint.

> The glue intentionally does **not** use `sragents.corpus.load_corpus` (it `json.loads` a JSON
> array; the pipeline's canonical corpus is JSONL) — it loads the corpus line-by-line itself.

---

## 9. Known deviations from the spec (intentional / cosmetic)

- Task rows store `(system, user)` + `prompt_hash` (chat template applied at gen time), not the
  spec's pre-rendered `prompt_text`. Enables GPU-free Stage 0/1 and vLLM prefix caching.
- Reports are `quality_report_{split}.json`, not the spec's `dataset_quality_report.md` /
  `ucb_policy_report.md`.
- No `run_pipeline.py`; `run_probehyrr_full.sh` orchestrates.
- qids are 5-digit with full dataset name (`medcalcbench_00007`), not the spec's `medcalc_0007`
  (decision D4 — required for joins).
- `ce_top1` / CE-HYRR-v0 fully removed (Option A cold-start, decision D1).

---

## 10. Quick reference — single-stage reruns (CPU, no regen needed)

If you only changed CPU logic (labels/groups), you do **not** need the GPU stages:

```bash
# rebuild labels → pairs → groups → report for one split, from existing verified/logs:
SPLIT=train
python -m sragents.probehyrr_h100.build_utility_pairs \
  --anchors results/ce_probehyrr_h100/verified/anchor_verified_${SPLIT}.jsonl \
  --probe-logs results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_${SPLIT}.jsonl \
  --m4 results/m4/m4_top50_${SPLIT}.jsonl --skills data/corpus/skills.jsonl \
  --out results/ce_probehyrr_h100/utility_pairs_${SPLIT}.jsonl
python -m sragents.probehyrr_h100.build_training_groups \
  --pairs results/ce_probehyrr_h100/utility_pairs_${SPLIT}.jsonl \
  --regimes results/ce_probehyrr_h100/query_regime_labels_${SPLIT}.jsonl \
  --out results/ce_probehyrr_h100/utility_train_groups_${SPLIT}.jsonl
python -m sragents.probehyrr_h100.report_quality --split ${SPLIT}
```

`rebuild_smoke.sh` is a tiny end-to-end smoke (Stages 8–9 on sample data) for sanity checks.
```
