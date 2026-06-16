# H100 + Qwen3-8B Batched Dataset Pipeline for CE-ProbeHYRR

> **Goal**: process SRA-Bench / CE-ProbeHYRR data generation as fast as possible on an H100 using Qwen3-8B with high-throughput batching.
>
> **Design choice**: Option A cold-start. Do **not** train CE-HYRR-v0 first. Do **not** require `ce_top1`. Use M4 top-50 as the candidate pool and train CE-ProbeHYRR directly from counterfactual utility labels.

---

## 0. What this file implements

This file is an implementation spec for Codex. It designs the data-generation runner that creates:

```text
m4_top50_{split}.jsonl
anchor_probe_tasks_{split}.jsonl
anchor_outputs_{split}.jsonl
query_regime_labels_{split}.jsonl
ucb_probe_tasks_{split}.jsonl
ucb_outputs_{split}.jsonl
ucb_candidate_probe_logs_{split}.jsonl
utility_pairs_{split}.jsonl
utility_train_groups_{split}.jsonl
dataset_quality_report.md
ucb_policy_report.md
```

The core pipeline is:

```text
Raw query + 26k skill corpus
  ↓
M4 top-50 cache
  ↓
Batched anchor probes on H100:
  no_skill, gold_skill, m4_top1
  ↓
Verifier + query regime labels
  ↓
Micro-batch UCB candidate scheduling
  ↓
Batched candidate probes on H100
  ↓
Verifier + LabelBuilder
  ↓
utility_pairs + utility_train_groups
  ↓
Train CE-ProbeHYRR
```

Important:

```text
UCB is training-time only.
UCB does not create labels.
UCB only chooses which candidate in M4 top-50 should be probed.
Labels come from Qwen3-8B generation + deterministic verifier.
```

---

## 1. Dataset scope and expected calls

SRA-Bench full benchmark contains:

```text
5,400 test instances
636 gold skills
26,262 total skills
```

Source datasets:

| Dataset | Instances | Use in first build? | Reason |
|---|---:|---|---|
| TheoremQA | 747 | yes | valid single-shot, use longer output cap |
| LogicBench | 760 | yes | valid single-shot |
| MedCalc-Bench | 1,100 | yes | valid single-shot, strong oracle-gap |
| CHAMP | 223 | yes | valid single-shot, small |
| ToolQA | 1,430 | no, phase 2 | needs tool-loop/ReAct harness |
| BigCodeBench | 1,140 | no, phase 2 | needs Linux execution sandbox |

Recommended first build:

```text
valid_4 = TheoremQA + LogicBench + MedCalc-Bench + CHAMP
queries = 747 + 760 + 1100 + 223 = 2830
```

Option A calls per query:

```text
anchor calls:
  no_skill = 1
  gold_skill = 1
  m4_top1 = 1

candidate calls:
  UCB budget B = 3

calls/query = 3 + B = 6
```

Expected generation calls:

```text
valid_4:  2,830 * 6 = 16,980 generations
full_6:   5,400 * 6 = 32,400 generations
```

---

## 2. H100 runtime strategy

### 2.1 Use vLLM offline batch mode by default

Use vLLM offline inference inside Python instead of sending 16k–32k HTTP requests to a server.

Why:

```text
lower request overhead
better control of prompt sorting / bucketing
one Python process owns the GPU
simpler checkpointing and resume
```

Use server mode only if you need multi-process clients or remote job control.

---

### 2.2 Qwen3-8B configuration

Model:

```text
Qwen/Qwen3-8B
```

Recommended default for this dataset generation:

```text
enable_thinking = False
temperature = 0.0
top_p = 1.0
max_model_len = 8192 or 16384
```

Why disable thinking:

```text
faster generation
shorter outputs
less parser noise
more stable JSON / final-answer extraction
```

Qwen3 thinking mode can be tested later as an ablation, but the first CE-ProbeHYRR data build should use non-thinking mode for throughput and label consistency.

Dataset-specific `max_new_tokens`:

| Dataset | max_new_tokens | Notes |
|---|---:|---|
| LogicBench | 128 | MCQ / short answer |
| MedCalc-Bench | 256 | numeric answer + brief calculation |
| CHAMP | 512 | math, but answer usually short |
| TheoremQA | 1024 | use longer cap; previous diagnostics showed truncation risk |
| ToolQA | exclude | needs tool-loop |
| BigCodeBench | exclude | needs execution sandbox |

---

### 2.3 vLLM engine settings

Start with this conservative H100-80GB setup:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    dtype="bfloat16",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enable_prefix_caching=True,
    generation_config="vllm",
)
```

If prompts are long because skill content is large, use:

```python
max_model_len=16384
```

If memory is stable, tune upward:

```text
gpu_memory_utilization: 0.90 → 0.95
max_num_batched_tokens: increase gradually if your vLLM version exposes it
```

Do not tune blindly. Record throughput metrics first:

```text
input_tokens/sec
output_tokens/sec
prompts/sec
GPU memory usage
OOM count
average prompt length
average output length
```

---

## 3. Core optimization: task manifest + batch generator

The most important implementation rule:

```text
Never call the generator inside a for-loop one query at a time.
Always create a task manifest first, then batch-generate many prompts together.
```

A generation task is one row:

```json
{
  "task_id": "anchor::train::logicbench_0001::no_skill",
  "split": "train",
  "qid": "logicbench_0001",
  "dataset": "logicbench",
  "probe_kind": "no_skill",
  "skill_id": null,
  "ucb_arm": null,
  "prompt_text": "...",
  "prompt_hash": "sha256:...",
  "generation_config": {
    "model": "Qwen/Qwen3-8B",
    "enable_thinking": false,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 128
  },
  "metadata": {
    "rank_m4": null,
    "is_gold": false,
    "gold_rank_m4": 7
  }
}
```

---

## 4. Pipeline stages

## Stage 0 — Prepare inputs

Required inputs:

```text
data/splits/train.jsonl
data/splits/dev.jsonl
data/splits/test.jsonl
data/corpus/skills.jsonl
results/m4/m4_top50_{split}.jsonl
```

If M4 cache does not exist, build it first. The H100 runner assumes M4 top-50 already exists.

---

## Stage 1 — Build anchor task manifest

For every query, create these tasks:

```text
1. no_skill
2. gold_skill, if gold exists
3. m4_top1
```

No CE-HYRR anchor in Option A.

Output:

```text
results/ce_probehyrr_h100/tasks/anchor_probe_tasks_{split}.jsonl
```

Example:

```json
{
  "task_id": "anchor::train::medcalc_0007::m4_top1::skill_900",
  "split": "train",
  "qid": "medcalc_0007",
  "dataset": "medcalcbench",
  "probe_kind": "m4_top1",
  "skill_id": "skill_900",
  "rank_m4": 1,
  "prompt_text": "...",
  "prompt_hash": "sha256:...",
  "generation_config": {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "enable_thinking": false
  }
}
```

---

## Stage 2 — Batched anchor generation

Input:

```text
anchor_probe_tasks_{split}.jsonl
```

Output:

```text
anchor_outputs_{split}.jsonl
```

Batching rule:

```text
sort tasks by:
  1. dataset
  2. probe_kind
  3. max_new_tokens
  4. input_token_length bucket
```

Why:

```text
same dataset → similar output length
same probe kind → more prefix sharing
similar prompt length → less padding waste
```

Recommended length buckets:

```text
short:  input tokens <= 1024
medium: 1025–2048
long:   2049–4096
xlong:  >4096
```

Pseudocode:

```python
def run_batched_generation(tasks, llm, tokenizer):
    tasks = filter_cached(tasks)
    tasks = annotate_token_lengths(tasks, tokenizer)
    buckets = bucket_tasks(tasks)

    for bucket in buckets:
        prompts = [t["prompt_text"] for t in bucket]
        sampling_params = make_sampling_params(bucket[0])
        outputs = llm.generate(prompts, sampling_params)
        write_outputs(bucket, outputs)
```

---

## Stage 3 — CPU-parallel verifier

Do not verify inside the GPU generation loop.

Instead:

```text
GPU process writes raw generations.
CPU verifier workers read outputs and produce verification records.
```

Output:

```text
anchor_verified_{split}.jsonl
```

Schema:

```json
{
  "task_id": "anchor::train::medcalc_0007::m4_top1::skill_900",
  "qid": "medcalc_0007",
  "dataset": "medcalcbench",
  "probe_kind": "m4_top1",
  "skill_id": "skill_900",
  "raw_output": "...",
  "parsed_answer": "41",
  "verifier_score": 0,
  "parse_failed": false,
  "verifier_type": "numeric_exact"
}
```

Use multiprocessing for CPU verification:

```python
ProcessPoolExecutor(max_workers=os.cpu_count() - 2)
```

---

## Stage 4 — Derive query regimes

Merge:

```text
m4_top50_{split}.jsonl
anchor_verified_{split}.jsonl
```

For each query derive:

```text
v_no
v_gold
v_m4
gold_rank_m4
gold_in_top50
```

Regimes:

```python
no_load_opportunity = (v_no == 1)
harmful_m4_top1 = (v_no == 1 and v_m4 == 0)
need_external = (v_no == 0 and v_gold == 1)
m4_oracle_gap = (v_gold == 1 and v_m4 == 0)
gold_absent_top50 = (gold_rank_m4 is None or gold_rank_m4 > 50)
```

Output:

```text
query_regime_labels_{split}.jsonl
```

---

## Stage 5 — Micro-batch UCB scheduling

### 5.1 Why micro-batch UCB?

Classic UCB is online:

```text
select one arm → probe → observe reward → update → select next arm
```

That is bad for H100 throughput because it serializes generation.

Use **micro-batch UCB** instead:

```text
select many actions using current UCB stats
batch-generate them together
verify all
update UCB stats once per round
repeat
```

This keeps the UCB idea while allowing large H100 batches.

---

### 5.2 UCB rounds

Recommended:

```text
budget_per_query B = 3
num_rounds = 3
one candidate probe per query per round
```

Round logic:

```text
Round 0: anchors already done.
Round 1: schedule one UCB candidate for many queries.
Round 2: update arm stats, schedule next candidate.
Round 3: update arm stats, schedule final candidate.
```

This is much faster than per-query online UCB because each round is one large batched generation job.

---

### 5.3 Arms for Option A

Do not use `ce_top1_anchor`.

Use:

```text
A1: m4_top1_anchor_already_done
A2: rank_2_to_10_non_gold
A3: rank_11_to_50_non_gold
A4: above_gold_boundary
A5: below_gold_boundary
A6: same_cluster_as_gold_non_gold
A7: high_bm25_low_dense
A8: high_dense_low_bm25
A9: no_load_high_rank_risk
A10: gold_absent_high_rank
A11: random_tail_control
```

The M4 top1 anchor is already generated in Stage 2, so UCB should not schedule it again.

---

### 5.4 Build available actions

For each query, build candidate actions only from M4 top-50.

Skip:

```text
candidate already probed
empty skill content
duplicate skill_id under same qid
gold skill already probed as anchor
m4_top1 already probed as anchor
```

Action schema:

```json
{
  "qid": "medcalc_0007",
  "dataset": "medcalcbench",
  "skill_id": "skill_900",
  "arm": "above_gold_boundary",
  "rank_m4": 12,
  "m4_score": 0.77,
  "bm25_rank": 3,
  "dense_rank": 35,
  "cluster_id": 4,
  "gold_rank_m4": 17,
  "priority_score": 0.85
}
```

---

### 5.5 Candidate priority inside each arm

When UCB selects an arm for a query, choose the best candidate inside that arm:

| Arm | Candidate selection rule |
|---|---|
| `rank_2_to_10_non_gold` | lowest `rank_m4` unprobed |
| `rank_11_to_50_non_gold` | closest to gold if gold covered, else lowest rank |
| `above_gold_boundary` | closest rank above gold, e.g. gold 17 → 16, 15 |
| `below_gold_boundary` | closest rank below gold, e.g. gold 17 → 18, 19 |
| `same_cluster_as_gold_non_gold` | same cluster, lowest rank |
| `high_bm25_low_dense` | largest rank conflict: low BM25 rank, high dense rank |
| `high_dense_low_bm25` | largest rank conflict: low dense rank, high BM25 rank |
| `no_load_high_rank_risk` | rank 2–10, lowest rank |
| `gold_absent_high_rank` | rank 1–10, lowest unprobed rank |
| `random_tail_control` | random from rank 21–50, max one per query |

---

### 5.6 UCB score

For each arm:

```python
ucb_score = mean_reward + C * sqrt(log(t) / n_arm)
```

Recommended:

```text
C = sqrt(2)
minimum_warmup_per_arm = 50 globally, not per query
```

Warm-up strategy:

```text
Before true UCB, schedule enough actions so every arm gets about 50 probes globally, if eligible.
Warm-up must still respect max B probes/query.
```

Why global warm-up:

```text
per-query warm-up is too expensive
arms represent global data-acquisition regimes
```

---

### 5.7 Reward

Reward is based on label usefulness:

```text
helpful                         -> 1.0
gold_verified_helpful           -> 1.0
harmful                         -> 1.2
strict_false_friend             -> 1.0
bad_candidate_when_skill_needed -> 0.8
no_load_preferred               -> 0.6
safe_but_unneeded               -> 0.3
weak_false_friend               -> 0.3
neutral                         -> 0.0
ambiguous                       -> -0.5
unverifiable                    -> -1.0
```

Update global and dataset-specific stats:

```text
arm_stats_global.json
arm_stats_by_dataset.json
```

Dataset-specific stats are useful because LogicBench, MedCalc, TheoremQA, and CHAMP have different failure modes.

---

## Stage 6 — Batched candidate generation

Input:

```text
ucb_probe_tasks_round_{r}_{split}.jsonl
```

Output:

```text
ucb_outputs_round_{r}_{split}.jsonl
```

Run the same H100 batched generator from Stage 2.

Recommended generation loop:

```bash
python -m src.sragents.probehyrr_h100.run_batched_generation \
  --tasks results/ce_probehyrr_h100/tasks/ucb_probe_tasks_round_1_train.jsonl \
  --out results/ce_probehyrr_h100/generations/ucb_outputs_round_1_train.jsonl \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --enable-thinking false \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

---

## Stage 7 — Candidate verification and label building

After each UCB round:

```text
raw outputs → verifier → LabelBuilder → reward → UCB update
```

Output:

```text
ucb_candidate_probe_logs_round_{r}_{split}.jsonl
```

After all rounds, concatenate:

```text
ucb_candidate_probe_logs_{split}.jsonl
```

---

## Stage 8 — Build utility pairs

Input:

```text
anchor_verified_{split}.jsonl
ucb_candidate_probe_logs_{split}.jsonl
m4_top50_{split}.jsonl
skills.jsonl
```

Output:

```text
utility_pairs_{split}.jsonl
```

Important:

```text
Include anchor rows as training pairs too:
  gold_skill anchor → gold_verified_helpful if v_gold=1
  m4_top1 anchor → helpful / harmful / false_friend / safe_but_unneeded depending on v_no, v_gold, v_m4
```

---

## Stage 9 — Build grouped ranking data

Input:

```text
utility_pairs_{split}.jsonl
query_regime_labels_{split}.jsonl
```

Output:

```text
utility_train_groups_{split}.jsonl
```

Group types for Option A:

```text
G1: need_external_with_positive
G2: no_load_risk_group
G3: m4_oracle_gap_group
G4: gold_absent_group
G5: classification_only_group
```

No `ce_top1` group is required.

---

## 10. Prompt design for throughput and parsing

### 10.1 Common system prefix

Use the same system prompt for all tasks to maximize prefix sharing:

```text
You are solving benchmark tasks. Follow the provided instruction and, if a skill is provided, use it only when relevant. Return the final answer in the required format. Do not include unnecessary explanation.
```

### 10.2 No-skill prompt

```text
User Query:
{query}

Return only the final answer.
```

### 10.3 Skill prompt

```text
User Query:
{query}

Candidate Skill:
Title: {title}
Description: {description}
Content:
{skill_content}

Use the skill only if it is useful for solving the query.
Return only the final answer.
```

### 10.4 Dataset-specific answer instruction

Append by dataset:

```text
LogicBench: Return only the option letter.
MedCalc-Bench: Return only the numeric answer.
TheoremQA: Return only the final theorem/math answer.
CHAMP: Return only the final numeric/symbolic answer.
```

This reduces parser failures and output length.

---

## 11. Cache design

Every generation must be cacheable.

Cache key:

```python
cache_key = sha256(json.dumps({
    "model": "Qwen/Qwen3-8B",
    "enable_thinking": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": max_new_tokens,
    "prompt_hash": prompt_hash,
    "skill_id": skill_id,
    "qid": qid,
}, sort_keys=True).encode()).hexdigest()
```

Cache files:

```text
cache/generations/{cache_key}.json
cache/verifications/{cache_key}.json
```

Never recompute if cache exists unless:

```text
--force
--model changed
--prompt template changed
--generation config changed
```

---

## 12. Failure recovery

The runner must support resume.

Rules:

```text
write outputs as JSONL append-only
flush every N records
write .done marker per task shard
skip completed cache keys
save UCB state after every round
```

Files:

```text
state/ucb_round_1_state.json
state/ucb_round_2_state.json
state/ucb_round_3_state.json
state/completed_task_ids.txt
```

---

## 13. Recommended module layout

Create:

```text
src/sragents/probehyrr_h100/__init__.py
src/sragents/probehyrr_h100/build_m4_cache.py
src/sragents/probehyrr_h100/build_anchor_tasks.py
src/sragents/probehyrr_h100/prompt_templates.py
src/sragents/probehyrr_h100/run_batched_generation.py
src/sragents/probehyrr_h100/verify_outputs.py
src/sragents/probehyrr_h100/query_regimes.py
src/sragents/probehyrr_h100/microbatch_ucb.py
src/sragents/probehyrr_h100/build_ucb_tasks.py
src/sragents/probehyrr_h100/label_builder.py
src/sragents/probehyrr_h100/build_utility_pairs.py
src/sragents/probehyrr_h100/build_training_groups.py
src/sragents/probehyrr_h100/report_quality.py
src/sragents/probehyrr_h100/run_pipeline.py
```

---

## 14. CLI design

### 14.1 Build anchor tasks

```bash
python -m src.sragents.probehyrr_h100.build_anchor_tasks \
  --split train \
  --m4 results/m4/m4_top50_train.jsonl \
  --queries data/splits/train.jsonl \
  --skills data/corpus/skills.jsonl \
  --out results/ce_probehyrr_h100/tasks/anchor_probe_tasks_train.jsonl \
  --model Qwen/Qwen3-8B \
  --enable-thinking false
```

### 14.2 Run batched anchors

```bash
python -m src.sragents.probehyrr_h100.run_batched_generation \
  --tasks results/ce_probehyrr_h100/tasks/anchor_probe_tasks_train.jsonl \
  --out results/ce_probehyrr_h100/generations/anchor_outputs_train.jsonl \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --enable-prefix-caching true
```

### 14.3 Verify anchors

```bash
python -m src.sragents.probehyrr_h100.verify_outputs \
  --generations results/ce_probehyrr_h100/generations/anchor_outputs_train.jsonl \
  --queries data/splits/train.jsonl \
  --out results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
  --workers 32
```

### 14.4 Derive regimes

```bash
python -m src.sragents.probehyrr_h100.query_regimes \
  --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
  --m4 results/m4/m4_top50_train.jsonl \
  --out results/ce_probehyrr_h100/query_regime_labels_train.jsonl
```

### 14.5 UCB rounds

```bash
for ROUND in 1 2 3; do
  python -m src.sragents.probehyrr_h100.build_ucb_tasks \
    --round $ROUND \
    --budget-per-query 3 \
    --m4 results/m4/m4_top50_train.jsonl \
    --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
    --regimes results/ce_probehyrr_h100/query_regime_labels_train.jsonl \
    --previous-logs results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
    --ucb-state results/ce_probehyrr_h100/state/ucb_state_train.json \
    --out results/ce_probehyrr_h100/tasks/ucb_probe_tasks_round_${ROUND}_train.jsonl

  python -m src.sragents.probehyrr_h100.run_batched_generation \
    --tasks results/ce_probehyrr_h100/tasks/ucb_probe_tasks_round_${ROUND}_train.jsonl \
    --out results/ce_probehyrr_h100/generations/ucb_outputs_round_${ROUND}_train.jsonl \
    --model Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --enable-prefix-caching true

  python -m src.sragents.probehyrr_h100.verify_outputs \
    --generations results/ce_probehyrr_h100/generations/ucb_outputs_round_${ROUND}_train.jsonl \
    --queries data/splits/train.jsonl \
    --out results/ce_probehyrr_h100/verified/ucb_verified_round_${ROUND}_train.jsonl \
    --workers 32

  python -m src.sragents.probehyrr_h100.label_builder \
    --verified results/ce_probehyrr_h100/verified/ucb_verified_round_${ROUND}_train.jsonl \
    --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
    --m4 results/m4/m4_top50_train.jsonl \
    --append-log results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
    --update-ucb-state results/ce_probehyrr_h100/state/ucb_state_train.json
done
```

### 14.6 Build final train files

```bash
python -m src.sragents.probehyrr_h100.build_utility_pairs \
  --anchors results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl \
  --probe-logs results/ce_probehyrr_h100/logs/ucb_candidate_probe_logs_train.jsonl \
  --m4 results/m4/m4_top50_train.jsonl \
  --skills data/corpus/skills.jsonl \
  --out results/ce_probehyrr_h100/utility_pairs_train.jsonl

python -m src.sragents.probehyrr_h100.build_training_groups \
  --pairs results/ce_probehyrr_h100/utility_pairs_train.jsonl \
  --regimes results/ce_probehyrr_h100/query_regime_labels_train.jsonl \
  --out results/ce_probehyrr_h100/utility_train_groups_train.jsonl
```

---

## 15. `run_batched_generation.py` implementation skeleton

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def make_sampling_params(task: dict) -> SamplingParams:
    cfg = task["generation_config"]
    return SamplingParams(
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_new_tokens", 256)),
        stop=cfg.get("stop", None),
    )


def sort_for_throughput(tasks: list[dict], tokenizer) -> list[dict]:
    for t in tasks:
        t["_input_len"] = len(tokenizer.encode(t["prompt_text"]))
    return sorted(
        tasks,
        key=lambda t: (
            t.get("dataset", ""),
            t.get("probe_kind", ""),
            t["generation_config"].get("max_new_tokens", 256),
            t["_input_len"],
        ),
    )


def chunk_by_sampling_params(tasks: list[dict], max_tasks_per_chunk: int = 512):
    chunk = []
    last_key = None
    for t in tasks:
        key = (
            t["generation_config"].get("max_new_tokens", 256),
            t["generation_config"].get("temperature", 0.0),
            t["generation_config"].get("top_p", 1.0),
        )
        if last_key is not None and key != last_key:
            yield chunk
            chunk = []
        chunk.append(t)
        last_key = key
        if len(chunk) >= max_tasks_per_chunk:
            yield chunk
            chunk = []
            last_key = None
    if chunk:
        yield chunk


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        generation_config="vllm",
    )

    tasks = load_jsonl(args.tasks)
    tasks = filter_out_cached(tasks, args.out)  # implement by task_id or cache_key
    tasks = sort_for_throughput(tasks, tokenizer)

    for chunk in chunk_by_sampling_params(tasks, args.max_tasks_per_chunk):
        prompts = [t["prompt_text"] for t in chunk]
        params = make_sampling_params(chunk[0])
        outputs = llm.generate(prompts, params)

        rows = []
        for task, output in zip(chunk, outputs):
            text = output.outputs[0].text
            rows.append({
                "task_id": task["task_id"],
                "qid": task["qid"],
                "dataset": task["dataset"],
                "probe_kind": task["probe_kind"],
                "skill_id": task.get("skill_id"),
                "prompt_hash": task["prompt_hash"],
                "raw_output": text,
                "finish_reason": output.outputs[0].finish_reason,
                "num_output_tokens": len(output.outputs[0].token_ids),
                "generation_config": task["generation_config"],
                "metadata": task.get("metadata", {}),
            })

        write_jsonl(args.out, rows)
```

---

## 16. Expected runtime planning

### 16.1 Calls

```text
valid_4 = 16,980 generations
full_6  = 32,400 generations
```

### 16.2 Estimate by effective prompt throughput

Actual runtime depends on input/output token length, not just number of prompts.

Use this table for rough planning:

| Scope | Generations | 0.5 gen/s | 1 gen/s | 2 gen/s | 4 gen/s |
|---|---:|---:|---:|---:|---:|
| valid_4 | 16,980 | 9.4h | 4.7h | 2.4h | 1.2h |
| full_6 | 32,400 | 18.0h | 9.0h | 4.5h | 2.25h |

Do not trust this table until you measure a pilot of 500–1,000 prompts on the actual H100.

---

## 17. Pilot plan

Before full valid_4:

```text
500 queries
B = 3
approx generations = 500 * 6 = 3,000
```

Run reports:

```text
throughput_report.md
label_summary_pilot.json
ucb_arm_stats_pilot.json
dataset_quality_report_pilot.md
```

Proceed to full valid_4 only if:

```text
parser failure < 5%
verifier failure < 2%
ambiguous labels < 10%
harmful labels present
strict_false_friend labels present
positive helpful/gold labels present
no major OOM/retry issue
```

---

## 18. Design decisions for Codex

Codex should implement these decisions exactly:

```text
1. Option A cold-start only.
2. Remove all required CE-HYRR / ce_top1 dependencies.
3. Use M4 top-50 as fixed candidate pool.
4. Use batched task manifests instead of generator calls inside query loops.
5. Use Qwen/Qwen3-8B with enable_thinking=False by default.
6. Use vLLM offline batched inference on H100.
7. Use micro-batch UCB rounds, not fully online per-query UCB.
8. Write append-only JSONL outputs and cache by prompt hash.
9. Run verifier separately on CPU workers.
10. Build utility_pairs and utility_train_groups after all generation + verification.
```

---

## 19. One-line summary

```text
Implement a high-throughput H100 dataset-generation pipeline for CE-ProbeHYRR using Qwen3-8B, vLLM offline batching, Option A cold-start anchors, micro-batch UCB candidate scheduling, deterministic verification, and JSONL cache/resume outputs.
```
