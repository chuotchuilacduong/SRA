#!/bin/bash
#SBATCH --job-name=probehyrr_full          # Job name
#SBATCH --output=log_probehyrr_full_%j.out # Stdout log file
#SBATCH --error=log_probehyrr_full_%j.err  # Stderr log file
#SBATCH --time=12:00:00                     # Max runtime (HH:MM:SS); set 0 for unlimited
#SBATCH --partition=main                    # Partition/queue
#SBATCH --ntasks=1                          # Single process
#SBATCH --cpus-per-task=8                   # CPU threads (verifier workers)
#SBATCH --mem=64G                           # Memory (Qwen3-8B vLLM)
#SBATCH --gres=gpu:1                        # Request 1 H100
#
# ===========================================================================
# FULL-SCALE end-to-end build for CE-ProbeHYRR (Stages 0-9, no --limit-queries).
#
#   Stage 0 (CPU)  build_splits / build_m4_cache / build_anchor_tasks
#   Stage 2 (GPU)  run_batched_generation  -> anchor_outputs_{split}.jsonl
#   Stage 3 (CPU)  verify_outputs          -> anchor_verified_{split}.jsonl
#   Stage 4 (CPU)  query_regimes           -> query_regime_labels_{split}.jsonl
#   Stage 5 (CPU)  build_ucb_tasks         -> ucb_probe_tasks_round_{r}_{split}.jsonl
#   Stage 6 (GPU)  run_batched_generation  -> ucb_outputs_round_{r}_{split}.jsonl
#   Stage 7 (CPU)  verify_outputs + label_builder -> ucb_candidate_probe_logs_{split}.jsonl
#   Stage 8 (CPU)  build_utility_pairs     -> utility_pairs_{split}.jsonl
#   Stage 9 (CPU)  build_training_groups   -> utility_train_groups_{split}.jsonl  (FINAL)
#
# Stages 5-7 loop NUM_ROUNDS times (micro-batch UCB). Anchors for all splits are
# built first, then Stages 4-9 run per split.
#
# Env stack pinned for this cluster's CUDA 12.8 driver:
#   torch 2.8.0+cu128 · vllm 0.10.2 · transformers 4.56.1 (conda env `sra`)
#
# Run interactively on a GPU node:   bash run_probehyrr_full.sh
# Or submit as a batch job:          sbatch run_probehyrr_full.sh
# Resume after a crash: just re-run — outputs are append-only and skip cached
# tasks. Use FORCE=1 to ignore caches and regenerate from scratch.
#
# Overridable via env, e.g.:
#   SPLITS="train" bash run_probehyrr_full.sh          # one split only
#   GPU_MEM_UTIL=0.95 MAX_MODEL_LEN=16384 bash run_probehyrr_full.sh
# ===========================================================================

# NOTE: no `set -u` — nounset breaks `conda activate` in non-interactive shells.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- config (override via env) -------------------------------------------
SPLITS="${SPLITS:-train dev test}"
DATASETS="${DATASETS:-theoremqa logicbench medcalcbench champ}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
ENV_NAME="${ENV_NAME:-sra}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"  # longest anchor prompt is 9337 tok; 8192 overflows on the xlong bucket
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
SKIP_PREP="${SKIP_PREP:-0}"     # 1 = skip Stage 0 (inputs already built)
FORCE="${FORCE:-0}"             # 1 = ignore caches / no resume

RUN="results/ce_probehyrr_h100"
FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"

echo "Start at: $(date)"
echo "Running on node: $(hostname)"
echo "Splits: ${SPLITS} | Datasets: ${DATASETS} | Env: ${ENV_NAME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || { echo "ERROR: no GPU visible — are you on a GPU node (srun --gres=gpu:1)?"; exit 1; }

conda activate "$ENV_NAME"

# Real CUDA check: nvidia-smi can succeed on a login node while the GPU is not
# usable by this process, which makes vLLM die later with the cryptic
# "Device string must not be empty". Fail fast here instead.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "ERROR: torch.cuda.is_available()==False — not on an allocated GPU node. Use: srun --partition=main --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 --pty bash"; exit 1; }

# --- Stage 0+1: prepare inputs + anchor manifests (CPU, idempotent) ------
if [ "$SKIP_PREP" != "1" ]; then
  echo "=== Stage 0: build splits / corpus ==="
  python -m sragents.probehyrr_h100.build_splits   --datasets $DATASETS
  echo "=== Stage 0: build m4 top-50 cache ==="
  python -m sragents.probehyrr_h100.build_m4_cache --splits   $SPLITS
  echo "=== Stage 1: build anchor task manifests ==="
  python -m sragents.probehyrr_h100.build_anchor_tasks --splits $SPLITS
else
  echo "=== Stage 0/1 skipped (SKIP_PREP=1) ==="
fi

# --- Stage 2+3 per split: FULL anchor generation + verification ----------
for SPLIT in $SPLITS; do
  TASKS="$RUN/tasks/anchor_probe_tasks_${SPLIT}.jsonl"
  GEN="$RUN/generations/anchor_outputs_${SPLIT}.jsonl"
  VER="$RUN/verified/anchor_verified_${SPLIT}.jsonl"

  if [ ! -f "$TASKS" ]; then
    echo "WARN: missing tasks for split=${SPLIT} ($TASKS) — skipping"; continue
  fi

  echo "================================================================"
  echo "=== [$SPLIT] Stage 2: batched anchor generation (FULL, GPU) ==="
  echo "================================================================"
  python -m sragents.probehyrr_h100.run_batched_generation \
    --tasks "$TASKS" \
    --out   "$GEN" \
    --model "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-prefix-caching \
    $FORCE_FLAG

  echo "=== [$SPLIT] Stage 3: verify + gate summary (CPU) ==="
  python -m sragents.probehyrr_h100.verify_outputs \
    --generations "$GEN" \
    --out "$VER" \
    --workers "$WORKERS" \
    --summary \
    $FORCE_FLAG
done

echo "Done (anchors) at: $(date) | splits: ${SPLITS}"

# --- Stages 4-9 per split: regimes + micro-batch UCB rounds + utility dataset
NUM_ROUNDS="${NUM_ROUNDS:-3}"
BUDGET="${BUDGET:-3}"
LIMIT_FLAG=""
[ -n "${LIMIT_QUERIES:-}" ] && LIMIT_FLAG="--limit-queries ${LIMIT_QUERIES}"

for SPLIT in $SPLITS; do
  VER="$RUN/verified/anchor_verified_${SPLIT}.jsonl"
  REG="$RUN/query_regime_labels_${SPLIT}.jsonl"
  LOG="$RUN/logs/ucb_candidate_probe_logs_${SPLIT}.jsonl"
  STATE="$RUN/state/ucb_state_${SPLIT}.json"
  M4="results/m4/m4_top50_${SPLIT}.jsonl"

  if [ ! -f "$VER" ]; then
    echo "WARN: missing anchor verifications for split=${SPLIT} ($VER) — skipping Stages 4-9"; continue
  fi

  echo "================================================================"
  echo "=== [$SPLIT] Stage 4: derive query regimes (CPU) ==="
  echo "================================================================"
  python -m sragents.probehyrr_h100.query_regimes --anchors "$VER" --m4 "$M4" --out "$REG"

  # FORCE: clear the cumulative log + UCB state so rounds rebuild cleanly.
  if [ "$FORCE" = "1" ]; then rm -f "$LOG" "$STATE"; fi

  for ROUND in $(seq 1 "$NUM_ROUNDS"); do
    UTASKS="$RUN/tasks/ucb_probe_tasks_round_${ROUND}_${SPLIT}.jsonl"
    UGEN="$RUN/generations/ucb_outputs_round_${ROUND}_${SPLIT}.jsonl"
    UVER="$RUN/verified/ucb_verified_round_${ROUND}_${SPLIT}.jsonl"

    echo "=== [$SPLIT] Stage 5 (round $ROUND): build UCB probe tasks (CPU) ==="
    python -m sragents.probehyrr_h100.build_ucb_tasks \
      --round "$ROUND" --budget-per-query "$BUDGET" \
      --m4 "$M4" --anchors "$VER" --regimes "$REG" \
      --previous-logs "$LOG" --ucb-state "$STATE" \
      --out "$UTASKS" $LIMIT_FLAG $FORCE_FLAG

    if [ ! -s "$UTASKS" ]; then
      echo "  [round $ROUND] no UCB tasks scheduled (budget/candidates exhausted) — skipping"; continue
    fi

    echo "=== [$SPLIT] Stage 6 (round $ROUND): batched candidate generation (GPU) ==="
    python -m sragents.probehyrr_h100.run_batched_generation \
      --tasks "$UTASKS" --out "$UGEN" \
      --model "$MODEL" --dtype bfloat16 \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --enable-prefix-caching $FORCE_FLAG

    echo "=== [$SPLIT] Stage 7a (round $ROUND): verify candidates (CPU) ==="
    python -m sragents.probehyrr_h100.verify_outputs \
      --generations "$UGEN" --out "$UVER" --workers "$WORKERS" $FORCE_FLAG

    echo "=== [$SPLIT] Stage 7b (round $ROUND): label + UCB arm update (CPU) ==="
    python -m sragents.probehyrr_h100.label_builder \
      --verified "$UVER" --anchors "$VER" --m4 "$M4" \
      --append-log "$LOG" --update-ucb-state "$STATE" --round "$ROUND"
  done

  echo "=== [$SPLIT] Stage 8: build utility pairs (CPU) ==="
  python -m sragents.probehyrr_h100.build_utility_pairs \
    --anchors "$VER" --probe-logs "$LOG" --m4 "$M4" \
    --skills data/corpus/skills.jsonl \
    --out "$RUN/utility_pairs_${SPLIT}.jsonl"

  echo "=== [$SPLIT] Stage 9: build training groups (CPU, FINAL) ==="
  python -m sragents.probehyrr_h100.build_training_groups \
    --pairs "$RUN/utility_pairs_${SPLIT}.jsonl" --regimes "$REG" \
    --out "$RUN/utility_train_groups_${SPLIT}.jsonl"

  echo "=== [$SPLIT] quality report ==="
  python -m sragents.probehyrr_h100.report_quality --split "$SPLIT" || true
done

echo "Done (full pipeline, Stages 0-9) at: $(date)"
echo "FINAL training dataset:"
echo "  -> $RUN/utility_train_groups_{${SPLITS// /,}}.jsonl"
