#!/bin/bash
#SBATCH --job-name=probehyrr_pilot        # Job name
#SBATCH --output=log_probehyrr_%j.out     # Stdout log file
#SBATCH --error=log_probehyrr_%j.err      # Stderr log file
#SBATCH --time=0                   # Max runtime (HH:MM:SS) — pilot
#SBATCH --partition=main                   # Partition/queue
#SBATCH --ntasks=1                         # Single process
#SBATCH --cpus-per-task=8                  # CPU threads (verifier workers)
#SBATCH --mem=64G                          # Memory (Qwen3-8B vLLM)
#SBATCH --gres=gpu:1                       # Request 1 H100

# ---------------------------------------------------------------------------
# Phase 1 pilot: CE-ProbeHYRR anchor generation + verification on H100.
#   Stage 0 (CPU, sra env): build splits / m4 cache / anchor manifest
#   Stage 2 (GPU, vllm env): vLLM offline batched generation, 500-query pilot
#   Stage 3 (CPU, sra env): deterministic verification + §17 gate summary
#
# One-time env setup (vLLM was installed into the `sra` env directly; the
# thinc/spacy numpy<2 warning is harmless — spacy is not used here):
#   conda activate sra && pip install vllm && pip install -e .   # editable sragents
#   huggingface-cli download Qwen/Qwen3-8B                       # ~16GB bf16
# (To isolate vLLM instead, create a separate env and set VLLM_ENV=that-env.)
# ---------------------------------------------------------------------------

# NOTE: no `set -u` — nounset breaks `conda activate` in non-interactive shells.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- config (override via env: SPLIT=dev PILOT_QUERIES=200 bash run_probehyrr_pilot.sh) ---
SPLIT="${SPLIT:-train}"
PILOT_QUERIES="${PILOT_QUERIES:-500}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
VLLM_ENV="${VLLM_ENV:-sra}"                # env with vllm (installed into sra)
SRA_ENV="${SRA_ENV:-sra}"                  # env with sragents + verifiers (sympy/latex2sympy2)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

RUN="results/ce_probehyrr_h100"
TASKS="$RUN/tasks/anchor_probe_tasks_${SPLIT}.jsonl"
GEN="$RUN/generations/anchor_outputs_${SPLIT}_pilot.jsonl"
VER="$RUN/verified/anchor_verified_${SPLIT}_pilot.jsonl"

echo "Start at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

# --- Stage 0: prepare inputs (CPU, idempotent) ---------------------------
conda activate "$SRA_ENV"
python -m sragents.probehyrr_h100.build_splits --datasets theoremqa logicbench medcalcbench champ
python -m sragents.probehyrr_h100.build_m4_cache --splits "$SPLIT"
python -m sragents.probehyrr_h100.build_anchor_tasks --splits "$SPLIT"

# --- Stage 2: batched anchor generation (GPU) ----------------------------
conda activate "$VLLM_ENV"
python -m sragents.probehyrr_h100.run_batched_generation \
  --tasks "$TASKS" \
  --out "$GEN" \
  --model "$MODEL" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-prefix-caching \
  --limit-queries "$PILOT_QUERIES"

# --- Stage 3: verify + pilot gate summary (CPU) --------------------------
conda activate "$SRA_ENV"
python -m sragents.probehyrr_h100.verify_outputs \
  --generations "$GEN" \
  --out "$VER" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --summary

echo "Done at: $(date)"
echo "Gate check: parser_failure<5%, verifier_failure<2%, harmful+false_friend+helpful present, no OOM."
echo "If GO -> scale to full ${SPLIT} (drop --limit-queries) then run UCB rounds."
