#!/bin/bash
# ===========================================================================
# rebuild_smoke.sh — recreate the /tmp smoke-test inputs for the CE-ProbeHYRR
# Stages 4-9 pipeline. Recovered from session 05fa1f80-e3d6-43b8-b7fc-7121461bc87b
# (the /ultraplan session) on 2026-06-15 after the original /tmp files were lost.
#
# These are 12-query (--limit-queries 12, budget B=3) smoke artifacts derived
# from the REAL on-disk anchor + M4 inputs, so they rebuild deterministically.
#
#   Stage 4 (CPU)  query_regimes        -> /tmp/regimes_train_smoke.jsonl
#   Stage 5 (CPU)  build_ucb_tasks      -> /tmp/ucb_tasks_r1_smoke.jsonl  (+ ucb_state_train_smoke.json)
#   Stage 6 (GPU)  run_batched_generation -> /tmp/ucb_gen_smoke.jsonl   <-- needs the H100
#   Stage 7a(CPU)  verify_outputs       -> /tmp/ucb_ver_smoke.jsonl
#   Stage 7b(CPU)  label_builder        -> /tmp/ucb_log_smoke.jsonl (+ ucb_state2_smoke.json)
#   Stage 8 (CPU)  build_utility_pairs  -> /tmp/utility_pairs_smoke.jsonl
#   Stage 9 (CPU)  build_training_groups-> /tmp/groups_smoke.jsonl
#
# GPU CONTENTION: Stage 6 launches vLLM. Do NOT run it while the full pipeline
# (run_probehyrr_full.sh, e.g. slurm job 20190) is using the H100 — they will
# fight over GPU memory. The CPU-only stages (4,5) are safe to run any time.
# Use CPU_ONLY=1 to stop before the GPU stage.
# ===========================================================================
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-sra}"
cd "$HOME/projects/SRA"

ANCH=results/ce_probehyrr_h100/verified/anchor_verified_train.jsonl
M4=results/m4/m4_top50_train.jsonl
LIMIT="${LIMIT_QUERIES:-12}"
CPU_ONLY="${CPU_ONLY:-0}"

for f in "$ANCH" "$M4"; do
  [ -s "$f" ] || { echo "ERROR: missing source input $f — cannot rebuild smoke files"; exit 1; }
done

echo "=== Stage 4 (CPU): query regimes ==="
python -m sragents.probehyrr_h100.query_regimes \
  --anchors "$ANCH" --m4 "$M4" \
  --out /tmp/regimes_train_smoke.jsonl 2>&1 | tail -5

echo "=== Stage 5 (CPU): build UCB round-1 tasks (B=3, limit=$LIMIT) ==="
rm -f /tmp/ucb_state_train_smoke.json
python -m sragents.probehyrr_h100.build_ucb_tasks \
  --round 1 --budget-per-query 3 \
  --m4 "$M4" --anchors "$ANCH" \
  --regimes /tmp/regimes_train_smoke.jsonl \
  --ucb-state /tmp/ucb_state_train_smoke.json \
  --out /tmp/ucb_tasks_r1_smoke.jsonl \
  --limit-queries "$LIMIT" 2>&1 | tail -4

if [ "$CPU_ONLY" = "1" ]; then
  echo "CPU_ONLY=1 set — stopping before GPU Stage 6. CPU smoke inputs rebuilt."
  exit 0
fi

# --- Guard: refuse to start vLLM if a GPU is already busy (full run in flight) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$USED" ] && [ "$USED" -gt 2000 ]; then
    echo "ERROR: GPU already using ${USED} MiB — the full pipeline is probably running."
    echo "       Refusing to launch a second vLLM. Re-run after it finishes, set CPU_ONLY=1,"
    echo "       or run on a different GPU node."
    exit 1
  fi
fi

echo "=== Stage 6 (GPU): generate 12 UCB candidates ==="
python -m sragents.probehyrr_h100.run_batched_generation \
  --tasks /tmp/ucb_tasks_r1_smoke.jsonl --out /tmp/ucb_gen_smoke.jsonl \
  --model Qwen/Qwen3-8B --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 16384 --enable-prefix-caching 2>&1 | tail -3

echo "=== Stage 7a (CPU): verify ==="
python -m sragents.probehyrr_h100.verify_outputs \
  --generations /tmp/ucb_gen_smoke.jsonl --out /tmp/ucb_ver_smoke.jsonl --workers 8 2>&1 | tail -2

echo "=== Stage 7b (CPU): label + UCB arm update ==="
rm -f /tmp/ucb_log_smoke.jsonl /tmp/ucb_state2_smoke.json
python -m sragents.probehyrr_h100.label_builder \
  --verified /tmp/ucb_ver_smoke.jsonl --anchors "$ANCH" --m4 "$M4" \
  --append-log /tmp/ucb_log_smoke.jsonl --update-ucb-state /tmp/ucb_state2_smoke.json --round 1 2>&1 | tail -2

echo "=== Stage 8 (CPU): utility pairs ==="
python -m sragents.probehyrr_h100.build_utility_pairs \
  --anchors "$ANCH" --probe-logs /tmp/ucb_log_smoke.jsonl --m4 "$M4" \
  --out /tmp/utility_pairs_smoke.jsonl 2>&1 | tail -2

echo "=== Stage 9 (CPU, FINAL): training groups ==="
python -m sragents.probehyrr_h100.build_training_groups \
  --pairs /tmp/utility_pairs_smoke.jsonl --regimes /tmp/regimes_train_smoke.jsonl \
  --out /tmp/groups_smoke.jsonl 2>&1 | tail -3

echo "Done — smoke inputs rebuilt under /tmp/."
