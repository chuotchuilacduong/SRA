#!/usr/bin/env bash
# End-task accuracy pipeline (Stage B -> E) for ONE model, on H100.
# Runs: make test-instances -> serve vLLM (+wait) -> ToolQA probe -> `endtask`
# experiment (all 23 cells, infer+evaluate) -> aggregate the 2 tables.
#
# IMPORTANT: vLLM needs a GPU. On a SLURM cluster, run this INSIDE a GPU
# allocation, e.g.:  srun --gres=gpu:2 --pty bash  then run this script.
# (login nodes have no GPU.) Qwen3-32B needs tensor-parallel >= 2.
#
# Usage (from repo root or anywhere):
#   bash src/kmeans/scripts/analysis/run_endtask_h100.sh Qwen/Qwen3-4B  8000 1
#   bash src/kmeans/scripts/analysis/run_endtask_h100.sh Qwen/Qwen3-32B 8001 2
#   bash src/kmeans/scripts/analysis/run_endtask_h100.sh --aggregate          # just build tables
#
# Env knobs (defaults): MAXLEN=32768 WORKERS=32 EVAL_WORKERS=8
#   READY_TIMEOUT=1200 KEEP_SERVER=0 SKIP_PROBE=0
#   NO_SERVE=0   # set 1 if you start vLLM yourself (script then just waits on $PORT)
set -uo pipefail

cd "$(dirname "$0")/../../../.."        # src/kmeans/scripts/analysis -> repo root
export PYTHONPATH=src
PY="python -m sragents.cli.main"
DATASETS="theoremqa logicbench toolqa champ medcalcbench bigcodebench"
SOURCES="bm25 l6_final ceraw_bge_base ceraw_rrf ceraw_bge_ft bge_ft_retriever l6_union_bgeft_k100"
MAXLEN=${MAXLEN:-32768}; WORKERS=${WORKERS:-32}; EVAL_WORKERS=${EVAL_WORKERS:-8}
READY_TIMEOUT=${READY_TIMEOUT:-1200}; KEEP_SERVER=${KEEP_SERVER:-0}
SKIP_PROBE=${SKIP_PROBE:-0}; NO_SERVE=${NO_SERVE:-0}

aggregate() { python src/kmeans/scripts/analysis/aggregate_endtask_tables.py --models Qwen3-4B Qwen3-32B; }

if [ "${1:-}" = "--aggregate" ]; then aggregate; exit 0; fi

MODEL=${1:?usage: $0 <model> <port> <tp>  |  --aggregate}
PORT=${2:?need port}
TP=${3:-1}
SHORT=$(basename "$MODEL")
BASE="http://localhost:$PORT/v1"

echo "==[Stage B] test-only instances (expect TOTAL 1079) =="
python src/kmeans/scripts/analysis/make_test_instances.py

echo "==[pre-flight] retrieval sources present? =="
miss=0
for ds in $DATASETS; do for s in $SOURCES; do
  [ -e "results/retrieval/$ds-$s.json" ] || { echo "  MISSING results/retrieval/$ds-$s.json"; miss=1; }
done; done
[ $miss -eq 0 ] && echo "  all 42 sources present (7 sources × 6 datasets)" \
  || echo "  [warn] missing above -> those cells skipped. Run Stage A first (run_fair_eval.py, ceraw_eval.py, add_bgeft_features.py, run_l6_union_bgeft.py, bm25 symlinks)."

VLLM_PID=""
cleanup() { if [ -n "$VLLM_PID" ] && [ "$KEEP_SERVER" != "1" ]; then
  echo "stopping vLLM (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; pkill -f "vllm serve $MODEL" 2>/dev/null; fi; }
trap cleanup EXIT

if [ "$NO_SERVE" != "1" ]; then
  echo "==[Stage C] serve $MODEL on :$PORT (tp=$TP, max-len=$MAXLEN) =="
  mkdir -p logs
  LOG="logs/vllm_${SHORT}_${PORT}.log"
  vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
      --max-model-len "$MAXLEN" --gpu-memory-utilization 0.90 --dtype bfloat16 \
      --served-model-name "$MODEL" > "$LOG" 2>&1 &
  VLLM_PID=$!
  echo "  vLLM pid=$VLLM_PID, log=$LOG; waiting up to ${READY_TIMEOUT}s for readiness..."
  ready=0
  for _ in $(seq 1 "$READY_TIMEOUT"); do
    if curl -sf "$BASE/models" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "  vLLM exited early — tail of $LOG:"; tail -n 25 "$LOG"; exit 1; fi
    sleep 1
  done
  [ $ready -eq 1 ] || { echo "  not ready in ${READY_TIMEOUT}s — tail of $LOG:"; tail -n 25 "$LOG"; exit 1; }
  echo "  server ready."
else
  echo "==[Stage C] NO_SERVE=1 — assuming vLLM already at $BASE =="
  curl -sf "$BASE/models" >/dev/null 2>&1 || { echo "  ERROR: no server at $BASE"; exit 1; }
fi

if [ "$SKIP_PROBE" != "1" ] && [ -e "results/retrieval/toolqa-bm25.json" ]; then
  echo "==[probe] ToolQA sanity (3 transcripts) =="
  $PY infer --instances data/bench/instances_test/toolqa.json --output "/tmp/tq_${SHORT}.jsonl" \
    --model "$MODEL" --api-base "$BASE" --provider topk \
    --provider-arg source=results/retrieval/toolqa-bm25.json --provider-arg k=1 \
    --engine react --workers 4 --label probe --force || true
  python - "$SHORT" <<'PY'
import json, sys
p = f"/tmp/tq_{sys.argv[1]}.jsonl"
try:
    for i, l in enumerate(open(p)):
        if i >= 3: break
        r = json.loads(l); t = r.get("transcript") or r.get("raw_output") or ""
        print("---", r["instance_id"]); print(t[:500])
except FileNotFoundError:
    print("  (no probe output produced)")
PY
  echo "  ^ Observation phải là output tool thật (KHÔNG 'is not in the list' / 'Error executing'). Ctrl-C nếu sai."
fi

echo "==[Stage D] endtask experiment — $MODEL (23 cells × 6 datasets; resume-safe) =="
$PY experiment --exp endtask --model "$MODEL" --api-base "$BASE" \
  --instances-dir data/bench/instances_test \
  --workers "$WORKERS" --eval-workers "$EVAL_WORKERS" --temperature 0.7 --max-tokens 4096

echo "==[Stage E] aggregate tables =="
aggregate
echo "DONE: $MODEL -> results/comparisons/endtask_${SHORT}.{md,csv} (+ endtask_tables.md)"
