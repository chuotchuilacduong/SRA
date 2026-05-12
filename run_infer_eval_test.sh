#!/usr/bin/env bash
# Stage 2 + 3: infer (progressive_disclosure) then evaluate.
#
# Backend selection (mutually exclusive):
#   Timely API  — set TIMELY_API_KEY in .env (default, no API_BASE needed)
#   Local model — set API_BASE to your vLLM/Ollama/LM Studio endpoint
#                 e.g. API_BASE=http://localhost:8000/v1
#
# Supported models (examples):
#   Timely   : gpt-4o-mini  gpt-4o  gpt-4.1  gpt-4.1-mini  o4-mini
#   Local    : Qwen/Qwen3-8B-Instruct  meta-llama/Llama-3.1-8B-Instruct
#              mistralai/Mistral-7B-Instruct-v0.3  (any OpenAI-compat model)
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_infer_eval_test.sh                            # theoremqa, gpt-4o-mini
#   bash run_infer_eval_test.sh champ logicbench           # multiple datasets
#   MODEL=gpt-4o SUBSET_SIZE=500 bash run_infer_eval_test.sh
#   API_BASE=http://localhost:8000/v1 MODEL=Qwen/Qwen3-8B-Instruct bash run_infer_eval_test.sh

set -euo pipefail

INSTANCES_DIR="data/bench/instances"
RETRIEVAL_DIR="results/retrieval_test"
INFER_DIR="results/infer_test"
EVAL_DIR="results/eval_test"
MODEL="${MODEL:-Qwen/Qwen3-8B-Instruct}"
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
PROVIDER_K="${PROVIDER_K:-5}"
WORKERS="${WORKERS:-8}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"   # local vLLM; set empty for Timely

mkdir -p "$INFER_DIR" "$EVAL_DIR"

DATASETS=("${@:-theoremqa}")

for DS in "${DATASETS[@]}"; do
    INSTANCES="$INSTANCES_DIR/${DS}.json"
    RETRIEVAL="$RETRIEVAL_DIR/${DS}-linearrag-${SUBSET_SIZE}.json"
    INFER_OUT="$INFER_DIR/${DS}-linearrag-${SUBSET_SIZE}.jsonl"
    EVAL_OUT="$EVAL_DIR/${DS}-linearrag-${SUBSET_SIZE}.json"

    if [ ! -f "$INSTANCES" ]; then
        echo "[SKIP] $DS: instances not found at $INSTANCES"
        continue
    fi

    if [ ! -f "$RETRIEVAL" ]; then
        echo "[SKIP] $DS: retrieval output not found at $RETRIEVAL (run stage 1 first)"
        continue
    fi

    # --- Stage 2: infer ---
    if [ -f "$INFER_OUT" ]; then
        echo "[SKIP infer] $DS: $INFER_OUT already exists"
    else
        echo ""
        echo "=== Stage 2 — Infer: $DS (model=$MODEL, k=$PROVIDER_K) ==="
        echo "    Retrieval : $RETRIEVAL"
        echo "    Output    : $INFER_OUT"
        echo "    Started   : $(date '+%H:%M:%S')"

        API_BASE_ARG=""
        if [ -n "$API_BASE" ]; then
            API_BASE_ARG="--api-base $API_BASE"
        fi

        sragents infer \
            --provider topk \
            --provider-arg source="$RETRIEVAL" \
            --provider-arg k="$PROVIDER_K" \
            --engine progressive_disclosure \
            --instances "$INSTANCES" \
            --model "$MODEL" \
            --workers "$WORKERS" \
            --output "$INFER_OUT" \
            $API_BASE_ARG

        echo "    Done      : $(date '+%H:%M:%S')"
    fi

    # --- Stage 3: evaluate ---
    if [ -f "$EVAL_OUT" ]; then
        echo "[SKIP eval]  $DS: $EVAL_OUT already exists (use --force to rerun)"
    else
        echo ""
        echo "=== Stage 3 — Evaluate: $DS ==="
        echo "    Input  : $INFER_OUT"
        echo "    Output : $EVAL_OUT"

        sragents evaluate \
            --input "$INFER_OUT" \
            --instances "$INSTANCES" \
            --output "$EVAL_OUT"
    fi
done

echo ""
echo "=== Done. Results ==="
for DS in "${DATASETS[@]}"; do
    EVAL_OUT="$EVAL_DIR/${DS}-linearrag-${SUBSET_SIZE}.json"
    if [ -f "$EVAL_OUT" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$EVAL_OUT')); print(f\"{d['metrics']['correct']}/{d['metrics']['total']} ({d['metrics']['accuracy']:.4f})\")" 2>/dev/null || echo "?")
        echo "  $DS: $ACC"
    fi
done
