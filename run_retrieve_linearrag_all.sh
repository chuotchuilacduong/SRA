#!/usr/bin/env bash
# Run LinearRAG retrieval on ALL datasets in data/bench/instances
# Uses full corpus (26k skills), processes all 6 datasets
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_retrieve_linearrag_all.sh                          # All 6 datasets
#   bash run_retrieve_linearrag_all.sh champ theoremqa          # Subset of datasets
#   BATCH_SIZE=32 MAX_WORKERS=2 bash run_retrieve_linearrag_all.sh

set -euo pipefail

CORPUS="data/bench/corpus/corpus.json"
INSTANCES_DIR="data/bench/instances"
OUTPUT_DIR="results/retrieval"
DATASET_NAME="bench_full"
TOP_K=50

# Configurable parameters
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_CHARS="${MAX_CHARS:-5000}"

mkdir -p "$OUTPUT_DIR"

# Auto-detect all datasets if no args given
if [ $# -eq 0 ]; then
    echo "Auto-detecting datasets from $INSTANCES_DIR..."
    DATASETS=()
    for file in "$INSTANCES_DIR"/*.json; do
        if [ -f "$file" ]; then
            basename=$(basename "$file" .json)
            DATASETS+=("$basename")
        fi
    done
    echo "Found ${#DATASETS[@]} datasets: ${DATASETS[@]}"
else
    DATASETS=("$@")
fi

echo ""
echo "=========================================="
echo "LinearRAG Retrieval - All Datasets"
echo "=========================================="
echo "Corpus       : $CORPUS"
echo "Output dir  : $OUTPUT_DIR"
echo "Batch size  : $BATCH_SIZE"
echo "Max workers : $MAX_WORKERS"
echo "Max chars   : $MAX_CHARS"
echo "Datasets    : ${DATASETS[@]}"
echo "=========================================="
echo ""

PROCESSED=0
SKIPPED=0
FAILED=0

for DS in "${DATASETS[@]}"; do
    INSTANCES="$INSTANCES_DIR/${DS}.json"
    OUTPUT="$OUTPUT_DIR/${DS}-linearrag.json"

    if [ ! -f "$INSTANCES" ]; then
        echo "[SKIP] $DS: instances file not found at $INSTANCES"
        ((SKIPPED++))
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] $DS: output already exists at $OUTPUT"
        ((SKIPPED++))
        continue
    fi

    echo ""
    echo "=== Dataset: $DS ==="
    echo "    Instances : $INSTANCES"
    echo "    Output    : $OUTPUT"
    echo "    Started   : $(date '+%H:%M:%S')"

    if sragents retrieve \
        --retriever linearrag \
        --retriever-arg dataset_name="$DATASET_NAME" \
        --retriever-arg max_chars_per_passage="$MAX_CHARS" \
        --retriever-arg batch_size="$BATCH_SIZE" \
        --retriever-arg max_workers="$MAX_WORKERS" \
        --corpus "$CORPUS" \
        --instances "$INSTANCES" \
        --output "$OUTPUT" \
        --top-k $TOP_K; then
        echo "    Done      : $(date '+%H:%M:%S')"
        ((PROCESSED++))
    else
        echo "    FAILED    : $(date '+%H:%M:%S')"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Processed : $PROCESSED"
echo "Skipped   : $SKIPPED"
echo "Failed    : $FAILED"
echo "=========================================="
echo ""

if [ $PROCESSED -gt 0 ]; then
    echo "Results in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR"/*linearrag.json 2>/dev/null | tail -$PROCESSED || true
fi
