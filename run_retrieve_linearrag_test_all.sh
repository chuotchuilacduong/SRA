#!/usr/bin/env bash
# Run LinearRAG retrieval on ALL datasets in data/bench/instances
# Uses SMALL corpus subset (1k skills by default) - optimized for limited RAM
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_retrieve_linearrag_test_all.sh                          # All 6 datasets, 1k corpus
#   bash run_retrieve_linearrag_test_all.sh champ theoremqa          # Subset of datasets
#   SUBSET_SIZE=500 MAX_INSTANCES=500 bash run_retrieve_linearrag_test_all.sh

set -euo pipefail

# Enable Metal GPU acceleration on macOS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

CORPUS_FULL="data/bench/corpus/corpus.json"
INSTANCES_DIR="data/bench/instances"
OUTPUT_DIR="results/retrieval_test"
TOP_K=50

# Configurable parameters
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
MAX_INSTANCES="${MAX_INSTANCES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_WORKERS="${MAX_WORKERS:-1}"  # macOS: use 1 to avoid semaphore leaks
MAX_CHARS="${MAX_CHARS:-4000}"

CORPUS_SMALL="data/bench/corpus/corpus_${SUBSET_SIZE}.json"
DATASET_NAME="bench_${SUBSET_SIZE}"

mkdir -p "$OUTPUT_DIR"

# Create the subset corpus if it doesn't exist
if [ ! -f "$CORPUS_SMALL" ]; then
    echo "Creating ${SUBSET_SIZE}-skill subset corpus from $CORPUS_FULL ..."
    python3 - <<PYEOF
import json, pathlib, sys

src = pathlib.Path("$CORPUS_FULL")
if not src.exists():
    sys.exit(f"ERROR: corpus not found at {src}")

corpus = json.loads(src.read_text())
subset = corpus[:$SUBSET_SIZE]
pathlib.Path("$CORPUS_SMALL").write_text(json.dumps(subset, ensure_ascii=False))
print(f"Wrote {len(subset)} skills to $CORPUS_SMALL")
PYEOF
else
    echo "Reusing existing subset corpus: $CORPUS_SMALL (${SUBSET_SIZE} skills)"
fi

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
echo "LinearRAG Retrieval Test - All Datasets"
echo "=========================================="
echo "Corpus size  : ${SUBSET_SIZE} skills"
echo "Corpus file  : $CORPUS_SMALL"
echo "Output dir   : $OUTPUT_DIR"
echo "Batch size   : $BATCH_SIZE"
echo "Max workers  : $MAX_WORKERS"
echo "Max chars    : $MAX_CHARS"
echo "Max instances: $MAX_INSTANCES"
echo "Datasets     : ${DATASETS[@]}"
echo "=========================================="
echo ""

PROCESSED=0
SKIPPED=0
FAILED=0

for DS in "${DATASETS[@]}"; do
    INSTANCES="$INSTANCES_DIR/${DS}.json"
    OUTPUT="$OUTPUT_DIR/${DS}-linearrag-${SUBSET_SIZE}.json"

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
    echo "=== Dataset: $DS (test, ${SUBSET_SIZE}-skill corpus) ==="
    echo "    Instances : $INSTANCES"
    echo "    Output    : $OUTPUT"
    echo "    Started   : $(date '+%H:%M:%S')"

    if sragents retrieve \
        --retriever linearrag \
        --retriever-arg dataset_name="$DATASET_NAME" \
        --retriever-arg max_chars_per_passage="$MAX_CHARS" \
        --retriever-arg batch_size="$BATCH_SIZE" \
        --retriever-arg max_workers="$MAX_WORKERS" \
        --corpus "$CORPUS_SMALL" \
        --instances "$INSTANCES" \
        --output "$OUTPUT" \
        --top-k $TOP_K \
        --max-instances $MAX_INSTANCES; then
        echo "    Done      : $(date '+%H:%M:%S')"
        ((PROCESSED++))

        # Optional: copy to results/retrieval/ for consistency
        FULL_DIR="results/retrieval"
        mkdir -p "$FULL_DIR"
        cp "$OUTPUT" "$FULL_DIR/${DS}-linearrag.json"
        echo "    Copied    : $FULL_DIR/${DS}-linearrag.json"
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
    ls -lh "$OUTPUT_DIR"/*linearrag-${SUBSET_SIZE}.json 2>/dev/null | tail -$PROCESSED || true
fi
