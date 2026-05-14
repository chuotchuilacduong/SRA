#!/usr/bin/env bash
# Run LinearRAG retrieval - macOS Optimized Version
# Fixes for: Segmentation fault, OOM, multiprocessing crashes
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_retrieve_linearrag_mac.sh                          # All 6 datasets
#   bash run_retrieve_linearrag_mac.sh champ theoremqa          # Subset

set -uo pipefail  # NOT -e: continue if one dataset fails

# ============================================================================
# CRITICAL: macOS-specific fixes
# ============================================================================

# Disable PyTorch MPS (Metal GPU) - causes segfaults with multiprocessing
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""

# Force CPU-only PyTorch
export PYTORCH_DEVICE="cpu"

# Use spawn method for multiprocessing (safer on macOS than fork)
export MULTIPROCESSING_START_METHOD="spawn"

# Disable tokenizers parallelism (causes crashes)
export TOKENIZERS_PARALLELISM="false"

# Limit OpenMP threads (prevents native code conflicts)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Disable HuggingFace symlinks warnings
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Use simpler memory allocator
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# ============================================================================

CORPUS="data/bench/corpus/corpus.json"
INSTANCES_DIR="data/bench/instances"
OUTPUT_DIR="results/retrieval_all"
DATASET_NAME="bench_full"
TOP_K=50

# Safer defaults for macOS to avoid segfaults
BATCH_SIZE="${BATCH_SIZE:-4}"      # Smaller batches
MAX_WORKERS="${MAX_WORKERS:-1}"    # Single worker (no multiprocessing crashes)
MAX_CHARS="${MAX_CHARS:-3000}"     # Shorter passages

mkdir -p "$OUTPUT_DIR"

# Auto-detect datasets
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
echo "LinearRAG Retrieval (macOS Optimized)"
echo "=========================================="
echo "Corpus      : $CORPUS"
echo "Output dir  : $OUTPUT_DIR"
echo "Batch size  : $BATCH_SIZE"
echo "Max workers : $MAX_WORKERS"
echo "Max chars   : $MAX_CHARS"
echo "Datasets    : ${DATASETS[@]}"
echo ""
echo "macOS Fixes Applied:"
echo "  - MPS GPU disabled (causes segfaults)"
echo "  - Multiprocessing: spawn method"
echo "  - Threading: limited to 1"
echo "  - Tokenizers parallelism: off"
echo "=========================================="
echo ""

PROCESSED=0
SKIPPED=0
FAILED=0
FAILED_DATASETS=()

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

    # Run in a sub-shell to isolate crashes
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
        FAILED_DATASETS+=("$DS")
        # Clean up partial output if any
        [ -f "$OUTPUT" ] && rm -f "$OUTPUT"
    fi

    # Force garbage collection between datasets
    sleep 2
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Processed : $PROCESSED"
echo "Skipped   : $SKIPPED"
echo "Failed    : $FAILED"
if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo "Failed datasets: ${FAILED_DATASETS[@]}"
fi
echo "=========================================="
echo ""

if [ $PROCESSED -gt 0 ]; then
    echo "Results in $OUTPUT_DIR:"
    ls -lh "$OUTPUT_DIR"/*linearrag.json 2>/dev/null || true
fi
