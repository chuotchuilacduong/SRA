#!/usr/bin/env bash
# Test script: run LinearRAG retrieval on a small corpus subset.
# Avoids OOM on limited-RAM machines.
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_retrieve_linearrag_test.sh                         # theoremqa, 1k corpus + 1k queries
#   bash run_retrieve_linearrag_test.sh champ                   # single dataset
#   SUBSET_SIZE=500 MAX_INSTANCES=500 bash run_retrieve_linearrag_test.sh  # custom size

set -euo pipefail

CORPUS_FULL="data/bench/corpus/corpus.json"
INSTANCES_DIR="data/bench/instances"
OUTPUT_DIR="results/retrieval_test"
TOP_K=50
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
MAX_INSTANCES="${MAX_INSTANCES:-1000}"

CORPUS_SMALL="data/bench/corpus/corpus_${SUBSET_SIZE}.json"
DATASET_NAME="bench_${SUBSET_SIZE}"   # separate NER/embedding cache from bench_full

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
    echo "Reusing existing subset corpus: $CORPUS_SMALL"
fi

# Default to one dataset for quick testing; override via args
DATASETS=("${@:-theoremqa}")

for DS in "${DATASETS[@]}"; do
    INSTANCES="$INSTANCES_DIR/${DS}.json"
    OUTPUT="$OUTPUT_DIR/${DS}-linearrag-${SUBSET_SIZE}.json"

    if [ ! -f "$INSTANCES" ]; then
        echo "[SKIP] $DS: instances file not found at $INSTANCES"
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] $DS: output already exists at $OUTPUT"
        continue
    fi

    echo ""
    echo "=== Retrieving (${SUBSET_SIZE}-skill subset): $DS ==="
    echo "    Corpus  : $CORPUS_SMALL"
    echo "    Output  : $OUTPUT"
    echo "    Started : $(date '+%H:%M:%S')"

    sragents retrieve \
        --retriever linearrag \
        --retriever-arg dataset_name="$DATASET_NAME" \
        --retriever-arg max_chars_per_passage=4000 \
        --retriever-arg batch_size=64 \
        --retriever-arg max_workers=10 \
        --corpus "$CORPUS_SMALL" \
        --instances "$INSTANCES" \
        --output "$OUTPUT" \
        --top-k $TOP_K \
        --max-instances $MAX_INSTANCES

    echo "    Done    : $(date '+%H:%M:%S')"

    # Copy sang results/retrieval/ để experiment runner có thể dùng trực tiếp
    EXPERIMENT_DIR="results/retrieval"
    mkdir -p "$EXPERIMENT_DIR"
    cp "$OUTPUT" "$EXPERIMENT_DIR/${DS}-linearrag.json"
    echo "    Copied  : $EXPERIMENT_DIR/${DS}-linearrag.json"
done

echo ""
echo "Test runs complete. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*linearrag* 2>/dev/null || echo "  (none found)"