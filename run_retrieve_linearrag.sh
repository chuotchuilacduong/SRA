#!/usr/bin/env bash
# Stage 1: build LinearRAG retrieval index for all 6 SRA-Bench datasets.
#
# First dataset (~theoremqa) performs full NER on the 26k-skill corpus
# (~30–90 min on CPU depending on hardware). All subsequent datasets
# reuse the NER cache at import/bench_full/ner_results.json and finish
# in minutes (only the query step re-runs).
#
# Usage (inside conda env linearag311, from SR-Agents/):
#   bash run_retrieve_linearrag.sh
#   bash run_retrieve_linearrag.sh champ          # single dataset
#   bash run_retrieve_linearrag.sh champ theoremqa  # subset

set -euo pipefail

CORPUS="data/bench/corpus/corpus.json"
INSTANCES_DIR="data/bench/instances"
OUTPUT_DIR="data/bench/results/retrieval"
DATASET_NAME="bench_full"   # shared NER cache key across all datasets
TOP_K=50

mkdir -p "$OUTPUT_DIR"

DATASETS=("${@:-theoremqa logicbench toolqa champ medcalcbench bigcodebench}")
# If no args given, run all 6
if [ $# -eq 0 ]; then
    DATASETS=(theoremqa logicbench toolqa champ medcalcbench bigcodebench)
fi

for DS in "${DATASETS[@]}"; do
    INSTANCES="$INSTANCES_DIR/${DS}.json"
    OUTPUT="$OUTPUT_DIR/${DS}-linearrag.json"

    if [ ! -f "$INSTANCES" ]; then
        echo "[SKIP] $DS: instances file not found at $INSTANCES"
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "[SKIP] $DS: output already exists at $OUTPUT"
        continue
    fi

    echo ""
    echo "=== Retrieving: $DS ==="
    echo "    Output  : $OUTPUT"
    echo "    Started : $(date '+%H:%M:%S')"

    sragents retrieve \
        --retriever linearrag \
        --retriever-arg dataset_name="$DATASET_NAME" \
        --retriever-arg max_chars_per_passage=10000 \
        --retriever-arg batch_size=64 \
        --retriever-arg max_workers=1 \
        --corpus "$CORPUS" \
        --instances "$INSTANCES" \
        --output "$OUTPUT" \
        --top-k $TOP_K

    echo "    Done    : $(date '+%H:%M:%S')"
done

echo ""
echo "All retrieve runs complete. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*linearrag* 2>/dev/null || echo "  (none found)"
