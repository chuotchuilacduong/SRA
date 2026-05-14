#!/bin/bash
# Standalone script - completely detached
cd /Users/hiro/Documents/Vinuni/code/SRA

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p results/retrieval_all

exec /Users/hiro/miniconda3/envs/linearag311/bin/sragents retrieve \
    --retriever linearrag \
    --retriever-arg dataset_name="bench_full" \
    --retriever-arg max_chars_per_passage=3000 \
    --retriever-arg batch_size=8 \
    --retriever-arg max_workers=1 \
    --corpus data/bench/corpus/corpus.json \
    --instances data/bench/instances/champ.json \
    --output results/retrieval_all/champ-linearrag.json \
    --top-k 50
