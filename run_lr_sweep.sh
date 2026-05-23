#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
PY=/Users/hiro/miniconda3/envs/linearag311/bin/python

DEV_POOL=results/pool/loss_lr_dev_pool.json
TRAIN_PAIRS=results/train/all_train_pairs_v2.json
INSTANCES=results/joint_all_instances.json

for cfg in 5e-6 5e-5; do
    echo "===================================================="
    echo "=== TRAINING: listwise lr=$cfg, 2 epochs ==="
    echo "===================================================="
    OUT="results/models/ce-lr-$cfg"
    if [ -f "$OUT/train_summary.json" ]; then
        echo "[$cfg] already trained, skip"
        continue
    fi
    $PY -m sragents.cli.main train-rerank \
        --config "configs/train_lr_${cfg}.yaml" \
        --train-pairs "$TRAIN_PAIRS" \
        --dev-pool "$DEV_POOL" \
        --instances "$INSTANCES" \
        --out "$OUT"
done
echo "ALL DONE"
