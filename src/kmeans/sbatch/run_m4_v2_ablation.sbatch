#!/bin/bash
#SBATCH --job-name=m4v2
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1            # only needed for the query-embedding stage; drop if cached
#SBATCH --time=12:00:00
#SBATCH --output=logs/m4_v2/%x-%j.out
#SBATCH --error=logs/m4_v2/%x-%j.err

# M4-v2 ablation: cluster stats -> query embeddings -> ablation -> report.
# All code lives in src/kmeans/; outputs go to results/ and logs/.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
ENV_NAME="${ENV_NAME:-sra}"
conda activate "$ENV_NAME"

CFG="${CFG:-src/kmeans/configs/m4_v2_default.yaml}"
DATASETS="${DATASETS:-theoremqa logicbench toolqa champ medcalcbench bigcodebench}"

# Step 4 (offline): cluster reliability stats
python src/kmeans/scripts/build_m4_v2_cluster_stats.py --config "$CFG"

# Query embeddings (the only GPU-friendly step; skipped if already cached)
python src/kmeans/scripts/build_m4_v2_query_embeddings.py --config "$CFG" --datasets $DATASETS

# Ablation A0-A8 (CPU; reuses embeddings + centroids)
python src/kmeans/scripts/run_m4_v2_ablation.py --config "$CFG" --datasets $DATASETS

# Reports + tables
python src/kmeans/scripts/write_m4_v2_report.py --config "$CFG"

echo "M4-v2 ablation complete."
