#!/bin/bash
#SBATCH --job-name=qsc_ltr
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qsc_ltr/%x-%j.out
#SBATCH --error=logs/qsc_ltr/%x-%j.err

# QSC + LTR extension. Reuses the M4-v2 artifacts (fused pools, query embeddings,
# cluster stats). Requires lightgbm in the env. All code under src/kmeans/.
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source ~/.bashrc
conda activate "${ENV_NAME:-sra}"

mkdir -p logs/qsc_ltr results/comparisons results/analysis
python -c "import lightgbm" 2>/dev/null || pip install --quiet lightgbm

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CFG="${CFG:-src/kmeans/configs/qsc_ltr_extension.yaml}"

PYTHONPATH=src python src/kmeans/scripts/run_qsc_ltr_extension.py --config "$CFG"

echo "QSC + LTR extension complete."
