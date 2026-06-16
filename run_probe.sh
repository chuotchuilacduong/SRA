#!/bin/bash
#SBATCH --job-name=skill_probe           # Job name

#SBATCH --output=log_%j.out              # Stdout log file
#SBATCH --error=log_%j.err               # Stderr log file

#SBATCH --time=02:00:00                  # Max runtime (HH:MM:SS)

#SBATCH --partition=main                 # Partition/queue

#SBATCH --ntasks=1                       # Single process
#SBATCH --cpus-per-task=4               # CPU threads

#SBATCH --mem=32G                        # Memory

#SBATCH --gres=gpu:1                     # Request 1 GPU

# Activate conda environment (adjust <env_name> to your environment)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate sra

echo "Start at: $(date)"
echo "Running on node: $(hostname)"

# python -m experiments.skill_probe_hidden.run_probe --n-queries 200 --datasets logicbench
# python -m experiments.skill_probe_hidden.run_probe --n-queries 200 --datasets medcalcbench
# python -m experiments.skill_probe_hidden.run_probe --n-queries 200 --datasets theoremqa
python -m experiments.skill_probe_hidden.run_probe --n-queries 200 --datasets champ

python -m experiments.skill_probe_hidden.analyze

echo "Done at: $(date)"


'''
#!/bin/bash
# Submit 4 independent jobs, one per dataset, all running in parallel.
# Usage: bash run_probe_parallel.sh

DATASETS=(logicbench medcalcbench theoremqa champ)

for DS in "${DATASETS[@]}"; do
    sbatch --job-name="probe_${DS}" \
           --output="log_${DS}_%j.out" \
           --error="log_${DS}_%j.err" \
           --time=02:00:00 \
           --partition=main \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=32G \
           --gres=gpu:1 \
           --wrap="source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate sra && \
                   echo 'Start at: \$(date)' && echo 'Node: \$(hostname)' && \
                   python -m experiments.skill_probe_hidden.run_probe --n-queries 200 --datasets ${DS} && \
                   echo 'Done at: \$(date)'"
    echo "Submitted job for dataset: ${DS}"
done

'''