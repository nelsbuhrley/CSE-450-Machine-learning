#!/bin/bash
# Batch training job.
#
#   sbatch submit.slurm
#   sbatch --qos=test --time=00:20:00 submit.slurm     # quick, high-priority smoke run
#
#SBATCH --job-name=poe-rnn
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate.sh

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== main.py ==="

cd code
srun python main.py
