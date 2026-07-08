#!/bin/bash
# Train the Mark Twain char-RNN. Submit from the module_6 root:
#   sbatch sbatch_jobs/train_twain.slurm
#
# Runs on the cs partition's A100s via standby QOS (borrowed capacity from
# another group) — ~4-5x faster than the m9g P100s and avoids a cuDNN bug
# they hit. Standby jobs can be preempted; if that becomes a problem, fall
# back to `--partition= --qos=gpu` (the m9g default) and set use_cudnn=False
# in main.py.
#SBATCH --job-name=train-twain
#SBATCH --account=rls62
#SBATCH --qos=standby
#SBATCH --partition=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate.sh

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== training on Twain corpus (runs/twain) ==="

cd runs/twain
srun python ../../code/main.py
