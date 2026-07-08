#!/bin/bash
# Train the Transformers-movies char-RNN. Submit from the module_6 root:
#   sbatch sbatch_jobs/train_transformers.slurm
#
# Runs on the cs partition's A100s via standby QOS (borrowed capacity from
# another group) — ~4-5x faster than the m9g P100s and avoids a cuDNN bug
# they hit. Standby jobs can be preempted; if that becomes a problem, fall
# back to `--partition= --qos=gpu` (the m9g default) and set use_cudnn=False
# in main.py.
#SBATCH --job-name=train-transformers
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

# The movie-transcript corpus is ~0.5MB (~40 batches/epoch) vs 3-22MB for the
# author corpora, so run more epochs to get a comparable number of gradient
# steps (150 x 40 ~= 6k, about what Poe gets from its 30).
export EPOCHS=150

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== training on Transformers movie corpus (runs/transformers) ==="

cd runs/transformers
srun python ../../code/main.py
