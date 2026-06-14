#!/bin/bash
# Batch training job. Args after the script name are passed straight to train.py.
#
#   sbatch submit.slurm --epochs 30 --model smallcnn --out models/run1
#
# Override scheduler resources on the CLI (these win over the #SBATCH lines):
#   sbatch --qos=test --time=00:20:00 submit.slurm --epochs 5     # quick, high-priority
#   sbatch --qos=gpu  --time=02:00:00 submit.slurm --model resnet18 --epochs 60
#
#SBATCH --job-name=gtsrb
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate.sh

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== train.py $* ==="

# Match dataloader workers to allocated CPUs unless caller overrode --workers.
srun python code/train.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
