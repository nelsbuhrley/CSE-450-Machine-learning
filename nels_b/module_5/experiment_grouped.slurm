#!/bin/bash
# Start-to-finish honest-eval experiment:
#   1. train SmallCNN @48  with the leakage-free group/track split
#   2. train ResNet18 @100 with the same split
#   3. evaluate both + their soft-vote ensemble on the identical held-out val set
# Writes to NEW dirs; never touches models/run*, models/resnet18.
#
#   sbatch experiment_grouped.slurm
#
#SBATCH --job-name=gtsrb-grouped
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --partition=m9g          # P100 — supported by this PyTorch, cheap, plentiful
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source activate.sh

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W="${SLURM_CPUS_PER_TASK:-8}"

echo "=== [1/3] SmallCNN @48 (group split) ==="
srun python code/train.py --model smallcnn --img-size 48 --epochs 60 --batch-size 512 \
     --group-split --workers "$W" --out models/grouped_smallcnn

echo "=== [2/3] ResNet18 @100 (group split) ==="
srun python code/train.py --model resnet18 --img-size 100 --lr 3e-3 --epochs 50 --patience 10 \
     --batch-size 256 --group-split --workers "$W" --out models/grouped_resnet18

echo "=== [3/3] Ensemble (honest val set) ==="
srun python code/ensemble.py \
     --models models/grouped_smallcnn/best.pt models/grouped_resnet18/best.pt \
     --group-split --workers "$W" --out models/grouped_ensemble.json

echo "=== DONE | $(date) ==="
cat models/grouped_ensemble.json
