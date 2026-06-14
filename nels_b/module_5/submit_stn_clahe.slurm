#!/bin/bash
# Train ONE STN + CLAHE model and self-score it on the mini-holdout.
# Designed to be submitted many times (one independent job / GPU per model).
#
#   sbatch submit_stn_clahe.slurm <model> <img_size> [extra train.py args...]
#   sbatch submit_stn_clahe.slurm smallcnn 100
#   sbatch submit_stn_clahe.slurm resnet18 100 --lr 3e-3 --batch-size 128
#
# Writes only to models/stn_clahe/<model>_<img>; trains on training/ (no leakage);
# the mini-holdout is touched only for the final scoring pass.
#
#SBATCH --job-name=stnclahe
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --partition=m9g          # P100 — supported by this PyTorch build
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/stn_clahe
source activate.sh

MODEL="${1:?usage: submit_stn_clahe.slurm <model> <img_size> [extra args]}"
IMG="${2:?need img_size}"
shift 2
OUT="models/stn_clahe/${MODEL}_${IMG}"
W="${SLURM_CPUS_PER_TASK:-8}"

echo "=== node $(hostname) | $(date) | ${MODEL}@${IMG} +STN +CLAHE -> ${OUT} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Defaults first so caller args (passed via "$@") win on conflict.
srun python code/train.py \
     --model "$MODEL" --img-size "$IMG" --stn --clahe \
     --epochs 35 --batch-size 256 --lr 1e-3 --patience 8 \
     --group-split --workers "$W" --out "$OUT" "$@"

echo "=== scoring ${OUT} on mini-holdout ==="
srun python code/predict.py --models "$OUT/best.pt" \
     --images mini_holdout --answers mini_holdout_answers.csv \
     --device cuda --out "$OUT/holdout_metrics.json"

echo "=== DONE | $(date) ==="
cat "$OUT/holdout_metrics.json"
