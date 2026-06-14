#!/bin/bash
# Spatial Transformer (STN) sweep: train a small variety of architectures/sizes,
# each with an identity-initialized STN front-end (learned crop/zoom/rotate), on
# the leakage-free group/track split. Then score every STN model AND the existing
# non-STN SmallCNN baseline on the mini-holdout for a direct head-to-head.
#
# Writes ONLY to models/grouped_stn/* — never touches models/run*, resnet18,
# grouped_smallcnn, grouped_resnet18. Training reads training/ only; the
# mini-holdout is used strictly for final scoring (no leakage).
#
#   sbatch experiment_stn_sweep.slurm
#
#SBATCH --job-name=gtsrb-stn
#SBATCH --account=rls62
#SBATCH --qos=gpu
#SBATCH --partition=m9g          # P100 — supported by this PyTorch, cheap, plentiful
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs models/grouped_stn
source activate.sh

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W="${SLURM_CPUS_PER_TASK:-8}"

# model  img  batch  lr     patience  epochs   -> a deliberate spread of arch + resolution,
# all light enough that the whole sweep finishes well inside the 1h allocation.
SWEEP=(
  "smallcnn            48  512  1e-3  7  30"   # direct twin of the 97.01% baseline
  "smallcnn            64  512  1e-3  7  30"   # same arch, higher resolution
  "resnet18            64  256  3e-3  8  30"   # deeper residual arch
  "mobilenet_v3_small  64  512  1e-3  8  30"   # lightweight inverted-residual arch
)

BESTS=()
i=0
for row in "${SWEEP[@]}"; do
  read -r MODEL IMG BS LR PAT EP <<<"$row"
  i=$((i + 1))
  OUT="models/grouped_stn/${MODEL}_${IMG}"
  echo "=== [${i}/${#SWEEP[@]}] ${MODEL} @${IMG} +STN -> ${OUT} ==="
  srun python code/train.py --model "$MODEL" --img-size "$IMG" --stn \
       --epochs "$EP" --batch-size "$BS" --lr "$LR" --patience "$PAT" \
       --group-split --workers "$W" --out "$OUT"
  BESTS+=("$OUT/best.pt")
done

echo "=== scoring all STN models + non-STN SmallCNN baseline on the mini-holdout ==="
srun python code/predict.py \
     --models "${BESTS[@]}" models/grouped_smallcnn/best.pt \
     --images mini_holdout --answers mini_holdout_answers.csv \
     --device cuda --out models/grouped_stn/holdout_metrics.json

echo "=== DONE | $(date) ==="
cat models/grouped_stn/holdout_metrics.json
