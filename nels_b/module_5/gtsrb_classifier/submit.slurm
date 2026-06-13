#!/bin/bash
# ===========================================================================
# SLURM batch script for training the GTSRB CNN on BYU's "Mary Lou" cluster.
# Submit from a login node with:   sbatch submit.slurm
# Monitor with: squeue --me      |  tail -f slurm-<jobid>.out
# ===========================================================================

#SBATCH --job-name=gtsrb-cnn
#SBATCH --time=04:00:00            # walltime  (REQUIRED). Plenty for ~30 epochs.
#SBATCH --nodes=1
#SBATCH --ntasks=8                 # CPU cores -> match --num-workers in python
#SBATCH --mem=32G                  # RAM (REQUIRED)
#SBATCH --gpus=1                   # one GPU. Pin a type with e.g. --gpus=l40s:1
#SBATCH --output=slurm-%j.out      # %j = job id

# --- For quick testing, use the short-queue 'test' QOS (max 1h walltime):
#       sbatch --qos=test --time=01:00:00 submit.slurm
# --- To grab idle private GPUs faster (job may be preempted -> we checkpoint):
#       sbatch --qos=standby --requeue submit.slurm

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname)  |  start $(date)"

# --- Software environment -------------------------------------------------- #
# CUDA runtime (confirmed available on Mary Lou):
module load cuda

# Python + PyTorch. EDIT THIS to match how you set up your env (see CLAUDE.md).
# Option A) conda/mamba environment:
module load miniconda3 2>/dev/null || module load mamba 2>/dev/null || true
source activate gtsrb 2>/dev/null || conda activate gtsrb
# Option B) plain venv instead — comment out A and use:
#   source "$HOME/envs/gtsrb/bin/activate"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- Paths ----------------------------------------------------------------- #
# DATA_ROOT must be the folder that CONTAINS training1/ and training2/.
# Put the data on fast scratch/compute storage, NOT in your git repo.
DATA_ROOT="${DATA_ROOT:-$HOME/gtsrb_data}"
OUT_DIR="runs/gtsrb_$SLURM_JOB_ID"

# --- Train ----------------------------------------------------------------- #
# --resume lets a preempted job pick up from runs/.../last.pt (use with standby)
srun python train_gtsrb.py \
    --data-root "$DATA_ROOT" \
    --out-dir   "$OUT_DIR" \
    --arch      cnn \
    --img-size  48 \
    --epochs    30 \
    --batch-size 256 \
    --num-workers 8 \
    --resume

echo "Done $(date)  ->  $OUT_DIR"
