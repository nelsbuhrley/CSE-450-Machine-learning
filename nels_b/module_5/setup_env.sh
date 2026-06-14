#!/bin/bash
# One-time environment build. Run on the LOGIN node (needs internet):
#   bash setup_env.sh
#
# Creates a conda env 'gtsrb' with PyTorch (CUDA 12.4 wheels) + helpers.
# The CUDA 12.4 runtime is bundled in the wheels; the cluster driver (CUDA 13.0
# capable) runs it fine on every GPU partition (P100 -> B200).
set -euo pipefail

ENV_NAME="gtsrb"
PY_VER="3.11"
TORCH_CUDA="cu124"   # set to cu128 for newest, cpu for a CPU-only env

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] env '${ENV_NAME}' already exists — skipping create."
else
    echo "[setup] creating env '${ENV_NAME}' (python ${PY_VER})..."
    mamba create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
fi

conda activate "${ENV_NAME}"

echo "[setup] installing PyTorch (${TORCH_CUDA})..."
pip install --upgrade pip
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup] installing training helpers..."
pip install numpy pillow scikit-learn tqdm

echo
echo "[setup] done. Verify GPU access from a compute node with:"
echo "    bash dev_gpu.sh        # grabs an interactive GPU shell"
echo "    source activate.sh && python -c 'import torch; print(torch.cuda.is_available())'"
