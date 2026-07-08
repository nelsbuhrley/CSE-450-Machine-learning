#!/bin/bash
# One-time environment build. Run on the LOGIN node (needs internet):
#   bash setup_env.sh
#
# Creates a conda env 'poe_rnn' with TensorFlow (GPU). TF's pip wheel bundles
# its own CUDA/cuDNN runtime (like the PyTorch wheels in module_5), so no
# `module load cuda` is needed — the cluster driver runs it directly.
set -euo pipefail

ENV_NAME="poe_rnn"
PY_VER="3.11"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[setup] env '${ENV_NAME}' already exists — skipping create."
else
    echo "[setup] creating env '${ENV_NAME}' (python ${PY_VER})..."
    mamba create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
fi

conda activate "${ENV_NAME}"

echo "[setup] installing TensorFlow (GPU, bundled CUDA/cuDNN via pip)..."
pip install --upgrade pip
pip install "tensorflow[and-cuda]"

echo "[setup] installing helpers..."
pip install numpy requests

echo
echo "[setup] done. Verify GPU access from a compute node with:"
echo "    bash dev_gpu.sh        # grabs an interactive GPU shell"
echo '    source activate.sh && python -c "import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))"'
