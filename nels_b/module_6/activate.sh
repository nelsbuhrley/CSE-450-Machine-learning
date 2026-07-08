# Source this to activate the project environment (used by jobs + interactive shells):
#   source activate.sh
# Safe to source on a login node or a compute node.

_POE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export POE_ROOT="$_POE_ROOT"

# Conda
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate poe_rnn || {
    echo "[activate] env 'poe_rnn' not found — run: bash setup_env.sh" >&2
    return 1 2>/dev/null || exit 1
}

# Keep BLAS/OpenMP from oversubscribing cores on shared nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# TF is chatty by default (INFO logs on every op placement) — keep it to warnings+.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "[activate] poe_rnn env ready | python=$(which python) | root=$POE_ROOT"
