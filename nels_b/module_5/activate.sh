# Source this to activate the project environment (used by jobs + interactive shells):
#   source activate.sh
# Safe to source on a login node or a compute node.

# Resolve repo root (dir containing this file) even when sourced from elsewhere.
_GTSRB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export GTSRB_ROOT="$_GTSRB_ROOT"

# Conda
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gtsrb || {
    echo "[activate] env 'gtsrb' not found — run: bash setup_env.sh" >&2
    return 1 2>/dev/null || exit 1
}

# Keep BLAS/OpenMP from oversubscribing cores on shared nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[activate] gtsrb env ready | python=$(which python) | root=$GTSRB_ROOT"
