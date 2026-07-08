#!/bin/bash
# Submit all 4 per-author training jobs at once.
#   bash train_all.sh
#   bash train_all.sh --qos=test --time=00:20:00     # smoke-test all 4 quickly
#
# Extra args are passed straight through to every `sbatch` call, same as
# submitting one script directly (see comments in sbatch_jobs/*.slurm).
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$_ROOT"

for author in poe doyle twain dickens; do
    echo "[train_all] submitting $author..."
    sbatch "$@" "sbatch_jobs/train_${author}.slurm"
done

echo "[train_all] all submitted — track with: squeue --me"
