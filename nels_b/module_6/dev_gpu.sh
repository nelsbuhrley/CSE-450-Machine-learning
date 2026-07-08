#!/bin/bash
# Grab a short interactive GPU shell for live debugging / tuning.
# Uses the 'test' QOS (1h max, high priority) so it schedules fast and uses
# little allocation. Override time/qos/gres via env vars if needed.
#
#   bash dev_gpu.sh
#   # then on the node:  source activate.sh && cd code && python main.py
#
QOS="${QOS:-test}"
TIME="${TIME:-01:00:00}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-16G}"
PART="${PART:-}"            # optional partition, e.g. cs3 (B200), m13h (H200)

part_arg=()
[ -n "$PART" ] && part_arg=(--partition="$PART")

echo "[dev_gpu] requesting ${GRES} | qos=${QOS} | part=${PART:-default} | time=${TIME} ..."
echo "[dev_gpu] once on the node run:  source activate.sh && cd code && python main.py"
exec srun --account=rls62 --qos="${QOS}" "${part_arg[@]}" --gres="${GRES}" \
     --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" \
     --pty bash
