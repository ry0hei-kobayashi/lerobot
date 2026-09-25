#!/bin/bash
# ABCI にジョブを投げる。
#   ./submit.sh train_pi0.sh            # rt_HG (H200 x1)
#   ./submit.sh train_pi0_8gpu.sh       # rt_HF (H200 x8, 1ノード)
#   ./submit.sh eval_pi0.sh
#   WALLTIME=48:00:00 ./submit.sh train_pi0.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
JOB="${1:?usage: submit.sh <job script> [extra qsub args]}"; shift || true
WALLTIME="${WALLTIME:-24:00:00}"
mkdir -p "$HERE/logs"
cd "$HERE"
qsub -P "$ABCI_GROUP" -l "walltime=$WALLTIME" -o "$HERE/logs/" -j oe -V "$@" "$JOB"
