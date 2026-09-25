#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -N pi05_train_8gpu
# ABCI 3.0: 1 ノード (H200 x8) で DDP。
#   投入: cd apptainer/common && ./submit.sh ../train/train_pi05_8gpu.sh
# 1 step で BATCH_SIZE x NGPU サンプル消費する (docs/source/multi_gpu_training.mdx)。
# STEPS / BATCH_SIZE はそれに合わせて調整する。
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # = apptainer/train
COMMON="$(cd ../common && pwd)"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load singularitypro
fi
source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
lerobot_require_token

export TMPDIR="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"
NGPU="${NGPU:-8}"
nvidia-smi || true

lerobot_train_args "_${NGPU}gpu"
echo "torchrun --nproc-per-node=$NGPU lerobot-train ${LEROBOT_ARGS[*]}"
exec "$COMMON/run.sh" torchrun --standalone --nproc-per-node="$NGPU" \
    /opt/venv/bin/lerobot-train "${LEROBOT_ARGS[@]}"
