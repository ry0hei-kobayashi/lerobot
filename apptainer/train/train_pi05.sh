#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -N pi05_train
# ABCI 3.0: H200 x1 で pi0.5 を fine-tune。
#   投入: cd apptainer/common && ./submit.sh ../train/train_pi05.sh
#   直接: (rt_HG の対話ジョブ内 / ローカル apptainer で) ./train_pi05.sh
# パラメータは common/env.sh、学習引数の本体は common/lerobot_args.sh (lerobot_train_args)。
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # = apptainer/train
COMMON="$(cd ../common && pwd)"

if [ -f /etc/profile.d/modules.sh ]; then      # ABCI のみ (ローカル apptainer では不要)
    source /etc/profile.d/modules.sh
    module load singularitypro
fi
source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
lerobot_require_token

export TMPDIR="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"   # 計算ノードのローカル NVMe
nvidia-smi || true

lerobot_train_args
echo "lerobot-train ${LEROBOT_ARGS[*]}"
exec "$COMMON/run.sh" lerobot-train "${LEROBOT_ARGS[@]}"
