#!/bin/bash
#SBATCH -p part_80gb
#SBATCH --gres=gpu:a100:1
#SBATCH -J pi05_train
# 研究室 Slurm クラスタ (A100 80GB) で pi0.5 を fine-tune。
#   投入: cd apptainer/common && ./submit.sh ../train/slurm_train_pi05.sh
#   40GB MIG: ./submit.sh ../train/slurm_train_pi05.sh -p part_40gb --gres=gpu:a100_3g.40gb:1
# 学習内容は train_pi05.sh (ABCI) と同じ lerobot_train_args。A100 80GB なら BATCH_SIZE=16-32。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # = apptainer/train
COMMON="$(cd ../common && pwd)"
source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
lerobot_require_token
nvidia-smi || true

lerobot_train_args
echo "lerobot-train ${LEROBOT_ARGS[*]}"
exec "$COMMON/run.sh" lerobot-train "${LEROBOT_ARGS[@]}"
