#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -N pi05_eval
# ABCI 3.0: 学習済み checkpoint をシミュレーション (aloha / libero) で評価する (成功率ベンチマーク)。
#   cd apptainer/common
#   ./submit.sh ../eval/eval_pi05.sh                                   # env.sh の CKPT / EVAL_ENV
#   CKPT=outputs/train/pi05_aloha_sim/checkpoints/010000/pretrained_model ./submit.sh ../eval/eval_pi05.sh
#   EVAL_ENV=libero EVAL_TASK=libero_spatial EVAL_BATCH=1 ./submit.sh ../eval/eval_pi05.sh
#   研究室 Slurm: ./submit.sh ../eval/eval_pi05.sh -p part_80gb --gres=gpu:a100:1
# 結果: outputs/eval/<JOB_NAME>/<env>_<task>/eval_info.json (+ videos/ に先頭 10 episode の mp4)
set -euo pipefail
cd "${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"   # = apptainer/eval
COMMON="$(cd ../common && pwd)"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load singularitypro
fi
source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
export TMPDIR="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"

lerobot_eval_args
echo "lerobot-eval ${LEROBOT_ARGS[*]}"
exec "$COMMON/run.sh" lerobot-eval "${LEROBOT_ARGS[@]}"
