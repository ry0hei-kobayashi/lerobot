#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -N pi0_eval
# ABCI 3.0: 学習済み checkpoint をシミュレーション環境で評価。
#   CKPT=outputs/train/pi0_finetune/checkpoints/last/pretrained_model ENV_TYPE=libero ./submit.sh eval_pi0.sh
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /etc/profile.d/modules.sh
module load singularitypro
source ./env.sh

export TMPDIR="${PBS_LOCALDIR:-/tmp}"
CKPT="${CKPT:-outputs/train/$JOB_NAME/checkpoints/last/pretrained_model}"
ENV_TYPE="${ENV_TYPE:-libero}"

cd ..   # apptainer/
exec ./run.sh lerobot-eval \
    --policy.path="$CKPT" \
    --env.type="$ENV_TYPE" \
    --eval.batch_size="${EVAL_BATCH:-10}" \
    --eval.n_episodes="${EVAL_EPISODES:-50}" \
    --output_dir="outputs/eval/$JOB_NAME" \
    --policy.device=cuda
