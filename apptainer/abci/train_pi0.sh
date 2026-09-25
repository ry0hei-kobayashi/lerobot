#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -N pi0_train
# ABCI 3.0: H200 1 枚で pi0 fine-tune。投入は ./submit.sh train_pi0.sh
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /etc/profile.d/modules.sh
module load singularitypro
source ./env.sh

export TMPDIR="${PBS_LOCALDIR:-/tmp}"     # 計算ノードのローカル NVMe
nvidia-smi

cd ..   # apptainer/
exec ./run.sh lerobot-train \
    --policy.type=pi0 \
    --policy.pretrained_path="$PRETRAINED" \
    --dataset.repo_id="$DATASET" \
    --output_dir="outputs/train/$JOB_NAME" \
    --job_name="$JOB_NAME" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --policy.device=cuda \
    --wandb.enable="${WANDB_ENABLE:-false}"
