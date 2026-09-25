#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1
#PBS -N pi0_train_8gpu
# ABCI 3.0: 1 ノード (H200 x8) で DDP。投入は ./submit.sh train_pi0_8gpu.sh
# 1 step で batch_size x 8 サンプル消費する点に注意 (docs/source/multi_gpu_training.mdx)
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /etc/profile.d/modules.sh
module load singularitypro
source ./env.sh

export TMPDIR="${PBS_LOCALDIR:-/tmp}"
NGPU="${NGPU:-8}"
nvidia-smi

cd ..   # apptainer/
exec ./run.sh torchrun --nproc-per-node="$NGPU" /opt/venv/bin/lerobot-train \
    --policy.type=pi0 \
    --policy.pretrained_path="$PRETRAINED" \
    --dataset.repo_id="$DATASET" \
    --output_dir="outputs/train/${JOB_NAME}_${NGPU}gpu" \
    --job_name="${JOB_NAME}_${NGPU}gpu" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_freq="$SAVE_FREQ" \
    --policy.device=cuda \
    --wandb.enable="${WANDB_ENABLE:-false}"
