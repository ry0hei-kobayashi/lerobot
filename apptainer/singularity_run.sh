#!/bin/bash
# 研究室 Slurm クラスタ用 (run_40gb.sh / run_80gb.sh から呼ばれる)。
# 実行内容は abci/train_pi0.sh と同じ。中身を変えたい場合はそちらと揃える。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/abci/env.sh" ] && source "$HERE/abci/env.sh"
[ -n "${HF_TOKEN:-}" ] || { echo 'ERROR: HF_TOKEN を設定してください (abci/env.sh 参照)'; exit 1; }

exec "$HERE/run.sh" lerobot-train \
    --policy.type=pi0 \
    --policy.pretrained_path="${PRETRAINED:-lerobot/pi0_base}" \
    --dataset.repo_id="${DATASET:-danaaubakirova/koch_test}" \
    --output_dir="outputs/train/${JOB_NAME:-pi0_finetune}" \
    --job_name="${JOB_NAME:-pi0_finetune}" \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.compile_model=true \
    --batch_size="${BATCH_SIZE:-8}" \
    --steps="${STEPS:-30000}" \
    --save_freq="${SAVE_FREQ:-5000}" \
    --policy.device=cuda \
    --wandb.enable="${WANDB_ENABLE:-false}"
