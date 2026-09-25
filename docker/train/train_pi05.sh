#!/bin/bash
# ローカル PC (docker, RTX A6000 48GB) で pi0.5 を fine-tune。
#   cp ../../apptainer/common/env.sh.example ../../apptainer/common/env.sh   (初回)
#   BATCH_SIZE=8 TRAIN_EXPERT_ONLY=true ./train_pi05.sh
#   BATCH_SIZE=4 STEPS=20 SAVE_FREQ=20 COMPILE_MODEL=false ./train_pi05.sh   # スモークテスト
# 学習引数は apptainer/common/lerobot_args.sh と共通 (ABCI と同じコマンドが走る)。
# 48GB では全パラメータ学習は厳しいので TRAIN_EXPERT_ONLY=true (VLM 凍結) を推奨。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$(cd "$HERE/../../apptainer/common" && pwd)"
[ -f "$COMMON/env.sh" ] && source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
lerobot_require_token

lerobot_train_args
echo "lerobot-train ${LEROBOT_ARGS[*]}"
exec "$HERE/../image/run.sh" lerobot-train "${LEROBOT_ARGS[@]}"
