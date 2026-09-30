#!/bin/bash
# ローカル PC (docker) で学習済み checkpoint をシミュレーション (aloha / libero) で評価する。
#   ./eval_pi05.sh                                                       # env.sh の CKPT / EVAL_ENV
#   CKPT=outputs/train/smoke/checkpoints/last/pretrained_model EVAL_EPISODES=2 EVAL_BATCH=2 ./eval_pi05.sh
#   EVAL_ENV=libero EVAL_TASK=libero_spatial EVAL_BATCH=1 ./eval_pi05.sh
# 結果: outputs/eval/<JOB_NAME>/<env>_<task>/eval_info.json (+ videos/ に先頭 10 episode の mp4)。
#       既にあれば <env>_<task>_<日時>/ に出る (上書きしない)。CKPT 未指定なら最新の学習 run を自動選択
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$(cd "$HERE/../../apptainer/common" && pwd)"
[ -f "$COMMON/env.sh" ] && source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"

lerobot_eval_args
echo "lerobot-eval ${LEROBOT_ARGS[*]}"
exec "$HERE/../image/run.sh" lerobot-eval "${LEROBOT_ARGS[@]}"
