#!/bin/bash
# ローカル PC (docker) で fine-tune 済み pi0.5 checkpoint を gym-aloha (ALOHA 2 シミュレーション) で
# rollout し、全 episode の動画を保存する (apptainer/deploy/deploy_pi05_aloha.sh のローカル版)。
#   ./deploy_pi05_aloha.sh
#   CKPT=... DEPLOY_TASK=AlohaInsertion-v0 DEPLOY_EPISODES=5 ./deploy_pi05_aloha.sh
# 出力: outputs/deploy/<JOB_NAME>/<task>_<timestamp>/{eval_info.json, videos/aloha_0/eval_episode_N.mp4}
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON="$(cd "$HERE/../../apptainer/common" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ -f "$COMMON/env.sh" ] && source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
[ "$DEPLOY_EPISODES" -le 10 ] || echo "WARN: 動画が保存されるのは先頭 10 episode のみです" >&2

lerobot_deploy_args
echo "lerobot-eval ${LEROBOT_ARGS[*]}"
"$HERE/../image/run.sh" lerobot-eval "${LEROBOT_ARGS[@]}"

echo "==== deploy 完了: $REPO/$LEROBOT_OUTPUT_DIR"
find "$REPO/$LEROBOT_OUTPUT_DIR/videos" -name "*.mp4" 2>/dev/null | sort || true
python3 - "$REPO/$LEROBOT_OUTPUT_DIR/eval_info.json" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d.get("overall", d), indent=2, ensure_ascii=False))
PY
