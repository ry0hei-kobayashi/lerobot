#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -N pi05_deploy_aloha
# ABCI 3.0: fine-tune 済み pi0.5 checkpoint を gym-aloha (MuJoCo ALOHA 2 シミュレーション) で
# rollout し、全 episode の動画を保存する ("deploy" = 実機の代わりにシミュレータへ展開)。
#   cd apptainer/common
#   ./submit.sh ../deploy/deploy_pi05_aloha.sh
#   CKPT=... DEPLOY_TASK=AlohaInsertion-v0 DEPLOY_EPISODES=5 ./submit.sh ../deploy/deploy_pi05_aloha.sh
#   研究室 Slurm: ./submit.sh ../deploy/deploy_pi05_aloha.sh -p part_80gb --gres=gpu:a100:1
# 出力: outputs/deploy/<JOB_NAME>/<task>_<timestamp>/{eval_info.json, videos/aloha_0/eval_episode_N.mp4}
# 注: lerobot-eval が動画を書くのは先頭 10 episode まで (src/lerobot/scripts/lerobot_eval.py)。
set -euo pipefail
cd "${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"   # = apptainer/deploy
COMMON="$(cd ../common && pwd)"
REPO="$(cd ../.. && pwd)"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load singularitypro
fi
source "$COMMON/env.sh"
source "$COMMON/lerobot_args.sh"
export TMPDIR="${PBS_LOCALDIR:-${TMPDIR:-/tmp}}"
[ "$DEPLOY_EPISODES" -le 10 ] || echo "WARN: 動画が保存されるのは先頭 10 episode のみです" >&2

lerobot_deploy_args
echo "lerobot-eval ${LEROBOT_ARGS[*]}"
"$COMMON/run.sh" lerobot-eval "${LEROBOT_ARGS[@]}"

echo "==== deploy 完了: $REPO/$LEROBOT_OUTPUT_DIR"
find "$REPO/$LEROBOT_OUTPUT_DIR/videos" -name "*.mp4" 2>/dev/null | sort || true
python3 - "$REPO/$LEROBOT_OUTPUT_DIR/eval_info.json" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d.get("overall", d), indent=2, ensure_ascii=False))
PY
