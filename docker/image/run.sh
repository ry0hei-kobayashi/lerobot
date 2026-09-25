#!/bin/bash
# コンテナ内で任意のコマンドを実行する共通ラッパー (apptainer/common/run.sh の docker 版)。
#   ./run.sh lerobot-train --policy.type=pi05 ...
#   ./run.sh                    # 対話 bash
#   ./run.sh down               # コンテナ停止・削除
# コンテナが無ければ docker compose up -d で常駐起動し、docker compose exec で実行する。
#
# 環境変数:
#   HF_HOME                 ホスト側 HF キャッシュ (default ~/.cache/huggingface)。up 時にマウントされる
#   HF_TOKEN / WANDB_API_KEY / CUDA_VISIBLE_DEVICES   exec ごとに渡す
# env.sh はここでは読まない。対話利用時は source ../../apptainer/common/env.sh してから呼ぶ
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UID
GID="$(id -g)"; export GID
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
cd "$HERE"

if [ "${1:-}" = "down" ]; then exec docker compose down; fi

# 起動していなければ起動 (イメージが無ければ build も走る)
if [ -z "$(docker compose ps -q --status running lerobot 2>/dev/null)" ]; then
    docker compose up -d
fi

OPTS=()
[ -t 0 ] || OPTS+=(-T)                          # パイプ / ジョブ実行時は擬似 TTY を切る
for v in HF_TOKEN WANDB_API_KEY CUDA_VISIBLE_DEVICES; do
    [ -n "${!v:-}" ] && OPTS+=(-e "$v=${!v}")
done

if [ $# -eq 0 ]; then
    exec docker compose exec "${OPTS[@]}" lerobot bash
fi
exec docker compose exec "${OPTS[@]}" lerobot "$@"
