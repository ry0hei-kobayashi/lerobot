#!/bin/bash
# コンテナ内で任意のコマンドを実行する共通ラッパー (apptainer / singularity 自動判定)。
#   ./run.sh lerobot-train --policy.type=pi05 ...
#   ./run.sh                       # 対話 bash
#
# 環境変数で調整:
#   LEROBOT_SIF    使う SIF (default: ../image/pi05.sif)
#   LEROBOT_REPO   /opt/lerobot に bind するリポジトリ (default: このディレクトリの 2 つ上)
#   HF_HOME        HF のキャッシュ (default: ~/.cache/huggingface)。ABCI ではグループ領域を指定
#   LEROBOT_BINDS  追加 bind ("src:dst,src2:dst2" 形式)
#   CONTAINER_CMD  apptainer / singularity を強制指定
#
# env.sh はここでは読まない (ジョブスクリプト側で source する)。対話利用時は
#   source ../common/env.sh && ../common/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${LEROBOT_SIF:-$(cd "$HERE/../image" && pwd)/pi05.sif}"
REPO="${LEROBOT_REPO:-$(cd "$HERE/../.." && pwd)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [ -n "${CONTAINER_CMD:-}" ]; then
    CTR="$CONTAINER_CMD"
elif command -v apptainer >/dev/null 2>&1; then
    CTR=apptainer
elif command -v singularity >/dev/null 2>&1; then
    CTR=singularity
else
    echo "ERROR: apptainer / singularity が見つかりません (ABCI なら: module load singularitypro)" >&2
    exit 1
fi

[ -f "$SIF" ] || { echo "ERROR: SIF がありません: $SIF  (image/build.sh で作るか LEROBOT_SIF を設定)" >&2; exit 1; }
[ -f "$REPO/pyproject.toml" ] || { echo "ERROR: リポジトリではありません: $REPO" >&2; exit 1; }
mkdir -p "$HF_HOME"

BINDS=("--bind" "$REPO:/opt/lerobot" "--bind" "$HF_HOME:$HF_HOME")
# ABCI の計算ノードローカル NVMe / グループ領域があれば bind
[ -n "${PBS_LOCALDIR:-}" ] && [ -d "$PBS_LOCALDIR" ] && BINDS+=("--bind" "$PBS_LOCALDIR")
[ -n "${ABCI_GROUP:-}" ] && [ -d "/groups/$ABCI_GROUP" ] && BINDS+=("--bind" "/groups/$ABCI_GROUP")
if [ -n "${LEROBOT_BINDS:-}" ]; then
    IFS=',' read -ra EXTRA <<< "$LEROBOT_BINDS"
    for b in "${EXTRA[@]}"; do BINDS+=("--bind" "$b"); done
fi

ENVS=("--env" "HF_HOME=$HF_HOME" "--env" "HF_LEROBOT_HOME=$HF_HOME/lerobot")
for v in HF_TOKEN WANDB_API_KEY CUDA_VISIBLE_DEVICES TMPDIR; do
    [ -n "${!v:-}" ] && ENVS+=("--env" "$v=${!v}")
done

exec "$CTR" run --nv --pwd /opt/lerobot "${BINDS[@]}" "${ENVS[@]}" "$SIF" "$@"
