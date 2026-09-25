#!/bin/bash
# ABCI 上で SIF を build する (ローカルで build して rsync する方が確実。その場合は不要)。
# SIF の作成はログインノードで行う (計算ジョブは投げない):
#   cd ~/lerobot/apptainer/image && ./build_on_abci.sh
# 20〜40 分かかるので、SSH 切断に備えて nohup / tmux の中で実行するとよい:
#   nohup ./build_on_abci.sh > build.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/env.sh"
source /etc/profile.d/modules.sh
module load singularitypro

# build 中の一時ファイル / キャッシュはホームに置かない (数十 GB になる)。
# ログインノードには $PBS_LOCALDIR が無いのでグループ領域を使う
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/groups/$ABCI_GROUP/$USER/tmp}"
export SINGULARITY_CACHEDIR="/groups/$ABCI_GROUP/$USER/.singularity"
SIF="${LEROBOT_SIF:-/groups/$ABCI_GROUP/$USER/sif/pi05.sif}"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$(dirname "$SIF")"

cd "$HERE"   # %files の相対パスは cwd 基準
singularity build --fakeroot --force "$SIF" pi05.def
singularity test "$SIF"
