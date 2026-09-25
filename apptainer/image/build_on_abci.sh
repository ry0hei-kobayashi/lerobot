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

# build 中の一時ファイル / キャッシュはホーム直下に置く (数十 GB になるので quota に注意)。
# ログインノードには $PBS_LOCALDIR が無い。グループ領域 (/groups/$ABCI_GROUP/$USER) に
# 書き込めるなら SINGULARITY_TMPDIR / SINGULARITY_CACHEDIR で上書きしてよい
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$HOME/.singularity/tmp}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity/cache}"
SIF="${LEROBOT_SIF:-$HOME/sif/pi05.sif}"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$(dirname "$SIF")"

cd "$HERE"   # %files の相対パスは cwd 基準
singularity build --fakeroot --force "$SIF" pi05.def
singularity test "$SIF"
