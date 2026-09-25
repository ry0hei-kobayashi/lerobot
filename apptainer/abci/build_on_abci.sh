#!/bin/bash
# ABCI 上で SIF をビルドする (ローカルでビルドして scp する方が確実。その場合は不要)。
# ログインノードでは重い処理ができないので対話ジョブで実行する:
#   qsub -I -P $ABCI_GROUP -q rt_HC -l select=1 -l walltime=2:00:00
#   cd lerobot/apptainer/abci && ./build_on_abci.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
source /etc/profile.d/modules.sh
module load singularitypro

# ビルド中の一時ファイル / キャッシュはホームに置かない (数十 GB になる)
export SINGULARITY_TMPDIR="${PBS_LOCALDIR:-/groups/$ABCI_GROUP/$USER/tmp}"
export SINGULARITY_CACHEDIR="/groups/$ABCI_GROUP/$USER/.singularity"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR" "$(dirname "$LEROBOT_SIF")"

cd "$HERE/.."
singularity build --fakeroot --force "$LEROBOT_SIF" pi0.def
