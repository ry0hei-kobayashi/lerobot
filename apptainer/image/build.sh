#!/bin/bash
# ローカルで SIF を build する。
#   ./build.sh                   # pi05.sif を build して apptainer test
#   ./build.sh --runscript-only  # %runscript だけ差し替え (数秒)
#   ./build.sh --sandbox         # pi05.sandbox/ を作る (デバッグ用。*.sandbox/ は git-ignore 済み)
#
# 環境変数:
#   LEROBOT_SIF       出力先 (default: このディレクトリの pi05.sif)
#   APPTAINER_TMPDIR  build 一時領域 (default: apptainer/.build_tmp。/tmp が小さいマシン向け)
#   CONTAINER_CMD     apptainer / singularity (default: apptainer)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF="$HERE/pi05.def"
SIF="${LEROBOT_SIF:-$HERE/pi05.sif}"
CTR="${CONTAINER_CMD:-apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$(cd "$HERE/.." && pwd)/.build_tmp}"
mkdir -p "$APPTAINER_TMPDIR" "$(dirname "$SIF")"

cd "$HERE"   # %files の相対パス (../../) は cwd 基準。必ず image/ で実行する
case "${1:-}" in
    --runscript-only) exec "$CTR" build --force --section runscript "$SIF" "$DEF" ;;
    --sandbox)        exec "$CTR" build --fakeroot --force --sandbox "${SIF%.sif}.sandbox" "$DEF" ;;
    "")
        "$CTR" build --fakeroot --force "$SIF" "$DEF"
        "$CTR" test "$SIF"
        ls -lh "$SIF"
        ;;
    *) echo "usage: $0 [--runscript-only|--sandbox]" >&2; exit 2 ;;
esac
