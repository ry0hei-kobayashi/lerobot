#!/bin/bash
# クラスタにジョブを投げる。qsub (ABCI, PBS Pro) / sbatch (研究室 Slurm) を自動判定。
#   ./submit.sh ../train/train_pi05.sh                  # ABCI rt_HG (H200 x1)
#   ./submit.sh ../train/train_pi05_8gpu.sh             # ABCI rt_HF (H200 x8)
#   ./submit.sh ../eval/eval_pi05.sh
#   ./submit.sh ../deploy/deploy_pi05_aloha.sh
#   ./submit.sh ../train/slurm_train_pi05.sh            # 研究室 Slurm
#   WALLTIME=48:00:00 ./submit.sh ../train/train_pi05.sh
#   CKPT=outputs/train/x/checkpoints/010000/pretrained_model EVAL_ENV=libero ./submit.sh ../eval/eval_pi05.sh
# 2 つ目以降の引数は qsub / sbatch にそのまま渡す (例: -p part_40gb --gres=gpu:a100_3g.40gb:1)。
# ログは apptainer/logs/ (git-ignore 済み)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
JOB="${1:?usage: submit.sh <job script> [extra qsub/sbatch args]}"; shift || true
JOB="$(cd "$(dirname "$JOB")" && pwd)/$(basename "$JOB")"
[ -f "$JOB" ] || { echo "ERROR: ジョブスクリプトがありません: $JOB" >&2; exit 1; }
mkdir -p "$HERE/../logs"; LOGS="$(cd "$HERE/../logs" && pwd)"

# ジョブスクリプトは PBS_O_WORKDIR / SLURM_SUBMIT_DIR (= 投入時の cwd) を起点に
# ../common を探すので、必ずスクリプトのあるディレクトリから投入する
cd "$(dirname "$JOB")"
if command -v qsub >/dev/null 2>&1; then
    qsub -P "$ABCI_GROUP" -l "walltime=${WALLTIME:-24:00:00}" -o "$LOGS/" -j oe -V "$@" "$JOB"
elif command -v sbatch >/dev/null 2>&1; then
    sbatch --export=ALL ${WALLTIME:+-t "$WALLTIME"} -o "$LOGS/%x-%j.out" "$@" "$JOB"
else
    echo "ERROR: qsub / sbatch が見つかりません" >&2; exit 1
fi
