#!/bin/bash
# ローカル PC 用 docker イメージを build する (context はリポジトリルート)。
#   ./build.sh              # ${USER}/lerobot-pi05:latest
#   ./build.sh --no-cache
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UID                                      # bash 組込み (readonly) だが export は可。Dockerfile の ARG に渡す
GID="$(id -g)"; export GID
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$HERE"
docker compose build "$@"
