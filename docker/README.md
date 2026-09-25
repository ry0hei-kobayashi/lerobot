# Docker

This directory contains Dockerfiles for running LeRobot in containerized environments. Both images are **built nightly from `main`** and published to Docker Hub with the full environment pre-baked — no dependency setup required.

## Pre-built Images

```bash
# CPU-only image (based on Dockerfile.user)
docker pull huggingface/lerobot-cpu:latest

# GPU image with CUDA support (based on Dockerfile.internal)
docker pull huggingface/lerobot-gpu:latest
```

## Quick Start

The fastest way to start training is to pull the GPU image and run `lerobot-train` directly. This is the same environment used for all of our CI, so it is a well-tested, batteries-included setup.

```bash
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest

# inside the container:
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
```

## Dockerfiles

### `Dockerfile.user` (CPU)

A lightweight image based on `python:3.12-slim`. Includes all Python dependencies and system libraries but does not include CUDA — there is no GPU support. Useful for exploring the codebase, running scripts, or working with robots, but not practical for training.

### `Dockerfile.internal` (GPU)

A CUDA-enabled image based on `nvidia/cuda`. This is the image for training — mostly used for internal interactions with the GPU cluster.

### `Dockerfile.jetson` (NVIDIA Jetson, community-maintained)

Builds `torch`/`torchcodec`/`torchvision` from source with CUDA support for NVIDIA Jetson Orin (JetPack 6.2, CUDA 12.6) — no prebuilt cp312 wheel exists for this platform yet. **Not** part of the nightly CI/Docker Hub pipeline above: maintained by [@ravediamond](https://github.com/ravediamond), manually kept in sync with [`ravediamond/lerobot-jetson`](https://github.com/ravediamond/lerobot-jetson) (the source of truth), where a prebuilt image is also published (`ghcr.io/ravediamond/lerobot-jetson`). See [#819](https://github.com/huggingface/lerobot/issues/819) for background.

```bash
docker build -f docker/Dockerfile.jetson -t lerobot-jetson .
docker run -it --rm --runtime nvidia lerobot-jetson
```

## Usage

### Running a pre-built image

```bash
# CPU
docker run -it --rm huggingface/lerobot-cpu:latest

# GPU
docker run -it --rm --gpus all --shm-size 16gb huggingface/lerobot-gpu:latest
```

### Building locally

From the repo root:

```bash
# CPU
docker build -f docker/Dockerfile.user -t lerobot-user .
docker run -it --rm lerobot-user

# GPU
docker build -f docker/Dockerfile.internal -t lerobot-internal .
docker run -it --rm --gpus all --shm-size 16gb lerobot-internal
```

### Multi-GPU training

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` when launching the container:

```bash
# Use 4 GPUs
docker run -it --rm --gpus all --shm-size 16gb \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  huggingface/lerobot-gpu:latest
```

### USB device access (e.g. robots, cameras)

```bash
docker run -it --device=/dev/ -v /dev/:/dev/ --rm huggingface/lerobot-cpu:latest
```

---

# ローカル PC 用 pi0.5 (pi05) 開発環境 (`docker/image/`)

`Dockerfile.internal` と同じ構成 (CUDA 12.8 / Ubuntu 24.04 / Python 3.12 / uv / `uv.lock` 固定) で、
venv をリポジトリ外 (`/opt/venv`) に置き、ホストのリポジトリを `/opt/lerobot` に bind mount して使う
(`apptainer/image/pi05.def` と同じ設計。コードを変えても再 build 不要)。ホストと同じ UID/GID で動くので
`outputs/` が root 所有にならない。クラスタ (ABCI / Slurm) 側は [`../apptainer/README.md`](../apptainer/README.md)。

```
docker/
├── image/
│   ├── Dockerfile            # CUDA 12.8 / Py3.12 / uv。extras: pi training aloha libero pusht async
│   ├── docker-compose.yml    # GPU, ipc=host, リポジトリと HF キャッシュを mount
│   ├── build.sh              # docker compose build
│   └── run.sh                # コンテナ内でコマンド実行 (無ければ up -d)。`run.sh down` で停止
├── train/train_pi05.sh       # fine-tune
├── eval/eval_pi05.sh         # 成功率評価 (aloha / libero)
└── deploy/deploy_pi05_aloha.sh   # gym-aloha で rollout + 動画保存
```

パラメータ (`HF_TOKEN`, `DATASET`, `BATCH_SIZE` ...) と `env.sh` は `apptainer/common/` と共有する。
学習・評価の引数は `apptainer/common/lerobot_args.sh` にあり、ABCI と同じコマンドが走る。

## 使い方

前提: NVIDIA ドライバ + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)、docker compose >= 2.30。

```bash
# 0. 設定 (初回)。HF_TOKEN は paligemma のライセンス承諾済みのもの
cp apptainer/common/env.sh.example apptainer/common/env.sh && vim apptainer/common/env.sh

# 1. build (10〜20 分)
cd docker/image && ./build.sh

# 2. 動作確認 / 対話 bash
./run.sh nvidia-smi
./run.sh                       # コンテナ内 bash (/opt/lerobot)

# 3. 学習 (RTX A6000 48GB なら VLM 凍結 + 小さい batch)
cd ../train
BATCH_SIZE=8 TRAIN_EXPERT_ONLY=true ./train_pi05.sh
BATCH_SIZE=4 STEPS=20 SAVE_FREQ=20 COMPILE_MODEL=false JOB_NAME=smoke ./train_pi05.sh   # スモークテスト

# 4. 評価 / デプロイ
cd ../eval   && JOB_NAME=smoke EVAL_EPISODES=2 EVAL_BATCH=2 ./eval_pi05.sh
cd ../deploy && JOB_NAME=smoke DEPLOY_EPISODES=2 ./deploy_pi05_aloha.sh
#   -> outputs/deploy/smoke/AlohaTransferCube-v0_<timestamp>/videos/aloha_0/eval_episode_{0,1}.mp4

# 5. 停止
cd ../image && ./run.sh down
```

- GPU を選ぶ: `CUDA_VISIBLE_DEVICES=1 ./train_pi05.sh` (compose の既定は 0)。
- HF キャッシュはホストの `$HF_HOME` (既定 `~/.cache/huggingface`) をそのまま mount する。
- `run.sh` は `HF_TOKEN` / `WANDB_API_KEY` / `CUDA_VISIBLE_DEVICES` を exec ごとにコンテナへ渡す。

### トラブルシューティング

- **pi0.5 で gemma のモデルが DL できない**: HF にログインし
  [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) のライセンスを承諾、
  token を `apptainer/common/env.sh` の `HF_TOKEN` に書く (`huggingface-cli login` は不要)。
- **`useradd: UID 1000 is not unique`**: Dockerfile の `userdel -r ubuntu` 行が消えている。
- **eval / deploy で `Namespace gym_aloha not found`**: 非同期 env の worker で `gym_aloha` が import されない lerobot 側の問題。
  既定の `EVAL_ASYNC=false` (同期 env) のままにする。
- **`QUANTILES normalization mode requires q01 and q99`**: `env.sh` で
  `NORM_MAPPING='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'` を設定。
