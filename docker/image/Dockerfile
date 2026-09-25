# LeRobot pi0.5 (pi05) ローカル開発用 GPU イメージ。docker/Dockerfile.internal と同じ構成
# (CUDA 12.8 / Ubuntu 24.04 / Python 3.12 / uv / uv.lock 固定) だが以下が違う:
#   - venv をリポジトリの外 /opt/venv に置く (UV_PROJECT_ENVIRONMENT)。
#     /opt/lerobot をホストのリポジトリで bind mount してもそのまま動く (apptainer/image/pi05.def と同じ設計)
#   - ホストと同じ UID/GID のユーザで動かす (outputs/ が root 所有にならない)
#   - extras: pi training aloha libero pusht async
# build はリポジトリルートを context にする (docker/image/build.sh 経由を推奨):
#   docker build -f docker/image/Dockerfile --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t lerobot-pi05 .

ARG CUDA_VERSION=12.8.1
ARG OS_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION}

ARG PYTHON_VERSION=3.12
ARG UID=1000
ARG GID=1000
ARG USERNAME=lerobot

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs curl wget ca-certificates sudo \
    vim less htop tmux tree \
    libglib2.0-0 libgl1 libegl1 ffmpeg \
    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
    cmake pkg-config ninja-build \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    # ubuntu24.04 ベースには uid 1000 の "ubuntu" が居るので消してからホスト UID で作る
    && (userdel -r ubuntu 2>/dev/null || true) \
    && (getent group ${GID} >/dev/null || groupadd -g ${GID} ${USERNAME}) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && mkdir -p /opt/lerobot /opt/venv && chown -R ${UID}:${GID} /opt/lerobot /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
ENV HOME=/home/${USERNAME} \
    HF_HOME=/home/${USERNAME}/.cache/huggingface \
    HF_LEROBOT_HOME=/home/${USERNAME}/.cache/huggingface/lerobot \
    TORCH_HOME=/home/${USERNAME}/.cache/torch \
    TRITON_CACHE_DIR=/home/${USERNAME}/.cache/triton
WORKDIR /opt/lerobot

RUN uv venv --python python${PYTHON_VERSION} /opt/venv \
    && git config --global --add safe.directory '*'

# 依存だけ先に入れてレイヤキャッシュを効かせる
COPY --chown=${UID}:${GID} setup.py pyproject.toml uv.lock README.md MANIFEST.in ./
RUN uv sync --locked --no-install-project --no-cache \
    --extra pi --extra training --extra aloha --extra libero --extra pusht --extra async \
    && (chmod +x /opt/venv/lib/python${PYTHON_VERSION}/site-packages/triton/backends/nvidia/bin/ptxas || true)

# 本体 (editable install)。実行時は同じパスにホストのリポジトリを bind するので、
# ここでの COPY はイメージ単体でも動くようにするため
COPY --chown=${UID}:${GID} src ./src
RUN uv sync --locked --no-cache \
    --extra pi --extra training --extra aloha --extra libero --extra pusht --extra async

CMD ["sleep", "infinity"]
