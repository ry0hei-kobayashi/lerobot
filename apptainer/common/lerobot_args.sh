#!/bin/bash
# lerobot-train / lerobot-eval の引数を組み立てる共通関数。
# apptainer/{train,eval,deploy} と docker/{train,eval,deploy} の両方から source され、
# 学習・評価の中身をこの 1 ファイルで管理する (ABCI / Slurm / docker で同じコマンドが走る)。
#
#   source "$COMMON/lerobot_args.sh"
#   lerobot_train_args [job_name_suffix]   # -> 配列 LEROBOT_ARGS
#   lerobot_eval_args                      # -> LEROBOT_ARGS と LEROBOT_OUTPUT_DIR
#   lerobot_deploy_args                    # -> LEROBOT_ARGS と LEROBOT_OUTPUT_DIR
#   run.sh lerobot-train "${LEROBOT_ARGS[@]}"
#
# パラメータはすべて環境変数 (common/env.sh.example 参照)。ここにあるのは最終デフォルト。

# ---- デフォルト ----
: "${POLICY_TYPE:=pi05}"
: "${PRETRAINED:=lerobot/pi05_base}"
: "${DATASET:=lerobot/aloha_sim_transfer_cube_human}"
: "${JOB_NAME:=${POLICY_TYPE}_aloha_sim}"
: "${BATCH_SIZE:=32}"
: "${STEPS:=20000}"
: "${SAVE_FREQ:=5000}"
: "${NUM_WORKERS:=8}"
: "${SEED:=1000}"
: "${N_ACTION_STEPS:=10}"
: "${EMPTY_CAMERAS:=0}"
: "${TRAIN_EXPERT_ONLY:=false}"
: "${FREEZE_VISION_ENCODER:=false}"
: "${COMPILE_MODEL:=true}"
: "${NORM_MAPPING:=}"
: "${WANDB_ENABLE:=false}"
: "${EXTRA_TRAIN_ARGS:=}"
: "${CKPT:=outputs/train/$JOB_NAME/checkpoints/last/pretrained_model}"
: "${EVAL_ENV:=aloha}"
: "${EVAL_TASK:=}"
: "${EVAL_EPISODES:=50}"
: "${EVAL_BATCH:=10}"
: "${EVAL_ASYNC:=false}"
: "${EVAL_COMPILE_MODEL:=false}"   # 推論で torch.compile を使うか。H200 で illegal memory access が出るので既定 off
: "${EVAL_TASK_DESCRIPTION:=}"     # 空なら aloha は dataset と同じ文を lerobot_aloha_task_description で補う
: "${EXTRA_EVAL_ARGS:=}"
: "${DEPLOY_TASK:=AlohaTransferCube-v0}"
: "${DEPLOY_EPISODES:=10}"
: "${EXTRA_DEPLOY_ARGS:=}"

lerobot_require_token() {
    [ -n "${HF_TOKEN:-}" ] && [ "$HF_TOKEN" != "hf_xxxxxxxxxxxxxxxxxxxxxxxx" ] \
        || { echo "ERROR: HF_TOKEN を設定してください (apptainer/common/env.sh 参照)" >&2; exit 1; }
}

# 学習。$1 は JOB_NAME に付ける suffix (8gpu 版は "_8gpu")
lerobot_train_args() {
    local job="${JOB_NAME}${1:-}"
    LEROBOT_ARGS=(
        --policy.type="$POLICY_TYPE"
        --policy.pretrained_path="$PRETRAINED"        # 重みだけ読む。設定は下で明示する
        --dataset.repo_id="$DATASET"
        --output_dir="outputs/train/$job"
        --job_name="$job"
        --policy.device=cuda
        --policy.dtype=bfloat16
        --policy.gradient_checkpointing=true
        --policy.compile_model="$COMPILE_MODEL"
        --policy.n_action_steps="$N_ACTION_STEPS"
        --policy.empty_cameras="$EMPTY_CAMERAS"
        --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER"
        --policy.train_expert_only="$TRAIN_EXPERT_ONLY"
        --policy.push_to_hub=false                    # PreTrainedConfig の default は true
        --batch_size="$BATCH_SIZE"
        --num_workers="$NUM_WORKERS"
        --steps="$STEPS"
        --save_freq="$SAVE_FREQ"
        --seed="$SEED"
        --wandb.enable="$WANDB_ENABLE"
    )
    [ -n "$NORM_MAPPING" ] && LEROBOT_ARGS+=(--policy.normalization_mapping="$NORM_MAPPING")
    # shellcheck disable=SC2206
    [ -n "$EXTRA_TRAIN_ARGS" ] && LEROBOT_ARGS+=($EXTRA_TRAIN_ARGS)
    return 0
}

# compile_model フィールドを持つ policy だけに --policy.compile_model を渡す (act などは弾かれる)
lerobot_compile_arg() {
    case "$POLICY_TYPE" in
        pi0|pi05|pi0_fast|smolvla|diffusion|molmoact2|flux3) echo "--policy.compile_model=$EVAL_COMPILE_MODEL" ;;
        *) echo "" ;;
    esac
}

# gym-aloha は task_description を持たず、そのままだと policy への指示文が "transfer cube" (gym id) になる。
# 学習 dataset (lerobot/aloha_sim_*_human) の文と同じものを返す
lerobot_aloha_task_description() {
    case "$1" in
        AlohaTransferCube-v0) echo "Pick up the cube with the right arm and transfer it to the left arm." ;;
        AlohaInsertion-v0)    echo "Insert the peg into the socket." ;;
        *) echo "" ;;
    esac
}

# 評価。--policy.path は重み + config.json を読むので --policy.type は付けない
lerobot_eval_args() {
    local task="$EVAL_TASK"
    if [ -z "$task" ]; then
        case "$EVAL_ENV" in
            aloha)  task=AlohaTransferCube-v0 ;;
            libero) task=libero_object ;;
        esac
    fi
    local desc="$EVAL_TASK_DESCRIPTION"
    [ -z "$desc" ] && [ "$EVAL_ENV" = aloha ] && desc="$(lerobot_aloha_task_description "$task")"
    LEROBOT_OUTPUT_DIR="outputs/eval/${JOB_NAME}/${EVAL_ENV}_${task}"
    LEROBOT_ARGS=(
        --policy.path="$CKPT"
        --policy.device=cuda
        --env.type="$EVAL_ENV"
        --env.task="$task"
        --eval.n_episodes="$EVAL_EPISODES"
        --eval.batch_size="$EVAL_BATCH"
        --eval.use_async_envs="$EVAL_ASYNC"        # aloha は forkserver の worker で gym_aloha が import されず落ちるので同期 env
        --output_dir="$LEROBOT_OUTPUT_DIR"
        --job_name="${JOB_NAME}_eval"
        --seed="$SEED"
    )
    [ -n "$desc" ] && LEROBOT_ARGS+=(--env.task_description="$desc")   # 学習時と同じ指示文を policy に渡す
    local comp; comp="$(lerobot_compile_arg)"; [ -n "$comp" ] && LEROBOT_ARGS+=("$comp")   # checkpoint の compile_model=true を上書き
    # shellcheck disable=SC2206
    [ -n "$EXTRA_EVAL_ARGS" ] && LEROBOT_ARGS+=($EXTRA_EVAL_ARGS)
    return 0
}

# deploy = gym-aloha で少数 episode を rollout し全 episode の動画を保存する。
# lerobot-eval は先頭 10 episode までしか動画を書かないので DEPLOY_EPISODES<=10、
# batch_size = n_episodes で 1 batch にまとめる。動画: <output_dir>/videos/aloha_0/eval_episode_N.mp4
lerobot_deploy_args() {
    local stamp; stamp="$(date +%Y%m%d_%H%M%S)"
    local desc="$EVAL_TASK_DESCRIPTION"
    [ -z "$desc" ] && desc="$(lerobot_aloha_task_description "$DEPLOY_TASK")"
    LEROBOT_OUTPUT_DIR="outputs/deploy/${JOB_NAME}/${DEPLOY_TASK}_${stamp}"
    LEROBOT_ARGS=(
        --policy.path="$CKPT"
        --policy.device=cuda
        --env.type=aloha
        --env.task="$DEPLOY_TASK"
        --eval.n_episodes="$DEPLOY_EPISODES"
        --eval.batch_size="$DEPLOY_EPISODES"
        --eval.use_async_envs="$EVAL_ASYNC"
        --output_dir="$LEROBOT_OUTPUT_DIR"
        --job_name="${JOB_NAME}_deploy"
        --seed="$SEED"
    )
    [ -n "$desc" ] && LEROBOT_ARGS+=(--env.task_description="$desc")
    local comp; comp="$(lerobot_compile_arg)"; [ -n "$comp" ] && LEROBOT_ARGS+=("$comp")   # checkpoint の compile_model=true を上書き
    # shellcheck disable=SC2206
    [ -n "$EXTRA_DEPLOY_ARGS" ] && LEROBOT_ARGS+=($EXTRA_DEPLOY_ARGS)
    return 0
}
