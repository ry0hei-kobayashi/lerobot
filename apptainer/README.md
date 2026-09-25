# Apptainer / Singularity で pi0.5 (pi05) を fine-tune・評価・デプロイする

`docker/Dockerfile.internal` (HF 公式 GPU 環境) と同じ構成
(CUDA 12.8 / Ubuntu 24.04 / Python 3.12 / uv / `uv.lock` 固定) を 1 つの SIF に固めて、
研究室の Slurm クラスタと **AIST ABCI 3.0** の両方で動かす。
ポリシーは **π₀.₅ (`--policy.type=pi05`, `lerobot/pi05_base`)**。
デプロイ先は **gym-aloha (MuJoCo ALOHA 2 シミュレーション)** で、fine-tune した checkpoint を rollout して動画を保存する。
ローカル PC (docker) 版は [`../docker/README.md`](../docker/README.md)。

```
apptainer/
├── image/
│   ├── pi05.def              # イメージ定義 (extras: pi training aloha libero pusht async)
│   ├── build.sh              # ローカル build (--runscript-only / --sandbox あり)
│   └── build_on_abci.sh      # ABCI 上で build する場合 (ログインノードで実行)
├── common/
│   ├── run.sh                # コンテナ内でコマンドを実行する共通ラッパー (apptainer / singularity 自動判定)
│   ├── env.sh.example        # → cp して env.sh を作る (HF_TOKEN・ABCI グループ・学習パラメータ。git-ignore 済み)
│   ├── lerobot_args.sh       # lerobot-train / lerobot-eval の引数を組む共通関数 (train/eval/deploy と docker/ が共有)
│   └── submit.sh             # qsub (ABCI) / sbatch (研究室 Slurm) 自動判定のジョブ投入ラッパー
├── train/
│   ├── train_pi05.sh         # ABCI rt_HG (H200 x1) で fine-tune
│   ├── train_pi05_8gpu.sh    # ABCI rt_HF (H200 x8) で DDP (torchrun)
│   └── slurm_train_pi05.sh   # 研究室 Slurm (A100 80GB) で fine-tune
├── eval/
│   └── eval_pi05.sh          # lerobot-eval で成功率を測る (aloha / libero)
├── deploy/
│   └── deploy_pi05_aloha.sh  # gym-aloha で rollout して全 episode の動画を保存
└── logs/                     # submit.sh が作る (git-ignore 済み)
```

## イメージの設計

- リポジトリは `/opt/lerobot` に editable install、venv は `/opt/venv`。
- `common/run.sh` は手元のリポジトリを `/opt/lerobot` に bind するので、
  **コードを変えても SIF の作り直しは不要**。依存 (`pyproject.toml` / `uv.lock`) を変えたときだけ再 build。
- `outputs/` はリポジトリ直下 (= bind 先) に書かれるのでホスト側にそのまま残る。
- 入っている extra: `pi` (pi0 / pi05) `training` `aloha` `libero` `pusht` `async`。増やしたいときは `image/pi05.def` の `uv sync` 行を編集。
- `%files` の相対パスは build 時の cwd 基準なので、build は必ず `image/` で行う (`build.sh` が cd する)。

## 1. SIF を build する

### ローカル (推奨: 手元の方が速く、失敗しても切り分けやすい)

```bash
cd apptainer/image
./build.sh                    # 20〜40 分、5〜10 GB。最後に apptainer test が走る
../common/run.sh nvidia-smi   # GPU が見えるか
```

できた `pi05.sif` を ABCI のグループ領域へ送る (ホームの quota を圧迫しないため):

```bash
rsync -avP pi05.sif abci:/groups/<ABCI_GROUP>/<user>/sif/
```

### ABCI 上で build する場合

SIF の作成は **ログインノード** で行う (計算ジョブは不要)。`image/build_on_abci.sh` を実行する:

```bash
cd ~/lerobot/apptainer/image && ./build_on_abci.sh
```

一時領域とキャッシュはホーム quota を圧迫しないよう `/groups/<ABCI_GROUP>/<user>/` 配下に置き、
出来上がった SIF も `/groups/<ABCI_GROUP>/<user>/sif/pi05.sif` に書く (`LEROBOT_SIF` で変更可)。
build は 20〜40 分かかるので、SSH が切れても続くよう `nohup ./build_on_abci.sh > build.log 2>&1 &` や `tmux` の中で実行するとよい。

## 2. 設定 (`common/env.sh`)

```bash
cd apptainer/common
cp env.sh.example env.sh && vim env.sh    # HF_TOKEN, ABCI_GROUP, DATASET, JOB_NAME, BATCH_SIZE ...
```

- すべて `export VAR="${VAR:-default}"` 形式なので、優先順位は **コマンドライン > env.sh > lerobot_args.sh の既定値**。
  例: `BATCH_SIZE=8 STEPS=20 ./submit.sh ../train/train_pi05.sh`
- `/groups/$ABCI_GROUP` が存在すれば (= ABCI) `HF_HOME` と `LEROBOT_SIF` をグループ領域に、
  無ければ `~/.cache/huggingface` と `image/pi05.sif` を使う。
- `DATASET` と `DEPLOY_TASK` / `EVAL_TASK` は対にする:
  `lerobot/aloha_sim_transfer_cube_human` ↔ `AlohaTransferCube-v0`、`lerobot/aloha_sim_insertion_human` ↔ `AlohaInsertion-v0`。

## 3. 学習

```bash
cd apptainer/common
./submit.sh ../train/train_pi05.sh                    # ABCI rt_HG (H200 x1)
./submit.sh ../train/train_pi05_8gpu.sh               # ABCI rt_HF (H200 x8, torchrun DDP)
WALLTIME=72:00:00 ./submit.sh ../train/train_pi05.sh
./submit.sh ../train/slurm_train_pi05.sh              # 研究室 Slurm (part_80gb, A100 80GB)
./submit.sh ../train/slurm_train_pi05.sh -p part_40gb --gres=gpu:a100_3g.40gb:1   # 40GB MIG

qstat -u $USER                                        # ABCI (PBS)
squeue -u $USER                                       # 研究室 Slurm
tail -f ../logs/pi05_train.o<jobid>                   # ABCI のログ
tail -f ../logs/pi05_train-<jobid>.out                # Slurm のログ
```

学習結果は `~/lerobot/outputs/train/<JOB_NAME>/` (8 GPU 版は `<JOB_NAME>_8gpu`) に出る。
ジョブスクリプトは対話ジョブやローカル apptainer から直接 `./train_pi05.sh` と実行してもよい。

## 4. 評価 (成功率)

```bash
cd apptainer/common
./submit.sh ../eval/eval_pi05.sh                                          # env.sh の CKPT / EVAL_ENV (既定: last checkpoint, aloha)
CKPT=outputs/train/pi05_aloha_sim/checkpoints/010000/pretrained_model ./submit.sh ../eval/eval_pi05.sh
EVAL_ENV=libero EVAL_TASK=libero_spatial EVAL_BATCH=1 ./submit.sh ../eval/eval_pi05.sh
```

結果: `outputs/eval/<JOB_NAME>/<env>_<task>/eval_info.json` (+ `videos/` に先頭 10 episode の mp4)。

## 5. デプロイ (gym-aloha で rollout)

```bash
cd apptainer/common
./submit.sh ../deploy/deploy_pi05_aloha.sh
CKPT=... DEPLOY_TASK=AlohaInsertion-v0 DEPLOY_EPISODES=5 ./submit.sh ../deploy/deploy_pi05_aloha.sh
```

出力: `outputs/deploy/<JOB_NAME>/<task>_<timestamp>/{eval_info.json, videos/aloha_0/eval_episode_N.mp4}`。
`lerobot-eval` が動画を書くのは先頭 10 episode までなので `DEPLOY_EPISODES` は 10 以下にする。
手元に回収するには `rsync -avP abci:~/lerobot/outputs/deploy/<JOB_NAME>/ outputs/deploy/<JOB_NAME>/`。

## 6. 対話的に試す

```bash
qsub -I -P <ABCI_GROUP> -q rt_HG -l select=1 -l walltime=1:00:00
module load singularitypro
cd ~/lerobot/apptainer/common && source env.sh
./run.sh                                  # コンテナ内 bash
./run.sh lerobot-train --help
```

## pi0.5 メモ

| 項目                                          | 内容                                                                                                                                                                                     |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--policy.pretrained_path` vs `--policy.path` | 学習は `pretrained_path` (重みのみ。`n_action_steps` 等は明示が必要)。eval / deploy は `path` (重み + config.json。`--policy.type` は付けない)                                           |
| `N_ACTION_STEPS=10`                           | aloha は 50fps、`chunk_size=50` (1 秒) のうち 10 step (0.2 秒) ごとに再推論。openpi の aloha_sim 例と同じ                                                                                |
| `EMPTY_CAMERAS=0`                             | aloha sim はカメラ 1 台 (`observation.images.top`)。env と dataset のキーが一致するので `--rename_map` も不要                                                                            |
| 正規化                                        | pi05 の既定は QUANTILES。`ValueError: QUANTILES normalization mode requires q01 and q99` が出る dataset は `NORM_MAPPING='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'` |
| `--policy.push_to_hub=false`                  | `PreTrainedConfig` の既定が true なので常に明示                                                                                                                                          |
| 省メモリ                                      | `TRAIN_EXPERT_ONLY=true` (VLM 凍結) / `BATCH_SIZE` を下げる。48GB クラスは `BATCH_SIZE=4-8 TRAIN_EXPERT_ONLY=true`                                                                       |
| pi0 に戻す                                    | `POLICY_TYPE=pi0 PRETRAINED=lerobot/pi0_base`                                                                                                                                            |

## ABCI 3.0 メモ

| 項目         | 内容                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| スケジューラ | PBS Pro (`qsub` / `qstat` / `qdel`)。Slurm ではない                                                                                                          |
| 資源タイプ   | `rt_HG`: H200 x1 (学習・eval・deploy)、`rt_HF`: 1 ノード H200 x8、SIF の build はログインノードで行う (計算ノードは使わない)                                                               |
| コンテナ     | `module load singularitypro` で `singularity` コマンド。`common/run.sh` が自動判定                                                                           |
| ストレージ   | `/home/<user>` (小さい)、`/groups/<group>` (大容量、ここに SIF と HF キャッシュ)、`$PBS_LOCALDIR` (ノードローカル NVMe、ジョブ終了で消える。`TMPDIR` に使用) |
| 環境変数     | `submit.sh` が `-V` で渡す。コンテナには `run.sh` が `HF_TOKEN` `WANDB_API_KEY` `HF_HOME` などを `--env` で渡す                                              |

キュー名・module 名・quota は変わることがあるので、最終的には
[ABCI 3.0 User Guide](https://docs.abci.ai/v3/) を確認すること。

## トラブルシューティング

- **paligemma / gemma が落ちてこない**: HF で
  [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) のライセンスを承諾し、
  `HF_TOKEN` を `common/env.sh` に書く。
- **OOM**: `BATCH_SIZE` を下げる。`TRAIN_EXPERT_ONLY=true` で VLM を凍結すると軽くなる。
- **`QUANTILES normalization mode requires q01 and q99`**: 上の pi0.5 メモの `NORM_MAPPING` を設定する。
- **eval / deploy で `Namespace gym_aloha not found` → `BrokenPipeError`**: `AsyncVectorEnv` (forkserver) の worker で `gym_aloha` が
  import されない lerobot 側の問題。スクリプトは既定で `EVAL_ASYNC=false` (同期 env) にしてある。
- **ホームの quota 超過**: `HF_HOME` と `LEROBOT_SIF` をグループ領域にする (env.sh.example の通り)。
- **`lerobot-train` refuses to start (accelerate env)**: ジョブスクリプトが `-V` でホスト環境を全部渡すので、
  ログインシェルで `ACCELERATE_*` を export していたら消す。
