# Apptainer / Singularity で pi0 を fine-tune する

`docker/Dockerfile.internal` (HF 公式 GPU 環境) と同じ構成
(CUDA 12.8 / Ubuntu 24.04 / Python 3.12 / uv / `uv.lock` 固定) を
1 つの SIF に固めて、研究室の Slurm クラスタと **AIST ABCI 3.0** の両方で動かす。

```
apptainer/
├── pi0.def              # イメージ定義
├── run.sh               # コンテナ内でコマンドを実行する共通ラッパー (apptainer / singularity 自動判定)
├── singularity_run.sh   # 研究室 Slurm 用 (run_40gb.sh / run_80gb.sh / send_job.sh から呼ぶ)
└── abci/
    ├── env.sh.example   # → cp して env.sh を作る (グループ名・HF_TOKEN・パス等。git-ignore 済み)
    ├── submit.sh        # qsub ラッパー
    ├── train_pi0.sh     # rt_HG (H200 x1) で学習
    ├── train_pi0_8gpu.sh# rt_HF (H200 x8) で DDP 学習
    ├── eval_pi0.sh      # rt_HG で eval
    └── build_on_abci.sh # ABCI 上で SIF を build する場合
```

## イメージの設計

- リポジトリは `/opt/lerobot` に editable install、venv は `/opt/venv`。
- `run.sh` は手元のリポジトリを `/opt/lerobot` に bind するので、
  **コードを変えても SIF の作り直しは不要**。依存 (`pyproject.toml` / `uv.lock`) を変えたときだけ再 build。
- `outputs/` はリポジトリ直下 (= bind 先) に書かれるのでホスト側にそのまま残る。
- 入っている extra: `pi` `training` `libero` `pusht`。増やしたいときは `pi0.def` の `uv sync` 行を編集。

## 1. SIF を build する

### ローカル (推奨: 手元の方が速く、失敗しても切り分けやすい)

```bash
cd apptainer
apptainer build --fakeroot pi0.sif pi0.def      # 20〜40 分、10 GB 前後
apptainer test pi0.sif                          # import 確認
./run.sh nvidia-smi                             # GPU が見えるか
```

できた `pi0.sif` を ABCI のグループ領域へ送る (ホームの quota を圧迫しないため):

```bash
rsync -avP pi0.sif abci:/groups/<ABCI_GROUP>/<user>/sif/
```

### ABCI 上で build する場合

ログインノードでは build しない。CPU ノードの対話ジョブで `build_on_abci.sh` を実行する:

```bash
qsub -I -P <ABCI_GROUP> -q rt_HC -l select=1 -l walltime=2:00:00
cd ~/lerobot/apptainer/abci && ./build_on_abci.sh
```

## 2. ABCI で学習する

```bash
# ABCI ログインノードで
git clone <this repo> ~/lerobot && cd ~/lerobot/apptainer/abci
cp env.sh.example env.sh && vim env.sh    # ABCI_GROUP, HF_TOKEN, HF_HOME, LEROBOT_SIF, DATASET ...

./submit.sh train_pi0.sh                  # H200 x1
./submit.sh train_pi0_8gpu.sh             # H200 x8 (1 ノード, torchrun DDP)
WALLTIME=72:00:00 ./submit.sh train_pi0.sh

qstat -u $USER                            # 状態確認
tail -f logs/pi0_train.o<jobid>           # ログ
```

学習結果は `~/lerobot/outputs/train/<JOB_NAME>/` に出る。

### eval

```bash
CKPT=outputs/train/pi0_finetune/checkpoints/last/pretrained_model ENV_TYPE=libero \
  ./submit.sh eval_pi0.sh
```

### 対話的に試す

```bash
qsub -I -P <ABCI_GROUP> -q rt_HG -l select=1 -l walltime=1:00:00
module load singularitypro
cd ~/lerobot/apptainer && source abci/env.sh
./run.sh                                  # コンテナ内 bash
./run.sh lerobot-train --help
```

### ABCI 3.0 メモ

| 項目 | 内容 |
| --- | --- |
| スケジューラ | PBS Pro (`qsub` / `qstat` / `qdel`)。Slurm ではない |
| 資源タイプ | `rt_HG`: H200 x1 (学習・eval)、`rt_HF`: 1 ノード H200 x8、`rt_HC`: CPU のみ (build 用) |
| コンテナ | `module load singularitypro` で `singularity` コマンド。`run.sh` が自動判定 |
| ストレージ | `/home/<user>` (小さい)、`/groups/<group>` (大容量、ここに SIF と HF キャッシュ)、`$PBS_LOCALDIR` (ノードローカル NVMe、ジョブ終了で消える。`TMPDIR` に使用) |
| 環境変数 | `submit.sh` が `-V` で渡す。コンテナには `run.sh` が `HF_TOKEN` `WANDB_API_KEY` `HF_HOME` などを `--env` で渡す |

キュー名・module 名・quota は変わることがあるので、最終的には
[ABCI 3.0 User Guide](https://docs.abci.ai/v3/) を確認すること。

## 3. 研究室 Slurm クラスタで使う (従来通り)

```bash
cd apptainer
cp abci/env.sh.example abci/env.sh && vim abci/env.sh   # HF_TOKEN など (ABCI_GROUP は無視される)
sh send_job.sh                                          # sbatch -p part_80gb run_80gb.sh
```

`singularity_run.sh` の学習コマンドは `abci/train_pi0.sh` と同じにしてある。

## トラブルシューティング

- **paligemma / gemma が落ちてこない**: HF で
  [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) のライセンスを承諾し、
  `HF_TOKEN` を `env.sh` に書く。
- **OOM**: `BATCH_SIZE` を下げる。`--policy.train_expert_only=true` を足すと VLM を凍結して軽くなる。
- **ホームの quota 超過**: `HF_HOME` と `LEROBOT_SIF` をグループ領域にする (env.sh.example の通り)。
- **`lerobot-train` refuses to start (accelerate env)**: ジョブスクリプトが `-V` でホスト環境を全部渡すので、
  ログインシェルで `ACCELERATE_*` を export していたら消す。
