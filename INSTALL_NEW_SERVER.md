# 在全新服务器上安装、训练和推理

本文是 semantic-goal MVP（Goal1/Goal3）的单一入口，面向一台没有本仓库、数据、模型权重
或 Python 环境的新 Linux 服务器。按顺序执行即可。方法定义仍以
[`SEMANTIC_GOAL_MVP.md`](SEMANTIC_GOAL_MVP.md) 为准，资源哈希的权威记录仍以
[`RESOURCE_SETUP.md`](RESOURCE_SETUP.md) 为准。

## 0. 范围与资源预算

训练和推理使用同一套 Python 3.10 环境，但仿真器只在 closed-loop rollout 时需要。

| 用途 | 最低硬件/空间建议 |
|---|---|
| 代码与单元测试 | Linux x86-64、Python 3.10、约 10 GiB |
| 两卡开发 smoke | 2×A40 48GB，至少 50 GiB 空闲空间 |
| 两卡正式训练 | 2×H200 141GB，单个实验至少 150 GiB 空闲空间 |
| 单个正式训练 | 单节点 4×A100 80GB，至少 150 GiB 空闲空间 |
| 同时保留 Goal1 和 Goal3 全部默认 checkpoint | 至少 300 GiB；建议预留 350 GiB |
| RoboCasa closed-loop 推理 | NVIDIA GPU、可用 EGL；另需约 20–25 GiB 场景资产 |

正式训练默认每 500 steps 保存一次、最多保留 7 个 checkpoint。一个完整训练 checkpoint
约 16 GiB，其中包含约 7.1 GiB 模型权重和约 8 GiB optimizer state；训练结束还会在输出目录
根部写一份约 7.1 GiB 的最终模型。因此 README 中“外部资源约 12 GB”不是正式训练的总空间。

开始前应确认：

```bash
nvidia-smi
df -h /path/to/workspace
git --version
curl --version
tar --version
```

服务器还需要可用的 C/C++ 编译工具和 `python3.10`/`python3.10-venv`。如果 HPC 使用 module、
conda 或预装 CUDA，请先加载相应环境。PyTorch wheel 自带 CUDA runtime，但 NVIDIA driver
必须与它兼容。

## 1. 定义路径并克隆本仓库

不要把下面的路径留成占位符。所有训练必须在单个节点内完成；Slurm/PBS 只负责先分配节点，
仓库 launcher 会在节点内用 `torchrun` 创建每个 GPU 的进程。

```bash
export FLF_WORK_ROOT=/absolute/path/to/flf_work
export FLF_RESOURCE_ROOT=/absolute/path/to/stage_state_resources
export FLF_REPO_ROOT="$FLF_WORK_ROOT/flf_progress_vla"

mkdir -p "$FLF_WORK_ROOT" "$FLF_RESOURCE_ROOT"
git clone https://github.com/zhiyuan-gao/flf_progress_vla.git "$FLF_REPO_ROOT"
cd "$FLF_REPO_ROOT"
```

在 semantic-goal PR 合并到 `main` 前，需要显式切换分支：

```bash
git switch feature/semantic-subgoal-goal-images
```

确认以下两个文件存在；否则说明检出的不是 semantic-goal 版本：

```bash
test -f configs/semantic_goal1.json
test -f configs/semantic_goal3.json
git status --short --branch
```

## 2. 安装固定 GR00T 和 Python 环境

先让仓库脚本克隆经过验证的 RoboCasa Isaac-GR00T fork。脚本会检出 commit
`9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`：

```bash
cd "$FLF_REPO_ROOT"
./scripts/bootstrap_gr00t.sh
```

创建 Python 3.10 环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

安装 GR00T 和本仓库的训练、测试依赖：

```bash
python -m pip install -e 'external/Isaac-GR00T[base]'

# GR00T 的 base extra 带入 TensorFlow 2.15，但本项目与 RoboCasa365 使用 NumPy 2.2.5，
# 且训练/推理均不使用 TensorFlow。移除它以避免 NumPy 版本冲突。
python -m pip uninstall -y tensorflow tensorflow-estimator keras || true

python -m pip install -e '.[train,dev]'
python -m pip install 'huggingface-hub==0.36.2' 'imageio-ffmpeg==0.6.0'
```

关键版本应为：

```bash
python - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "transformers": "4.51.3",
    "accelerate": "1.14.0",
    "numpy": "2.2.5",
    "pandas": "2.2.3",
    "pyarrow": "25.0.0",
}
for package, wanted in expected.items():
    actual = version(package)
    assert actual == wanted, (package, actual, wanted)
    print(f"{package}=={actual}")
PY
```

如果平台不能直接安装 PyTorch/PyTorch3D wheel，不要任意升级版本。应先按 HPC 的 CUDA
模块安装 `torch==2.5.1`、`torchvision==0.20.1` 和兼容的 PyTorch3D，再继续执行其余命令。

## 3. 下载四个固定数据集

数据必须是 RoboCasa365 **target-human/composite** 的四个指定 snapshot，不能替换成
`target_only`、Human300、MimicGen 或 pretrain 数据。

```bash
export ROBOCASA365_DATA_ROOT="$FLF_RESOURCE_ROOT/data/robocasa365/v1.0/target/composite"
mkdir -p "$ROBOCASA365_DATA_ROOT"

download_robocasa_task() {
  local task_name="$1"
  local snapshot="$2"
  local box_id="$3"
  local snapshot_dir="$ROBOCASA365_DATA_ROOT/$task_name/$snapshot"
  local archive="$snapshot_dir/lerobot.tar"

  mkdir -p "$snapshot_dir"
  if test -f "$snapshot_dir/lerobot/meta/info.json"; then
    echo "Already present: $task_name $snapshot"
    return 0
  fi
  curl --fail --location --retry 5 --retry-all-errors \
    "https://utexas.box.com/shared/static/${box_id}.tar" \
    --output "${archive}.part"
  mv "${archive}.part" "$archive"
  tar --extract --file "$archive" --directory "$snapshot_dir"
  test -f "$snapshot_dir/lerobot/meta/info.json"
  rm "$archive"
}

download_robocasa_task PreSoakPan 20250809 \
  krqwe33yytnrse06xchr5e41sap41xfv
download_robocasa_task KettleBoiling 20250814 \
  r3rwnzdw6caab3vivwv6uknl9sr8ru6j
download_robocasa_task LoadDishwasher 20250811 \
  k1qxg8rgjs0le1cnv98xylysh9t0dd8z
download_robocasa_task RinseSinkBasin 20250816 \
  blk33oaca11933xgre1sxemmtluo46eo
```

校验 snapshot、episode、frame 和视频数量：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROBOCASA365_DATA_ROOT"])
expected = {
    "PreSoakPan": ("20250809", 501, 395501),
    "KettleBoiling": ("20250814", 501, 228349),
    "LoadDishwasher": ("20250811", 501, 369430),
    "RinseSinkBasin": ("20250816", 509, 211036),
}
for task, (snapshot, episodes, frames) in expected.items():
    dataset = root / task / snapshot / "lerobot"
    info = json.loads((dataset / "meta/info.json").read_text())
    parquet_count = len(list((dataset / "data").glob("*/episode_*.parquet")))
    video_count = len(list((dataset / "videos").glob("*/*/episode_*.mp4")))
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "PandaOmron"
    assert info["total_episodes"] == episodes
    assert info["total_frames"] == frames
    assert info["fps"] == 20
    assert parquet_count == episodes
    assert video_count == episodes * 3
    print(task, "OK", episodes, frames, video_count)
PY
```

## 4. 下载并校验官方父模型

两个 v2 模型都必须从官方 Composite-Seen `checkpoint-60000` 开始，不能从旧的
progress-10k checkpoint 接着训练。

```bash
export GR00T_CHECKPOINT_REPO="$FLF_RESOURCE_ROOT/models/robocasa365_checkpoints"
export GR00T_CHECKPOINT_PREFIX="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"
export GR00T_BASE_CHECKPOINT="$GR00T_CHECKPOINT_REPO/$GR00T_CHECKPOINT_PREFIX"

mkdir -p "$GR00T_CHECKPOINT_REPO"
hf auth login
hf download robocasa/robocasa365_checkpoints \
  "$GR00T_CHECKPOINT_PREFIX/config.json" \
  "$GR00T_CHECKPOINT_PREFIX/experiment_cfg/metadata.json" \
  "$GR00T_CHECKPOINT_PREFIX/model.safetensors.index.json" \
  "$GR00T_CHECKPOINT_PREFIX/model-00001-of-00002.safetensors" \
  "$GR00T_CHECKPOINT_PREFIX/model-00002-of-00002.safetensors" \
  --revision 14895998fe7c8f8f2441cc8957ec2c510302758b \
  --local-dir "$GR00T_CHECKPOINT_REPO"

cd "$GR00T_BASE_CHECKPOINT"
sha256sum -c <<'EOF'
6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526  config.json
0c4a350867f621ed3192f81c282eb7d23c05f275f347fd0deed3531afa29dbcf  experiment_cfg/metadata.json
bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746  model.safetensors.index.json
672b8e49d32ff124e13c3c4e4e70380ab29cd25f013ff41db768639347f8057e  model-00001-of-00002.safetensors
95c52f05a00141ce4e433305a8ccfa41fd219e8d988f2a4c991f3f4f5cdb677c  model-00002-of-00002.safetensors
EOF
```

如果任何一项校验失败，不要训练。上面的 checkpoint 目录总大小应约为 7.1 GiB。

完成校验后必须切回本仓库：

```bash
cd "$FLF_REPO_ROOT"
```

## 5. 写入本机 `.env`

```bash
cp .env.example .env
```

编辑 `.env`，替换成刚才创建的绝对路径：

```text
GR00T_ROOT=/absolute/path/to/flf_work/flf_progress_vla/external/Isaac-GR00T
ROBOCASA365_DATA_ROOT=/absolute/path/to/stage_state_resources/data/robocasa365/v1.0/target/composite
GR00T_BASE_CHECKPOINT=/absolute/path/to/stage_state_resources/models/robocasa365_checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
STAGE_STATE_PYTHON=/absolute/path/to/flf_work/flf_progress_vla/.venv/bin/python
NPROC_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1
```

`CUDA_VISIBLE_DEVICES=0,1` 用于两卡 A40 smoke 或两卡 H200 训练。正式 4×A100 训练前
必须把该行改为：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
```

launcher 会读取 `.env`，所以仅在命令前临时设置同名变量可能会被 `.env` 覆盖。

## 6. 环境、索引、goal cache 和 smoke

以下顺序同时修复了一个容易遗漏的问题：`smoke_semantic_goal.py` 默认读取
`train_smoke1.jsonl`，因此除了完整索引之外，必须显式生成一份 smoke 索引。

```bash
cd "$FLF_REPO_ROOT"
set -a
. ./.env
set +a

"$STAGE_STATE_PYTHON" scripts/check_environment.py \
  --config configs/semantic_goal1.json
"$STAGE_STATE_PYTHON" -m pytest -q
bash -n scripts/launch_semantic_two_a40.sh
bash -n scripts/launch_semantic_four_a100.sh

"$STAGE_STATE_PYTHON" scripts/build_indices.py \
  --config configs/semantic_goal1.json
"$STAGE_STATE_PYTHON" scripts/build_indices.py \
  --config configs/semantic_goal1.json \
  --max-episodes-per-task 1

"$STAGE_STATE_PYTHON" scripts/build_goal_cache.py \
  --config configs/semantic_goal1.json

"$STAGE_STATE_PYTHON" scripts/smoke_semantic_goal.py \
  --config configs/semantic_goal1.json
"$STAGE_STATE_PYTHON" scripts/smoke_semantic_goal.py \
  --config configs/semantic_goal3.json
```

预期结果：

- 单元测试显示 `26 passed`；
- 完整索引为 train 233,808 frames、validation 22,779 frames；
- goal cache 含 train+validation 的 1,650 个 subtask endpoint，约 0.3 GiB；
- Goal1 smoke 显示 4 张 Eagle 图像，Goal3 显示 6 张；
- state mask 有 21 个有效维度，progress slot 为 20。

两张 A40 上再做一个不会保存权重的一步训练 smoke。此时 `.env` 应使用 GPU `0,1`：

```bash
SEMANTIC_GOAL_VARIANT=goal1 \
SEMANTIC_A40_MICRO_BATCH=1 \
scripts/launch_semantic_two_a40.sh \
  --global-batch-size 2 \
  --max-steps 1 \
  --train-index artifacts/semantic_goal_indices/train_smoke1.jsonl \
  --output-dir outputs/semantic_goal1_train_smoke \
  --no-eval \
  --smoke-only

SEMANTIC_GOAL_VARIANT=goal3 \
SEMANTIC_A40_MICRO_BATCH=1 \
scripts/launch_semantic_two_a40.sh \
  --global-batch-size 2 \
  --max-steps 1 \
  --train-index artifacts/semantic_goal_indices/train_smoke1.jsonl \
  --output-dir outputs/semantic_goal3_train_smoke \
  --no-eval \
  --smoke-only
```

只有以上步骤全部通过后才启动 10,000-step 正式训练。

## 7. 在 2×H200 上正式训练

现有 `launch_semantic_two_a40.sh` 只是两进程 launcher，代码不依赖 A40 型号，因此可以直接
用于两张 H200。继续保持 effective global batch 128；先做 100-step、`--no-eval`、
`--smoke-only` profile，并记录稳定阶段的 step time 和 peak GPU memory。

建议首先尝试：

- Goal1：每卡 micro-batch 64，gradient accumulation 1；
- Goal3：每卡 micro-batch 32，gradient accumulation 2。

```bash
cd "$FLF_REPO_ROOT"

SEMANTIC_GOAL_VARIANT=goal1 \
SEMANTIC_A40_MICRO_BATCH=64 \
scripts/launch_semantic_two_a40.sh \
  --max-steps 100 \
  --no-eval \
  --smoke-only \
  --output-dir outputs/semantic_goal1_h200_profile

SEMANTIC_GOAL_VARIANT=goal3 \
SEMANTIC_A40_MICRO_BATCH=32 \
scripts/launch_semantic_two_a40.sh \
  --max-steps 100 \
  --no-eval \
  --smoke-only \
  --output-dir outputs/semantic_goal3_h200_profile
```

如果 OOM，则分别回退到 Goal1 micro-batch 32、Goal3 micro-batch 16；launcher 会自动把
gradient accumulation 调整为 2 和 4，global batch 仍是 128。profile 通过后，移除 profile
参数，以不同输出目录启动两个 10,000-step 正式实验：

```bash
SEMANTIC_GOAL_VARIANT=goal1 \
SEMANTIC_A40_MICRO_BATCH=64 \
scripts/launch_semantic_two_a40.sh

SEMANTIC_GOAL_VARIANT=goal3 \
SEMANTIC_A40_MICRO_BATCH=32 \
scripts/launch_semantic_two_a40.sh
```

两张 H200 的理论性能不能直接换算成真实训练时间；以 100-step profile 的稳定 step time
估计完整运行时间。不要因为 GPU 型号变化而修改 global batch、学习率、训练步数或数据划分。

## 8. 在 4×A100-80GB 上正式训练

先通过调度器申请**一台包含四张 A100 的节点**。不要让调度器启动四份 launcher；每个实验
只启动一次 launcher，它会在该节点内部创建四个 worker。

确认 `.env` 中是：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
```

然后分别运行两个独立实验：

```bash
cd "$FLF_REPO_ROOT"
SEMANTIC_GOAL_VARIANT=goal1 scripts/launch_semantic_four_a100.sh
SEMANTIC_GOAL_VARIANT=goal3 scripts/launch_semantic_four_a100.sh
```

默认设置：

- Goal1：4 GPUs × micro-batch 32 × accumulation 1 = global batch 128；
- Goal3：4 GPUs × micro-batch 16 × accumulation 2 = global batch 128；
- 两者均从官方 checkpoint-60000 开始，训练 10,000 optimizer steps；
- 输出分别写入 `outputs/semantic_goal1_continuation/` 和
  `outputs/semantic_goal3_continuation/`，不会互相覆盖。

如果 Goal3 在目标节点 OOM，保持 global batch 128，只降低 micro-batch：

```bash
SEMANTIC_GOAL_VARIANT=goal3 scripts/launch_semantic_four_a100.sh \
  --per-device-batch-size 8
```

从输出目录中最近的合法 checkpoint 续训：

```bash
SEMANTIC_GOAL_VARIANT=goal1 scripts/launch_semantic_four_a100.sh --resume
```

不要同时把 `--smoke-only` 用于正式训练；它会禁用所有 checkpoint 保存。

## 9. 仅 closed-loop 推理需要：安装 RoboCasa 仿真器

纯训练可以跳过本节。已验证的仿真组合是：

| 组件 | 固定版本 |
|---|---|
| RoboCasa | 1.0.1，commit `b4684e6ee37d377cc392e98302a6b916d588b415` |
| robosuite | 1.5.2，commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| MuJoCo | 3.3.1 |
| Gymnasium | 1.0.0 |

在同一个 `.venv` 中安装，使用 `--no-deps` 避免 RoboCasa 把 GR00T 所需的
`tianshou==0.5.1` 降级成 0.4.10：

```bash
cd "$FLF_REPO_ROOT"
git clone https://github.com/ARISE-Initiative/robosuite.git external/robosuite
git -C external/robosuite checkout --detach \
  5ce6643f3092639d08f7b0f90ed1c6a84f50552c

git clone https://github.com/robocasa/robocasa.git external/robocasa
git -C external/robocasa checkout --detach \
  b4684e6ee37d377cc392e98302a6b916d588b415

python -m pip install -e external/robosuite --no-deps
python -m pip install -e external/robocasa --no-deps
python -m pip install \
  'mujoco==3.3.1' 'gymnasium==1.0.0' \
  'scipy==1.15.3' 'numba==0.61.2' \
  'qpsolvers[quadprog]==4.13.0' \
  'pygame' 'pynput' 'termcolor' 'lxml' 'hidapi' \
  'lerobot==0.3.3' 'tianshou==0.5.1' \
  'imageio-ffmpeg==0.6.0'

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

asset 下载器会要求交互确认，实际解压后约 20–25 GiB。不要把 `external/` 或这些 assets
提交到 Git。

检查版本和 EGL：

```bash
python - <<'PY'
from importlib.metadata import version
import robocasa
import robosuite

assert version("robocasa") == "1.0.1"
assert version("robosuite") == "1.5.2"
assert version("mujoco") == "3.3.1"
print("RoboCasa simulator packages OK")
PY

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

如果集群隐藏或重映射了物理 GPU，`MUJOCO_EGL_DEVICE_ID` 必须与当前
`CUDA_VISIBLE_DEVICES` 中可用于 offscreen rendering 的设备一致。

## 10. GT-reference closed-loop 推理

Goal1：

```bash
cd "$FLF_REPO_ROOT"
set -a
. ./.env
set +a
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

"$STAGE_STATE_PYTHON" scripts/rollout_reference_video.py \
  --config configs/semantic_goal1.json \
  --checkpoint outputs/semantic_goal1_continuation/checkpoint-10000 \
  --task PreSoakPan \
  --episode-split val
```

Goal3 只需要替换 config 和 checkpoint：

```bash
"$STAGE_STATE_PYTHON" scripts/rollout_reference_video.py \
  --config configs/semantic_goal3.json \
  --checkpoint outputs/semantic_goal3_continuation/checkpoint-10000 \
  --task PreSoakPan \
  --episode-split val
```

首次检查仿真环境时，可以加 `--max-steps 1 --no-video --overwrite`，只跑一步并跳过 MP4
编码。正式模型选择期间不要传 `--allow-locked-test`；默认只使用 train/validation episode。

## 11. 最终检查清单

在正式训练或推理前逐项确认：

- 当前 Git checkout 含 `configs/semantic_goal1.json` 和 `configs/semantic_goal3.json`；
- `scripts/check_environment.py --config configs/semantic_goal1.json` 返回 `status: ok`；
- GR00T commit、父 checkpoint 五个 SHA256、四个数据 snapshot 均匹配；
- 完整 `train.jsonl`/`val.jsonl` 和 `train_smoke1.jsonl` 都存在；
- 1,650 个 endpoint 的 goal cache 已生成；
- Goal1/Goal3 multimodal smoke 和对应的一步 DDP smoke 都通过；
- 正式 H200 节点使用两张 GPU，或 A100 节点使用四张 GPU；两个 variant 使用不同输出目录；
- closed-loop rollout 额外安装了固定 RoboCasa/robosuite/MuJoCo 和场景资产，并启用了 EGL。

如果任何版本或哈希不匹配，先停止，不要用“接近的版本”继续正式实验。
