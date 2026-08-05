# 本机有效实验记录与迁移清单

记录日期：2026-08-05（UTC）

本文只记录本机最终有效、可解释、可复现的实验。中止训练、失败的早期评估和已经被完整实验覆盖的 smoke test 不作为科研结论。

## 1. 实验目标与模型

本实验从 RoboCasa365 官方微调后的 GR00T N1.5 checkpoint 继续训练，在 GR00T 原生 64 维 state 输入中原本未使用的前两个 slot 注入：

1. 离散任务阶段 `stage_index`；
2. GT reference video 估计的连续阶段进度 `video_progress`。

父 checkpoint 是：

```text
robocasa/robocasa365_checkpoints
revision: 14895998fe7c8f8f2441cc8957ec2c510302758b
gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
```

GR00T 代码使用 RoboCasa fork：

```text
https://github.com/robocasa-benchmark/Isaac-GR00T.git
commit: 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
package: gr00t 1.1.0
```

本机代码仓库的远端基线 commit 是 `4f5c443`。实验和评估还依赖未提交的本地修改，见第 7 节；仅在新机器重新 clone 远端仓库不能完整复现本机结果。

## 2. 数据与固定划分

使用四个 RoboCasa365 target-human/composite 任务：

| 任务 | 数据 snapshot | 训练轨迹 | 验证轨迹 | 原 test 轨迹 | 新冻结评估轨迹 |
|---|---:|---:|---:|---:|---:|
| PreSoakPan | 20250809 | 100 | 10 | 20 | 50 |
| KettleBoiling | 20250814 | 100 | 10 | 20 | 50 |
| LoadDishwasher | 20250811 | 100 | 10 | 20 | 50 |
| RinseSinkBasin | 20250816 | 100 | 10 | 20 | 50 |

训练/验证/原 test 划分保存在 `configs/robocasa365_four_task_split_100.json`。训练索引包含 233,808 帧，验证索引包含 22,779 帧；训练时采用“先均匀采样任务，再在任务内均匀采样帧”，不是按各任务帧数比例采样。

最终评估使用冻结的 `configs/gt_test50_v1.json`。每个任务 50 个 episode，由原 test20 加上从未进入 train/val/test 的 episode 中确定性选择的 30 个组成。选择算法、episode 编号和源数据元信息哈希都写在该文件中。

```text
gt_test50_v1 aggregate episode-set SHA-256:
7e333eaff88e985e6bfd9772ab2c2303e8a2e920d025e3cbe673e9c70a6ea1d6
```

数据的准确来源、下载方法和目录结构见 `RESOURCE_SETUP.md`。

## 3. 最终有效训练

最终训练从官方 `checkpoint-60000` 重新开始，训练到 10,000 optimizer steps。没有从先前中止的 1k/3k/6k 实验续跑。

| 项目 | 最终设置 |
|---|---|
| GPU | 4 × NVIDIA A100-SXM4-80GB |
| 每卡 batch | 32 |
| gradient accumulation | 1 |
| effective global batch | 128 |
| max steps | 10,000 |
| 优化器 | AdamW (`beta1=0.95`, `beta2=0.999`) |
| 学习率 | 3e-5 |
| 调度 | 5% warmup + cosine decay |
| weight decay | 1e-5 |
| 数值精度 | bfloat16，TF32 开启 |
| 可训练模块 | action projector + diffusion model |
| 冻结模块 | visual backbone + LLM |
| 可训练参数 | 1,068,806,144 / 2,724,163,520 |
| checkpoint 周期 | 每 500 steps，最多保留 7 个 |
| periodic validation | 关闭；最终用独立 simulator rollout 评估 |
| 总训练时间 | 9,951.35 秒（约 2 小时 46 分） |
| 最终 epoch | 5.4735 |

训练 loss 平稳下降：

| step 区间 | 平均 loss |
|---|---:|
| 1–100 | 0.03255 |
| 101–500 | 0.03109 |
| 501–1,000 | 0.02960 |
| 1,001–2,000 | 0.02674 |
| 2,001–4,000 | 0.02247 |
| 4,001–6,000 | 0.01925 |
| 6,001–8,000 | 0.01649 |
| 8,001–10,000 | 0.01539 |

Trainer 报告的全程平均 loss 为 `0.02044084`；step 10 的单次记录为 `0.0338`，step 10,000 为 `0.0179`。这些是训练目标 loss，不能替代 simulator success rate。

最终 checkpoint：

```text
outputs/mvp_continuation/checkpoint-10000
```

该目录约 16 GiB，其中推理必需权重约 7.1 GiB，`optimizer.pt` 约 8.0 GiB。`outputs/mvp_continuation/` 根目录还有一份与 `checkpoint-10000` bitwise 相同的两片模型权重；迁移时不要重复复制两份。

## 4. 最终 simulator evaluation

两组策略都在同一个 `gt_test50_v1` 的 200 个 episode 上运行，并恢复完全相同的 MuJoCo XML 和初始 simulator state。共同设置：

- `execute_steps=16`；
- `denoising_steps=4`；
- RoboCasa v1.0.1 官方任务 horizon；
- 成功判据为环境 `info["success"]`；
- episode policy seed 使用相同确定性算法；
- 12 个常驻 worker，4 张 GPU，每卡 3 个 worker；
- 两组均完成 200/200，运行错误为 0。

### 4.1 我们的方法：checkpoint-10000 + GT-video DTW condition

| 任务 | 成功数 | 成功率 |
|---|---:|---:|
| PreSoakPan | 40/50 | 80% |
| KettleBoiling | 29/50 | 58% |
| LoadDishwasher | 43/50 | 86% |
| RinseSinkBasin | 44/50 | 88% |
| **总计** | **156/200** | **78%** |

批量耗时 1,212.3 秒（约 20.2 分钟）。权威汇总文件：

```text
outputs/reference_video_rollout/gt_test50_v1/batch_summary.json
```

### 4.2 Baseline：官方微调 checkpoint-60000 原生 GR00T

Baseline 直接使用官方 `Gr00tPolicy`。它不构建 GT reference、不运行 DTW，也不向 condition slots 人为写入 0，因此不存在“给条件模型输入全零”的 OOD 问题。

| 任务 | 成功数 | 成功率 |
|---|---:|---:|
| PreSoakPan | 46/50 | 92% |
| KettleBoiling | 34/50 | 68% |
| LoadDishwasher | 37/50 | 74% |
| RinseSinkBasin | 37/50 | 74% |
| **总计** | **154/200** | **77%** |

批量耗时 1,385.5 秒（约 23.1 分钟）。权威汇总文件：

```text
outputs/reference_video_rollout/gt_test50_v1/official_groot_checkpoint60000/batch_summary.json
```

### 4.3 同 episode 配对比较

| 范围 | 我们的方法 | 官方 baseline | 差值 | 仅我们成功 | 仅 baseline 成功 | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 78% | 77% | +1 pp | 33 | 31 | 0.901 |
| PreSoakPan | 80% | 92% | -12 pp | 4 | 10 | 0.180 |
| KettleBoiling | 58% | 68% | -10 pp | 5 | 10 | 0.302 |
| LoadDishwasher | 86% | 74% | +12 pp | 13 | 7 | 0.263 |
| RinseSinkBasin | 88% | 74% | +14 pp | 11 | 4 | 0.118 |

总体配对列联表为：两者都成功 123、仅我们成功 33、仅 baseline 成功 31、两者都失败 13。

结论边界：当前结果没有证明整体成功率显著提升；效果具有明显任务异质性。这个对比衡量的是“继续训练后的条件模型 + GT video”相对“官方微调模型”的整体变化，因为模型权重和条件输入同时改变，不能把差异单独因果归因给 GT video。

## 5. DTW progress 诊断

在同一冻结的 200 个 GT episode 上，把 GT observation video 依次重放给 online DTW localizer：

| 指标 | 结果 |
|---|---:|
| samples | 7,373 |
| stage accuracy | 96.35% |
| task progress MAE | 0.00831 |
| task progress RMSE | 0.03629 |
| task progress within 0.05 | 96.00% |
| transition absolute error | 7.50 control steps |

权威汇总文件：

```text
outputs/reference_video_rollout/gt_test50_v1/dtw_gt_replay_summary.json
```

该结果只说明“在 held-out GT trajectory replay 上，DTW 能否恢复标注进度”。它不是自由 rollout 偏离 demonstration 后的语义进度准确率，也不是策略成功率。

## 6. 哪些 checkpoint 和记录值得保留

### 自研产物：必须迁移或备份

1. `outputs/mvp_continuation/checkpoint-10000/` 中第 7.A 节列出的五个推理文件：唯一经过完整评估的最终自研模型权重；完整目录仅在需要精确续训时保存；
2. `configs/robocasa365_four_task_split_100.json`：逐任务记录 train100、val10 和原 test20 的 episode 编号；
3. `configs/gt_test50_v1.json`：逐任务记录冻结 test50 的 episode 编号、选择算法及数据元信息哈希；
4. 本地工作树代码、`configs/` 和 `artifacts/indices/`；
5. 三个最终汇总 JSON、本文件和 `SELF_METHOD_TRANSFER_MANIFEST.txt`。

### 官方资源：只记录版本，不迁移

- 四任务 RoboCasa365 原始数据可以按 `RESOURCE_SETUP.md` 从官方地址重新下载。本仓库只保存 snapshot、split 中的 episode 编号及源数据哈希，不复制或上传数据。
- 官方 GR00T `composite_seen/checkpoint-60000` 可以从固定 Hugging Face revision 重新下载，不进入自研产物迁移包。
- Isaac-GR00T、Robosuite 和 Python 依赖按本文及 `RESOURCE_SETUP.md` 记录的 commit/version 重装，不迁移 `.venv` 或 `external` 目录。

### 建议保留但可按空间选择

- `outputs/reference_video_rollout/gt_test50_v1/`：约 280 MiB，包含两组每 episode summary、manifest 和 rollout 视频；建议迁移，便于审计和配对分析。
- `outputs/formal_train.log`、`outputs/gt_test50_v1_eval.log`、`outputs/official_groot_baseline_eval.log` 和 `outputs/dtw_gt_replay_eval.log`：体积很小，建议迁移。
- `outputs/mvp_continuation/runs/`：TensorBoard 记录，建议迁移。

### 不需要为通常迁移携带

- `checkpoint-7000`、`7500`、`8000`、`8500`、`9000`、`9500`：每个约 16 GiB，均未做完整 gt_test50 评估；只有研究 checkpoint trajectory 时才需要。
- `outputs/mvp_continuation/` 根目录的模型 shards：与 `checkpoint-10000` 的模型 shards 完全相同，是重复推理权重。
- `outputs/aborted_periodic_eval_step243_*`、`cancelled_6000_step42_*`、`failed_eval_step250_*` 和对应旧日志：只记录早期工程调试，不是有效实验结果。
- `.venv/`：约 7.4 GiB，不建议跨机器复制；应在目标机器重新创建环境。

本次整理没有删除或移动任何 checkpoint、日志或 rollout。

## 7. 自研产物迁移到另一台服务器需要什么

### A. 最小自研模型包（约 7.1 GiB）

必须从 `outputs/mvp_continuation/checkpoint-10000/` 迁移下面五个文件。总大小必须为 **7,586,192,783 bytes（约 7.1 GiB）**：

| 相对于 `checkpoint-10000/` 的路径 | 精确大小（bytes） | 用途 | SHA-256 |
|---|---:|---|---|
| `config.json` | 1,706 | GR00T N1.5 模型结构与 action-head 配置 | `6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526` |
| `model-00001-of-00002.safetensors` | 4,999,367,032 | 自研模型权重 shard 1/2 | `3cd179363c04de14e1a63e24e6c0272674a98bfa2944a7688e606955d73ee2ef` |
| `model-00002-of-00002.safetensors` | 2,586,705,312 | 自研模型权重 shard 2/2 | `fec08c553ce45e566b924949f1ef00dd34d10d90bb718086fe4898ab269801ee` |
| `model.safetensors.index.json` | 104,606 | 参数名到两个权重 shard 的映射 | `bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746` |
| `experiment_cfg/metadata.json` | 14,127 | PandaOmron modality 定义及 state/action normalization statistics | `0c4a350867f621ed3192f81c282eb7d23c05f275f347fd0deed3531afa29dbcf` |

迁移后的目录层级必须保持为：

```text
outputs/mvp_continuation/checkpoint-10000/
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
└── experiment_cfg/
    └── metadata.json
```

两个 `.safetensors` shard、index、config 和 metadata 缺一不可；不要只上传其中一个 shard。上传到目标服务器后，从 `/workspace` 运行：

```bash
grep 'checkpoint-10000' flf_progress_vla/MIGRATION_SHA256SUMS.txt | sha256sum -c -
```

五行都显示 `OK` 才能确认权重包完整且内容与本机完全一致。

不要再复制 `outputs/mvp_continuation/` 根目录中的同名模型 shards；它们与 `checkpoint-10000` 完全相同。

本机已经将这五个文件打包为一个可直接批量复制的归档：

```text
transfer/self_method_checkpoint10000/checkpoint-10000-inference.tar.zst
```

归档信息：

| 项目 | 值 |
|---|---|
| 压缩格式 | tar + zstd |
| 压缩文件大小 | 5,995,314,809 bytes（约 5.6 GiB） |
| 归档 SHA-256 | `4901eed6ff98713935ff954d2e03d42704698e2a3d8f8d0a41ef7718474f563a` |
| 内部文件数 | 5 |
| 是否包含 optimizer | 否 |
| 是否包含官方模型或数据 | 否 |

上传时直接复制整个 `transfer/self_method_checkpoint10000/` 文件夹即可。同目录的 `README.md` 和 `ARCHIVE_SHA256.txt` 给出了目标服务器上的校验与解压命令。压缩归档已加入 `.gitignore`，不会误进入普通 Git history。

### B. 自研代码与实验记录（约 0.4 GiB）

1. 当前 repo 工作树，排除 `.venv/`、`external/` 和大体积 `outputs/`；代码、配置与 indices 约 105 MiB；
2. `configs/robocasa365_four_task_split_100.json` 和 `configs/gt_test50_v1.json`；
3. `outputs/reference_video_rollout/gt_test50_v1/`，约 280 MiB，包含两组逐 episode summary、manifest 和 rollout 视频；
4. 最终训练日志、TensorBoard 记录和三份最终汇总 JSON。

官方 baseline 的 summary 属于我们生成的实验记录，可以保留；官方模型权重本身不复制。

### C. 可选：从 step 10,000 精确续训

如果未来确实要从 step 10,000 续训，再额外保存 `checkpoint-10000/` 中的训练状态：

- `optimizer.pt`；
- `scheduler.pt`；
- `trainer_state.json`；
- `rng_state_0.pth` 至 `rng_state_3.pth`；
- A 中的全部推理文件和 `experiment_cfg/metadata.json`。

还应携带：

- `outputs/mvp_continuation/stage_state_run_config.json`；
- `outputs/mvp_continuation/training_args.bin`；
- `artifacts/indices/train.jsonl`、`val.jsonl` 和 `summary.json`；
- 官方父 checkpoint 不迁移；在新服务器按固定 revision 下载。当前训练入口需要它来构造模型和读取 normalization metadata。

RoboCasa365 数据也不迁移；按固定 snapshot 重新下载后，使用两个 split JSON 即可精确恢复训练、验证和评估使用的 episode。

### 推荐传输方式

先从当前机器传工作树，保留未提交修改：

```bash
rsync -a --info=progress2 \
  --exclude='.venv/' \
  --exclude='external/' \
  --exclude='outputs/' \
  /workspace/flf_progress_vla/ \
  USER@NEW_HOST:/workspace/flf_progress_vla/
```

再传自研推理权重和评估记录：

```bash
cd /workspace/flf_progress_vla

rsync -aR --info=progress2 \
  ./outputs/mvp_continuation/checkpoint-10000/config.json \
  ./outputs/mvp_continuation/checkpoint-10000/model-00001-of-00002.safetensors \
  ./outputs/mvp_continuation/checkpoint-10000/model-00002-of-00002.safetensors \
  ./outputs/mvp_continuation/checkpoint-10000/model.safetensors.index.json \
  ./outputs/mvp_continuation/checkpoint-10000/experiment_cfg/metadata.json \
  USER@NEW_HOST:/workspace/flf_progress_vla/

rsync -a --info=progress2 \
  /workspace/flf_progress_vla/outputs/reference_video_rollout/gt_test50_v1/ \
  USER@NEW_HOST:/workspace/flf_progress_vla/outputs/reference_video_rollout/gt_test50_v1/
```

上述权重命令只传 `SELF_METHOD_TRANSFER_MANIFEST.txt` 中 `required_inference` 的五个文件，不传 optimizer 或官方权重。官方数据和父 checkpoint 均不上传，在目标机器按固定来源重下。

目标机器重新执行 README/`RESOURCE_SETUP.md` 中的环境安装，并设置：

```text
GR00T_ROOT
ROBOCASA365_DATA_ROOT
GR00T_BASE_CHECKPOINT
```

本机关键环境版本：Python 3.10 环境、PyTorch 2.5.1、Transformers 4.51.3、Gymnasium 1.0.0、RoboCasa 1.0.1、Robosuite 1.5.2、GR00T 1.1.0、NumPy 2.2.5。Robosuite 源码 commit 为 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。

## 8. Git 分支与实验代码发布

本机整理前位于 `main`，`HEAD=4f5c443`；`origin/main=634c353`，因此本地 main 已比远端 main 多一个资源说明 commit。实验代码通过独立分支发布，不直接覆盖远端 main。

采用独立分支 `experiment/gt-video-progress-10k`，不要直接覆盖或 force-push 远端 main。推荐流程是：

1. 在实验分支提交代码、冻结 split、实验文档和小型结果汇总；
2. 模型权重使用 Git LFS、Hugging Face/private object storage 或服务器间 `rsync`，不要进入普通 Git history；
3. 推送该分支并通过 PR 检查 diff；
4. 确认无误后再 squash/merge 到 main。

本次实验分支纳入的本机变更包括：

```text
modified:
  .gitignore
  configs/mvp.json
  scripts/rollout_reference_video.py
  scripts/train_continuation.py

added:
  configs/gt_test50_v1.json
  scripts/evaluate_dtw_gt_replay.py
  scripts/launch_dtw_gt_replay_eval.sh
  scripts/launch_gt_test50_eval.sh
  scripts/launch_official_groot_baseline_eval.sh
  scripts/run_gt_test50_batch.py
  EXPERIMENT_RECORD_AND_MIGRATION.md
  MIGRATION_SHA256SUMS.txt
  SELF_METHOD_TRANSFER_MANIFEST.txt
  transfer/self_method_checkpoint10000/README.md
  transfer/self_method_checkpoint10000/ARCHIVE_SHA256.txt
```

其中 `rollout_reference_video.py` 包含准确恢复 episode XML/MuJoCo 初始状态和稳定写视频所需的修正。5.6 GiB 的 `checkpoint-10000-inference.tar.zst` 被 `.gitignore` 排除，只通过用户选择的外部方式传输；Git 中只保存它的说明和校验值。

## 9. 完整性校验

关键模型、split 和汇总结果的 SHA-256 保存在：

```text
MIGRATION_SHA256SUMS.txt
```

保持本文档中的目标目录布局后，在 `/workspace` 执行：

```bash
sha256sum -c flf_progress_vla/MIGRATION_SHA256SUMS.txt
```

校验清单只包含自研模型、代码、split 和本机生成的实验记录，不包含可在线下载的官方模型或数据。
