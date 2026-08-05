# Stage-State + Video Progress GR00T Continuation

这是一个可独立克隆和运行的研究代码仓库，用于在 RoboCasa365 上对 GR00T N1.5 做
stage-state + continuous video progress continuation fine-tuning。

仓库不依赖任何父目录布局。RoboCasa365 数据、GR00T checkpoint 和 RoboCasa 版本的
Isaac-GR00T 均作为显式外部资源，通过环境变量或本地 `.env` 提供。

## 方法与数据合同

每个 active-subtask control frame 构成一个训练样本：

- 视觉：GR00T `panda_omron` 原生三视角（left、right、wrist）；
- 语言：完整 composite task instruction，不替换为子任务文本；
- 动作：从当前帧开始的原生 16-step action chunk；
- stage-state：0-based ordinal stage index，用固定 `Kmax=5` 映射到 `[-1,1]`；
- video progress：参考视频每 8 个 control frames（20 Hz 下为 0.4 秒）一个节点，节点位置作为连续标量映射到 `[-1,1]`；
- 条件注入：stage 和 progress 写入 GR00T 已归一化并 padding 后的 64-D state 中最前面的两个空维度。RoboCasa transformed state 占前 20 维，因此当前使用第 20、21 维（0-based），不扩 state projector；
- normalization：四个任务统一使用父 checkpoint 的 `experiment_cfg/metadata.json`，与 `Gr00tPolicy` 推理一致；
- action mask：不在子任务边界截断 chunk，只移除真正越过 episode 末尾的重复 padding。

四个任务的 active stage 数分别为：

| Task | Stages |
|---|---:|
| PreSoakPan | 5 |
| KettleBoiling | 3 |
| LoadDishwasher | 5 |
| RinseSinkBasin | 2 |

`RinseSinkBasin` 的两个阶段都叫 `execute`，索引器同时使用 `subtask_idx` 判断边界。
仓库包含固定的 100 train / 10 validation / 20 locked test episode split，但索引生成只处理
train 和 validation，不读取 test。

## 仓库内容

```text
.
├── configs/
│   ├── mvp.json
│   └── robocasa365_four_task_split_100.json
├── src/stage_state_vla/
├── scripts/
├── tests/
├── .env.example
├── pyproject.toml
└── THIRD_PARTY.md
```

以下内容均被 Git 忽略：`.env`、`external/`、数据索引、训练输出、checkpoint 和 Python cache。

## 1. 获取固定版本的 GR00T

已验证的 RoboCasa fork 与 commit 记录在 [THIRD_PARTY.md](THIRD_PARTY.md)。可以自动克隆：

```bash
./scripts/bootstrap_gr00t.sh
```

也可以复用其他位置的干净 checkout，只需在 `.env` 中设置 `GR00T_ROOT`。代码会检查
Transformers 主版本；`4.51.x` 可用，`5.x` 与该 GR00T revision 不兼容。

## 2. Python 环境

当前验证组合：Python 3.10.16、PyTorch 2.5.1、TorchVision 0.20.1、Transformers
4.51.3。推荐先按照固定 GR00T fork 的方式安装其训练环境，再安装本仓库：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'external/Isaac-GR00T[base]'
python -m pip install -e '.[dev]'
```

在已经配置好的 GR00T/HPC 环境中，可以使用 `python -m pip install -e . --no-deps`，避免
重新安装平台相关的 CUDA PyTorch wheel。

## 3. 配置外部资源

```bash
cp .env.example .env
```

编辑 `.env`：

```text
GR00T_ROOT=/absolute/path/to/Isaac-GR00T
ROBOCASA365_DATA_ROOT=/absolute/path/to/robocasa365/v1.0/target/composite
GR00T_BASE_CHECKPOINT=/absolute/path/to/checkpoint-60000
STAGE_STATE_PYTHON=/absolute/path/to/environment/bin/python
NPROC_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1
```

运行只读环境检查：

```bash
python scripts/check_environment.py
```

需要的外部资源约为 12 GB：四任务 RoboCasa365 数据约 4.3 GB，父 checkpoint 约
7.1 GB，固定 GR00T checkout 不到 100 MB。数据目录下每个任务应恰好包含一个
`*/lerobot/meta/info.json`。

## 4. 生成数据索引

完整索引：

```bash
python scripts/build_indices.py
```

预期结果：train 233,808 frames，validation 22,779 frames。每任务一个 episode 的 smoke：

```bash
python scripts/build_indices.py --max-episodes-per-task 1
python scripts/smoke_dataset.py
```

索引位于 `artifacts/indices/`，约 116 MB，可随时从固定 split 再生成，因此不提交 Git。

## 5. 测试

```bash
python -m pytest -q
bash -n scripts/launch_two_gpu.sh
```

## 6. 两卡训练

两卡 20-step smoke：

```bash
./scripts/launch_two_gpu.sh \
  --max-steps 20 \
  --no-eval \
  --train-index artifacts/indices/train_smoke1.jsonl \
  --output-dir outputs/smoke_20
```

正式 MVP 默认是 effective global batch 128、1000 optimizer steps，每 250 steps 做
validation 和 checkpoint：

```bash
./scripts/launch_two_gpu.sh
```

训练冻结视觉和语言 backbone，训练 GR00T 原生 action-side projector、VLLN/self-attention
和 DiT。checkpoint 参数保持 FP32，由 Trainer 使用 BF16 autocast。采样时先均匀选择四个
任务（各 25%），再在该任务内均匀选择 frame。

## 7. 离线条件敏感性检查

```bash
python scripts/evaluate_conditions.py \
  --checkpoint outputs/mvp_continuation
```

脚本在相同 flow noise/time 下比较正确条件、固定 progress 和 shuffled stage/progress 的
validation loss。正确条件 loss 更低且 counterfactual 条件能改变输出，才说明模型没有忽略
新维度。

## 8. 推理接口

`stage_state_vla.policy.make_conditioned_policy_class()` 返回独立的 `Gr00tPolicy` 子类：

```python
actions = policy.get_action_with_condition(
    observations,
    stage_index=current_stage,
    video_progress=current_progress,
)
```

`MonotonicStageController` 只允许 hold 或 `+1`，默认 progress `>=0.9` 连续确认两次才推进。
Oracle progress 和后续 video localizer 都通过同一接口提供 `current_progress`，无需修改 VLA。

## 已完成的实现验证

详见 [SMOKE_VALIDATION.md](SMOKE_VALIDATION.md)。目前已完成单元测试、真实三视角 sample
contract、两卡 DDP forward/backward、保存和断点恢复验证；尚未启动正式 1000-step 训练。
