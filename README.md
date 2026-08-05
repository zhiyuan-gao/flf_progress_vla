# Stage-State + Video Progress GR00T Continuation

这是一个可独立克隆和运行的研究代码仓库，用于在 RoboCasa365 上对 GR00T N1.5 做
stage-state + continuous video progress continuation fine-tuning。

仓库不依赖任何父目录布局。RoboCasa365 数据、GR00T checkpoint 和 RoboCasa 版本的
Isaac-GR00T 均作为显式外部资源，通过环境变量或本地 `.env` 提供。

## 研究目标与整体框架

本研究的上层模块会在**每个子任务开始时调用一次视频生成模型**，得到该子任务从开始到
完成的参考视频。下层执行策略不直接接收整段参考视频，而是由一个 progress localizer 将
当前观测与参考视频对齐，输出当前子任务内的连续进度：

\[
p_t \in [0,1].
\]

系统同时维护当前的 ordinal subtask index：

\[
k_t \in \{0,\ldots,K-1\}.
\]

然后把 `(k_t, p_t)` 作为两个额外 robot-state 条件输入 GR00T，让同一个视觉状态和完整任务
语言在不同执行阶段产生不同的 action chunk：

```text
完整 composite instruction ───────────────────────────────┐
当前三视角 RGB + robot state ─────────────────────────────┤
                                                         ▼
子任务开始时生成一次参考视频 ──► progress localizer ──► p_t
                                             stage manager ──► k_t
                                                         │
robot state[0:20] + stage[20] + progress[21] ────────────┤
                                                         ▼
                                              GR00T N1.5 policy
                                                         │
                                                         ▼
                                               16-step action chunk
```

参考视频在这个方法中只负责产生 progress scalar；视频 pixels/features **不会直接拼到 VLA
输入中**。因此参考视频的时间长度与 16-step action chunk 不需要逐帧一一对齐。每次策略请求
使用当前时刻的 `(k_t,p_t)`，预测并执行一个 16-step chunk，然后在下一次请求前重新估计进度。

### 为什么采用这条路线

此前 latent world model 路线在 demonstration-only 数据上能学到平均运动模式，但 validation
上难以根据具体 state-action 判断正确的运动方向；要继续该路线需要额外收集带反事实覆盖的
rollout 数据。这个仓库实现的是另一条不训练 latent world model 的路线：直接告诉 VLA 当前
处于哪个子任务、子任务内部进行到哪里，检验这些阶段信息能否提高闭环成功率。

本仓库与 latent world model、CEM 和 residual-action 方法没有代码依赖，也不应把它们重新
混入当前 MVP。

## 研究假设与实验阶段

核心假设是：当前 observation、robot state 和完整 composite instruction 对动作选择仍存在
temporal aliasing；显式加入 coarse stage 与 continuous within-stage progress 后，GR00T 可以
更可靠地选择当前应执行的动作。

实验按以下阶段推进：

1. **离线 continuation training**：从 demonstration segment 构造 oracle `k_t` 与 `p_t`，训练
   conditioned GR00T。第一版使用干净的 stride-8 reference-grid progress。
2. **离线条件敏感性**：在 validation 上比较正确条件、固定 progress、shuffled stage/progress；
   确认模型不是忽略两个新维度。
3. **Oracle closed-loop MVP**：仿真器提供正确 stage/progress，比较 parent GR00T 与 conditioned
   GR00T 的四任务成功率。当前阶段允许使用 oracle 信息，目标是先证明阶段信息确实有价值。
4. **Video-progress rollout**：每个子任务只生成一次参考视频，用 progress localizer 替代 oracle
   progress；stage controller 根据连续进度单调推进。
5. **鲁棒性实验**：在干净 MVP 有正信号后，再加入 `±1` reference-node progress noise、多 seed
   和 progress localizer 误差。

第一阶段的成功条件不是单纯看 training loss，而是同时满足：

- validation correct-condition loss 不劣于并最好低于 counterfactual condition；
- 改变 stage/progress 会系统性改变 action prediction；
- oracle closed-loop success rate 高于同一 parent checkpoint；
- 最终 video-progress rollout 尽可能接近 oracle-progress 上限。

## 当前实现边界

已经实现：

- 固定 episode split 和 per-frame oracle stage/progress index；
- 父 checkpoint normalization 下的 64-D state 条件注入；
- 四任务等权 sampler、两卡 continuation trainer、validation/checkpoint/resume；
- counterfactual condition 离线评估；
- conditioned GR00T policy wrapper；
- 只允许 hold 或 `+1` 的 monotonic stage controller。
- 从真实 demonstration 三视角视频构造 per-stage reference nodes；
- 已在前序实验验证的 monotonic subsequence-DTW progress localizer；
- action chunk 执行期间按 stride-8 收集 observation，维护最近 8 个 query nodes；
- 从真实 demonstration XML/state 起点运行的单 episode RoboCasa rollout driver。

尚未实现：

- 上层子任务视频生成模型；
- 可处理 rollout 大幅偏离 reference 的 learned visual representation/localizer；
- 正式的多 episode、多 seed RoboCasa batch evaluation driver；
- 正式 1000-step continuation checkpoint。

HPC 上的 Codex 不应假定这些缺失模块已经存在。当前仓库首先负责跑通 oracle-conditioned
continuation；video generator 和 progress localizer 是后续通过相同 `(k_t,p_t)` 接口接入的模块。

## 已固定的设计决定

除非新的实验明确要求，不要擅自改变以下设置：

| 项目 | 固定设置 |
|---|---|
| Base policy | GR00T N1.5 target post-training `checkpoint-60000` |
| VLA visual input | `panda_omron` 原生 left/right/wrist 三视角 |
| Language | 完整 composite instruction |
| State condition | ordinal stage + continuous progress 两个维度 |
| State shape | 保持 64D，动态写入前两个空 slot，不扩 projector |
| Stage normalization | 固定 `Kmax=5` 映射到 `[-1,1]` |
| Progress target | 子任务内 stride-8 reference grid，连续值而非分类 |
| Action horizon | 预测并执行 16 steps；不在子任务边界截断 |
| Task sampling | 四任务各 25%，任务内均匀采 frame |
| Trainable modules | 冻结 visual/language backbone；训练 action-side projector、VLLN/self-attention、DiT |
| Data split | 100 train / 10 validation / 20 locked test episodes per task |
| Test policy | 正式模型选择期间不得读取或调参 test split |
| Stage transition | progress `>=0.9` 连续两次后只允许 `k→k+1`，禁止回退和跳级 |
| Progress localizer | `agentview_left` RGB32、stride-8、8-node monotonic subsequence-DTW |

HPC GPU 数量变化时可以调整 per-device batch 与 gradient accumulation，但第一轮仍应保持
effective global batch 128，从而不改变优化语义。

## 给新 Codex 的接手顺序

如果这是 HPC Codex 第一次看到项目，应按以下顺序工作：

1. 阅读本 README、`RESOURCE_SETUP.md`、`configs/mvp.json`、`THIRD_PARTY.md` 和
   `SMOKE_VALIDATION.md`。
2. 复制 `.env.example` 为 `.env`，只填写 HPC 上的外部资源绝对路径。
3. 运行 `python scripts/check_environment.py`，确认 GR00T commit、Transformers 和 GPU。
4. 运行 `python -m pytest -q`，再生成索引并运行 `scripts/smoke_dataset.py`。
5. 先做 20-step 多卡 smoke，记录 peak memory、samples/s、optimizer-step time。
6. 没有用户明确确认时，不启动正式 1000-step 训练，不访问 locked test，不改固定设计。
7. 正式训练完成后先运行 offline condition sensitivity，再决定是否进行 oracle rollout。

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
├── RESOURCE_SETUP.md
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

如果新机器上没有任何数据或模型，先严格按照
[RESOURCE_SETUP.md](RESOURCE_SETUP.md) 下载并校验资源。不要用 `target_only`、Human300、
pretrain/MimicGen 数据或其他 checkpoint 替换。

固定资源摘要：

| 资源 | 必须使用的版本 |
|---|---|
| 数据 | RoboCasa365 target-human / composite |
| PreSoakPan | snapshot `20250809` |
| KettleBoiling | snapshot `20250814` |
| LoadDishwasher | snapshot `20250811` |
| RinseSinkBasin | snapshot `20250816` |
| 父模型仓库 | `robocasa/robocasa365_checkpoints` |
| 父模型 revision | `14895998fe7c8f8f2441cc8957ec2c510302758b` |
| 父模型目录 | `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000` |
| GR00T 代码 | RoboCasa Isaac-GR00T commit `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10` |

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

## 9. Ground-truth reference video rollout

先验证真实 demonstration 能被切成每个 stage 的 `agentview_left` reference nodes，不加载模型
或仿真器：

```bash
python scripts/rollout_reference_video.py \
  --task PreSoakPan \
  --episode-split val \
  --prepare-only
```

使用训练后的 conditioned checkpoint，从同一 demonstration 的 XML 和初始 MuJoCo state 开始
闭环执行：

```bash
python scripts/rollout_reference_video.py \
  --checkpoint outputs/mvp_continuation \
  --task PreSoakPan \
  --episode-split val \
  --condition-source reference_dtw
```

`reference_dtw` 使用前序 held-out GT-video 实验验证过的协议：单个 `agentview_left` RGB32
descriptor，reference 和 rollout observation 都每 8 个 control frames 取一个节点，保留最近 8 个
query nodes，与当前 stage 的完整 reference 做 monotonic subsequence-DTW。对齐没有固定窗口或
全局速度，允许 stay 和任意 forward jump；`motion_weight`、`stay_penalty` 和 `jump_penalty` 均为
0。跨 policy call 额外约束 endpoint 不回退，stage 仍由连续两次 `progress>=0.9` 推进。

前序 validation/test 各 75 条 held-out stage video 的五种非均匀 GT path 实验中，该协议的
full-path exact accuracy 为 100%，path MAE 和 endpoint progress MAE 均为 0。该结果证明的是
同一 GT stage video 内的任意单调时间路径可恢复，不代表 policy rollout 偏离 reference 后仍有
同样准确率。

诊断时可以使用 `--condition-source reference_clock`，按已执行 control step 读取 demonstration
的 stage/progress。它只能验证 condition/policy/rollout plumbing；rollout 偏离 demonstration 后，
它不是 simulator semantic oracle。默认只允许 train/validation episode；访问 locked test 必须显式
传入 `--allow-locked-test`。

## 10. 已记录的推理待决项

query length 8 的含义、训练与推理设置的边界，以及训练完成后建议比较的 progress 修正和
stage 切换规则，统一记录在 [INFERENCE_DESIGN_NOTES.md](INFERENCE_DESIGN_NOTES.md)。这些设置
目前不改变训练合同；正式训练期间保持 oracle stage/progress targets 不变，待 checkpoint 完成后
固定同一个模型做推理 ablation。

## 已完成的实现验证

详见 [SMOKE_VALIDATION.md](SMOKE_VALIDATION.md)。目前已完成单元测试、真实三视角 sample
contract、两卡 DDP forward/backward、保存和断点恢复验证；尚未启动正式 1000-step 训练。
