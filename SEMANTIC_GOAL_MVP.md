# Semantic-subtask goal-image MVP (v2)

This document is the executable contract for the second MVP. The original ordinal-stage experiment
remains documented in `README.md` and is not modified or used as the parent checkpoint.

## Method contract

Both variants start from the official GR00T N1.5 Composite-Seen `checkpoint-60000` and train separate
checkpoints with the same four-task split and 10,000 optimizer steps.

| Input | Goal1 | Goal3 |
|---|---|---|
| Current RGB | left, right, wrist | left, right, wrist |
| Endpoint RGB | left | left, right, wrist |
| Language | full task + current natural-language subtask | same |
| State condition | exact within-subtask progress only | same |

The prompt is exactly:

```text
Task: {annotation.human.task_description}
Current subtask: {annotation.human.subtask}
```

There is no ordinal stage scalar and no prose describing image order. The controller keeps an
internal stage index only to select the current subtask text, endpoint images, and reference video.

### Progress

Every active-subtask frame is an eligible stride-1 action-chunk start. Training progress is

```text
(frame - segment_start) / (segment_end - segment_start)
```

and receives a reproducible random temporal perturbation of up to `±4` control frames, clipped to the
same subtask. Validation has no perturbation. Inference keeps the stride-8 rolling-subsequence DTW
localizer (`query_length=8`).

### Explicit HOLD padding

The action head still predicts 16 steps. Targets that cross the current subtask boundary are replaced
before official GR00T normalization:

- EEF translation, EEF rotation, and mobile-base motion: zero incremental command;
- gripper and control mode: repeat the last valid command.

All 16×12 real action values remain supervised. This is HOLD supervision, not a zero loss mask.

### Inference controller

The policy predicts 16 actions but executes at most 8 before relocalizing. Only a fresh stride-8 DTW
observation can increment the completion streak. Two fresh estimates with `progress >= 0.9` advance
the controller by one subtask. On advance, the unused old chunk is discarded, text/goal/reference are
changed, progress resets, and the VLA is called again.

## Exact external resources

Use the resources pinned in `RESOURCE_SETUP.md`:

- RoboCasa365 target-human/composite LeRobot v2.1 datasets:
  - `PreSoakPan/20250809`
  - `KettleBoiling/20250814`
  - `LoadDishwasher/20250811`
  - `RinseSinkBasin/20250816`
- `robocasa/robocasa365_checkpoints` revision
  `14895998fe7c8f8f2441cc8957ec2c510302758b`;
- checkpoint subdirectory
  `gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000`;
- RoboCasa Isaac-GR00T fork commit `9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`.

Do not initialize v2 from the earlier progress-10k checkpoint: it contains the retired ordinal-stage
state condition.

## Prepare indices and endpoint cache

Both variants share the same semantic index and three-camera endpoint cache:

```bash
set -a
. ./.env
set +a

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

The fixed split produces 233,808 train frames and 22,779 validation frames. Train+validation contain
1,650 subtask endpoints. A 30-endpoint smoke cache occupied 5.2 MiB, so the full compressed cache is
expected to be roughly 0.3 GiB; actual codec content can change this estimate.

## Training launchers

### Two A40 development/smoke

The A40 launcher defaults to micro-batch 4 and uses gradient accumulation to preserve the configured
global batch of 128. Override it downward if Goal3 runs out of memory.

```bash
SEMANTIC_GOAL_VARIANT=goal1 scripts/launch_semantic_two_a40.sh
SEMANTIC_GOAL_VARIANT=goal3 scripts/launch_semantic_two_a40.sh
```

A one-step no-checkpoint smoke is:

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
```

`--smoke-only` disables checkpoint saving.

### Four A100-80GB formal runs

```bash
SEMANTIC_GOAL_VARIANT=goal1 scripts/launch_semantic_four_a100.sh
SEMANTIC_GOAL_VARIANT=goal3 scripts/launch_semantic_four_a100.sh
```

Formal defaults are:

- Goal1: 4 GPUs × micro-batch 32 × accumulation 1 = global batch 128;
- Goal3: 4 GPUs × micro-batch 16 × accumulation 2 = global batch 128.

Profile Goal3 memory on the target HPC before committing to the full run. If micro-batch 16 does not
fit, pass `--per-device-batch-size 8`; accumulation is recomputed automatically to keep global batch
128.

## GT-reference inference

The existing rollout entry point detects `method_family=semantic_goal`. Its default execution length
comes from the v2 config and is 8, not the v1 default of 16.

```bash
"$STAGE_STATE_PYTHON" scripts/rollout_reference_video.py \
  --config configs/semantic_goal1.json \
  --checkpoint outputs/semantic_goal1_continuation/checkpoint-10000 \
  --task PreSoakPan \
  --episode-split val
```

Use `configs/semantic_goal3.json` and the Goal3 checkpoint for the six-image variant. The initial MVP
uses GT reference videos and cached GT endpoint images. Replacing those endpoint images with generated,
view-consistent goals is intentionally deferred until the video model is ready.

## Implemented verification

- unit tests cover prompt construction, exact progress jitter, progress-only state injection, goal
  image order, cache round-trip, HOLD targets, and fresh-only completion confirmations;
- real-data transforms produced exactly 4 Eagle images for Goal1 and 6 for Goal3;
- both variants completed a two-A40 distributed forward/backward optimizer step;
- Goal1 completed a one-step RoboCasa rollout through the semantic policy wrapper using the official
  checkpoint as a shape/interface smoke test.
