# Exact data and model resources for a clean machine

This repository does not vendor datasets or model weights. A clean machine needs exactly the
artifacts below. Do not substitute similarly named `pretrain`, `target_only`, Human300,
MimicGen, generic NVIDIA GR00T, or a different checkpoint step.

## 1. Required artifacts

### RoboCasa365 demonstrations

Use the official RoboCasa365 **target-human / composite** LeRobot datasets:

| Task | Exact snapshot | Episodes | Frames | Videos |
|---|---:|---:|---:|---:|
| `PreSoakPan` | `20250809` | 501 | 395,501 | 1,503 |
| `KettleBoiling` | `20250814` | 501 | 228,349 | 1,503 |
| `LoadDishwasher` | `20250811` | 501 | 369,430 | 1,503 |
| `RinseSinkBasin` | `20250816` | 509 | 211,036 | 1,527 |
| **Total** | | **2,012** | **1,204,316** | **6,036** |

Dataset contract:

- LeRobot `codebase_version`: `v2.1`;
- robot: `PandaOmron`;
- control/video rate: 20 Hz;
- three 256x256 H.264 RGB cameras: left, right, and eye-in-hand;
- per-frame `annotation.human.subtask_stage` and `subtask_idx` are required.

The URLs below come from RoboCasa's official `box_links_ds.json` at revision
`b4684e6ee37d377cc392e98302a6b916d588b415`:

<https://github.com/robocasa/robocasa/blob/b4684e6ee37d377cc392e98302a6b916d588b415/robocasa/models/assets/box_links/box_links_ds.json>

### GR00T parent checkpoint

Use exactly:

```text
Hugging Face repository:
  robocasa/robocasa365_checkpoints

Repository revision:
  14895998fe7c8f8f2441cc8957ec2c510302758b

Checkpoint subdirectory:
  gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
```

Official directory:

<https://huggingface.co/robocasa/robocasa365_checkpoints/tree/14895998fe7c8f8f2441cc8957ec2c510302758b/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000>

This is the official GR00T N1.5 **pretraining + 100% target post-training,
Composite-Seen** checkpoint at step 60,000. It is not the `target_only` checkpoint.

### GR00T source code

Use RoboCasa's Isaac-GR00T fork:

```text
Repository: https://github.com/robocasa-benchmark/Isaac-GR00T.git
Commit:     9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10
Package:    gr00t 1.1.0
```

The repository script checks out this exact commit:

```bash
./scripts/bootstrap_gr00t.sh
```

No V-JEPA, DINO, latent-world-model, CEM, or residual-policy weights are needed for this method.

## 2. Download the four datasets

Choose a resource directory with at least 100 GB free for source artifacts plus training output:

```bash
export STAGE_STATE_RESOURCE_ROOT=/absolute/path/to/stage_state_resources
export ROBOCASA365_DATA_ROOT="$STAGE_STATE_RESOURCE_ROOT/data/robocasa365/v1.0/target/composite"
mkdir -p "$ROBOCASA365_DATA_ROOT"
```

Download the four exact official archives:

```bash
set -euo pipefail

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

  curl --fail --location \
    --retry 5 --retry-all-errors \
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

The required final layout is:

```text
$ROBOCASA365_DATA_ROOT/
├── PreSoakPan/20250809/lerobot/
├── KettleBoiling/20250814/lerobot/
├── LoadDishwasher/20250811/lerobot/
└── RinseSinkBasin/20250816/lerobot/
```

Verify the download before building repository indices:

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
    assert info["codebase_version"] == "v2.1", (task, info["codebase_version"])
    assert info["robot_type"] == "PandaOmron", (task, info["robot_type"])
    assert info["total_episodes"] == episodes, (task, info["total_episodes"])
    assert info["total_frames"] == frames, (task, info["total_frames"])
    assert info["fps"] == 20, (task, info["fps"])
    assert parquet_count == episodes, (task, parquet_count)
    assert video_count == episodes * 3, (task, video_count)
    print(task, "OK", episodes, "episodes", frames, "frames", video_count, "videos")
PY
```

## 3. Download the exact GR00T checkpoint

Install the Hugging Face CLI in any download environment:

```bash
python3 -m pip install --upgrade huggingface_hub
hf auth login
```

Download only the five required inference/fine-tuning files:

```bash
export GR00T_CHECKPOINT_REPO="$STAGE_STATE_RESOURCE_ROOT/models/robocasa365_checkpoints"
export GR00T_CHECKPOINT_PREFIX="gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000"

mkdir -p "$GR00T_CHECKPOINT_REPO"

hf download robocasa/robocasa365_checkpoints \
  "$GR00T_CHECKPOINT_PREFIX/config.json" \
  "$GR00T_CHECKPOINT_PREFIX/experiment_cfg/metadata.json" \
  "$GR00T_CHECKPOINT_PREFIX/model.safetensors.index.json" \
  "$GR00T_CHECKPOINT_PREFIX/model-00001-of-00002.safetensors" \
  "$GR00T_CHECKPOINT_PREFIX/model-00002-of-00002.safetensors" \
  --revision 14895998fe7c8f8f2441cc8957ec2c510302758b \
  --local-dir "$GR00T_CHECKPOINT_REPO"

export GR00T_BASE_CHECKPOINT="$GR00T_CHECKPOINT_REPO/$GR00T_CHECKPOINT_PREFIX"
```

Verify every checkpoint file:

```bash
cd "$GR00T_BASE_CHECKPOINT"
sha256sum -c <<'EOF'
6713ae6e9ee07ebf30f18a231bedcf9c06f8c64595d62529b6eb175498ef0526  config.json
0c4a350867f621ed3192f81c282eb7d23c05f275f347fd0deed3531afa29dbcf  experiment_cfg/metadata.json
bec674fcd06f1c6c29e5ab0f057d148a5c76e7ef92d1688d6b4b8f838afc9746  model.safetensors.index.json
672b8e49d32ff124e13c3c4e4e70380ab29cd25f013ff41db768639347f8057e  model-00001-of-00002.safetensors
95c52f05a00141ce4e433305a8ccfa41fd219e8d988f2a4c991f3f4f5cdb677c  model-00002-of-00002.safetensors
EOF
```

Expected shard sizes are 4,999,367,032 and 2,586,705,312 bytes; the directory is approximately
7.1 GiB.

## 4. Connect the resources to this repository

After cloning the pinned GR00T fork and creating the Python environment as described in the main
README, copy `.env.example` to `.env` and set absolute paths:

```text
GR00T_ROOT=/absolute/path/to/stage_state_video_progress/external/Isaac-GR00T
ROBOCASA365_DATA_ROOT=/absolute/path/to/stage_state_resources/data/robocasa365/v1.0/target/composite
GR00T_BASE_CHECKPOINT=/absolute/path/to/stage_state_resources/models/robocasa365_checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000
STAGE_STATE_PYTHON=/absolute/path/to/gr00t-environment/bin/python
NPROC_PER_NODE=2
CUDA_VISIBLE_DEVICES=0,1
```

Then run:

```bash
set -a
. ./.env
set +a
"$STAGE_STATE_PYTHON" scripts/check_environment.py
"$STAGE_STATE_PYTHON" scripts/build_indices.py
"$STAGE_STATE_PYTHON" scripts/build_indices.py --max-episodes-per-task 1
"$STAGE_STATE_PYTHON" scripts/smoke_dataset.py
"$STAGE_STATE_PYTHON" -m pytest -q
```

Expected generated index sizes are 233,808 train frames and 22,779 validation frames. The locked
test split is listed in the split config but is not read while building these indices.
