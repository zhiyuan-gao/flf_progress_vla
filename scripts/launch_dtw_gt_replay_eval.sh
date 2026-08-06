#!/bin/bash
set -euo pipefail

repo_root=/workspace/flf_progress_vla
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

exec "${repo_root}/.venv/bin/python" -u scripts/evaluate_dtw_gt_replay.py \
  --config configs/mvp.json \
  --split-manifest configs/gt_test50_v1.json \
  --output-dir outputs/reference_video_rollout/gt_test50_v1/dtw_gt_replay \
  --num-workers 8 \
  --allow-frozen-test
