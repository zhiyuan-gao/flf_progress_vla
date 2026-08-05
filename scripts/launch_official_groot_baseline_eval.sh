#!/bin/bash
set -euo pipefail

repo_root=/workspace/flf_progress_vla
cd "${repo_root}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

exec "${repo_root}/.venv/bin/python" -u scripts/run_gt_test50_batch.py \
  --config configs/mvp.json \
  --split-manifest configs/gt_test50_v1.json \
  --checkpoint /workspace/stage_state_resources/models/robocasa365_checkpoints/gr00t_n1-5/foundation_model_learning/target_posttraining/composite_seen/checkpoint-60000 \
  --output-dir outputs/reference_video_rollout/gt_test50_v1/official_groot_checkpoint60000/episodes \
  --policy-mode official_unconditioned \
  --num-workers 12 \
  --num-gpus 4 \
  --execute-steps 16 \
  --denoising-steps 4 \
  --base-seed 20260805 \
  --max-attempts 2 \
  --allow-frozen-test
