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
  --checkpoint outputs/mvp_continuation/checkpoint-10000 \
  --output-dir outputs/reference_video_rollout/gt_test50_v1/reference_dtw \
  --num-workers 12 \
  --num-gpus 4 \
  --execute-steps 16 \
  --denoising-steps 4 \
  --base-seed 20260805 \
  --max-attempts 2 \
  --allow-frozen-test
