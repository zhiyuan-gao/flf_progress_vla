#!/bin/bash
set -euo pipefail

repo_root=/workspace/flf_progress_vla
cd "${repo_root}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

exec "${repo_root}/.venv/bin/python" -u scripts/evaluate_conditions.py \
  --config configs/mvp.json \
  --checkpoint outputs/mvp_continuation/checkpoint-10000 \
  --index artifacts/indices/val.jsonl \
  --output outputs/condition_sensitivity_mvp256/report.json \
  --samples-per-task 64 \
  --progress-bins 4 \
  --batch-size 32 \
  --num-gpus 4 \
  --seed 20260805 \
  --signal-threshold 0.01
