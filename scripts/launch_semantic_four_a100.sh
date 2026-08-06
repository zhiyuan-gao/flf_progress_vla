#!/usr/bin/env bash
set -euo pipefail

standalone_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${standalone_root}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${standalone_root}/.env"
  set +a
fi

python_bin="${STAGE_STATE_PYTHON:-python}"
variant="${SEMANTIC_GOAL_VARIANT:-goal1}"
config="${standalone_root}/configs/semantic_${variant}.json"
if [[ ! -f "${config}" ]]; then
  echo "Unknown SEMANTIC_GOAL_VARIANT=${variant}; expected goal1 or goal3" >&2
  exit 1
fi

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${standalone_root}/scripts/train_continuation.py" \
  --config "${config}" \
  "$@"
