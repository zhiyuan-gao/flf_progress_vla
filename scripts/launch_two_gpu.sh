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
processes="${NPROC_PER_NODE:-2}"

if ! "${python_bin}" -c 'import torch, transformers' >/dev/null 2>&1; then
  echo "Python environment cannot import torch and transformers: ${python_bin}" >&2
  exit 1
fi

export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${processes}" \
  "${standalone_root}/scripts/train_continuation.py" \
  --config "${standalone_root}/configs/mvp.json" \
  "$@"
