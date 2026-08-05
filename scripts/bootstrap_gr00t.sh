#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-${repo_root}/external/Isaac-GR00T}"
remote="https://github.com/robocasa-benchmark/Isaac-GR00T.git"
commit="9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"

if [[ -e "${target}" ]]; then
  echo "Target already exists; refusing to overwrite: ${target}" >&2
  exit 1
fi

mkdir -p "$(dirname "${target}")"
git clone "${remote}" "${target}"
git -C "${target}" checkout --detach "${commit}"

actual="$(git -C "${target}" rev-parse HEAD)"
if [[ "${actual}" != "${commit}" ]]; then
  echo "Unexpected GR00T commit: ${actual}" >&2
  exit 1
fi
echo "Pinned RoboCasa Isaac-GR00T ready at ${target} (${actual})"
