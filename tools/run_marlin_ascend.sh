#!/bin/bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${MARLIN_CANN_ROOT:-/usr/local/Ascend/cann}/set_env.sh"
source "${MARLIN_VENV:-$(dirname "$repo_root")/.venv}/bin/activate"
export FLAGTREE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES="${MARLIN_DEVICE:-7}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
exec python -u "$@"
