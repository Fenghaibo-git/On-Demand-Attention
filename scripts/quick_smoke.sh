#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: quick_smoke.sh RUN_DIR" >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -- "$1")
python_bin=${PYTHON:-python3}
"$python_bin" "$repo_dir/scripts/prepare_quick_smoke.py" "$run_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc-per-node=1 \
  "$repo_dir/src/tasks/train_torch.py" "$run_dir/qualification/quick_smoke/quick_smoke.yaml" \
  > "$run_dir/qualification/quick_smoke/train.log" 2>&1
"$python_bin" "$repo_dir/scripts/check_quick_smoke.py" "$run_dir"
