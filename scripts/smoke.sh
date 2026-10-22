#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: smoke.sh RUN_DIR" >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -- "$1")
python_bin=${PYTHON:-python3}
"$python_bin" "$repo_dir/scripts/prepare_smoke.py" "$run_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc-per-node=8 \
  "$repo_dir/src/tasks/train_torch.py" "$run_dir/qualification/smoke/smoke.yaml" \
  > "$run_dir/qualification/smoke/train.log" 2>&1
"$python_bin" "$repo_dir/scripts/prepare_smoke_restore.py" "$run_dir"
torchrun --standalone --nproc-per-node=8 \
  "$repo_dir/src/tasks/train_torch.py" "$run_dir/qualification/restored_from_step_1/restore.yaml" \
  > "$run_dir/qualification/restored_from_step_1/train.log" 2>&1
"$python_bin" "$repo_dir/scripts/check_smoke.py" "$run_dir"
