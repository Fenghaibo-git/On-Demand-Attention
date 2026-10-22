#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: train.sh RUN_DIR" >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -- "$1")
python_bin=${PYTHON:-python3}
[[ -f "$run_dir/PREPARED.json" && -f "$run_dir/qualification/smoke/SMOKE_PASSED.json" ]] || {
  echo "Prepared run and measured real-weight smoke are required" >&2
  exit 2
}
[[ ! -e "$run_dir/train_metrics.jsonl" && ! -e "$run_dir/logs/train.log" ]] || {
  echo "This launcher is for a fresh run only; preserve existing evidence" >&2
  exit 2
}
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
"$python_bin" "$repo_dir/scripts/preflight_train.py" "$run_dir"
mkdir -p "$run_dir/logs"
torchrun --standalone --nproc-per-node=8 \
  "$repo_dir/src/tasks/train_torch.py" "$run_dir/veomni_cli.yaml" \
  > "$run_dir/logs/train.log" 2>&1
"$python_bin" "$repo_dir/scripts/audit_run.py" "$run_dir"
