#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: resume_train.sh RUN_DIR [512|1024|1536|2048|2560]" >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -- "$1")
python_bin=${PYTHON:-python3}
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
[[ -z "${WANDB_RUN_ID:-}" && -z "${WANDB_RESUME:-}" ]] || {
  echo "Resume identity is taken only from the original run metadata" >&2
  exit 2
}
"$python_bin" -c 'import wandb; assert wandb.Api(timeout=20).viewer'
if [[ $# -eq 2 ]]; then
  "$python_bin" "$repo_dir/scripts/prepare_resume.py" "$run_dir" --checkpoint-step "$2"
else
  "$python_bin" "$repo_dir/scripts/prepare_resume.py" "$run_dir"
fi
attempt=$(find "$run_dir/recovery" -maxdepth 1 -type d -name 'attempt_*' | sort | tail -1)
[[ -f "$attempt/RESUME_PREPARED.json" ]] || exit 2
export WANDB_RUN_ID=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wandb_run_id"])' "$attempt/RESUME_PREPARED.json")
export WANDB_RESUME=must
torchrun --standalone --nproc-per-node=8 \
  "$repo_dir/src/tasks/train_torch.py" "$attempt/resume.yaml" \
  > "$attempt/train.log" 2>&1
"$python_bin" "$repo_dir/scripts/audit_run.py" "$run_dir"
