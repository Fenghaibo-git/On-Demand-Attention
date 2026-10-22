#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 || ( "$1" != infer && "$1" != score ) ]]; then
  echo "Usage: evaluate_ruler.sh infer|score RUN_DIR step_1024|step_2048|step_3072|merge_1024_2048_3072" >&2
  exit 2
fi
phase=$1
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=$(realpath -- "$2")
case_name=$3
case "$case_name" in
  step_1024|step_2048|step_3072|merge_1024_2048_3072) ;;
  *) echo "Unknown v4.2 evaluation case" >&2; exit 2 ;;
esac
case_dir=$run_dir/evaluations/$case_name
runtime_dir=$case_dir/hf_runtime
[[ -f "$runtime_dir/RUNTIME_PREPARED.json" ]] || { echo "Prepare the audited runtime first" >&2; exit 2; }
[[ -d "${RULER_DATA_ROOT:?Set RULER_DATA_ROOT}" ]] || exit 2
[[ -f "${OPENCOMPASS_ROOT:?Set OPENCOMPASS_ROOT}/run.py" ]] || exit 2
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "Select a GPU" >&2; exit 2; }
python_bin=${PYTHON:-python3}
export ROUTER_RULER_CONFIG="$repo_dir/configs/eval_ruler16k.py"
if [[ "${ROUTER_PAPER_BATCH:-0}" == 1 ]]; then
  export ROUTER_RULER_CONFIG="$repo_dir/configs/eval_ruler16k_paper.py"
fi
work_dir=$case_dir/opencompass
marker_dir=$case_dir/markers
mkdir -p "$marker_dir" "$case_dir/logs" "$case_dir/telemetry" "$work_dir"
if [[ -e "$marker_dir/$phase.json" || -e "$case_dir/logs/$phase.log" ]]; then
  echo "Preserve the previous phase before retrying" >&2
  exit 2
fi
if [[ "$phase" == score && ! -f "$marker_dir/infer.json" ]]; then
  echo "Scoring requires completed inference" >&2
  exit 2
fi
export PYTHONPATH="$repo_dir/src:$OPENCOMPASS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export RULER_NUM_SAMPLES=100
export RULER_RANDOM_SEED=42
export TOKENIZER_MODEL=$runtime_dir
export ROUTER_HF_RUNTIME=$runtime_dir
export QWEN35_SELECTED_CACHE_ROUTER_CHECKPOINT=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["router_checkpoint"])' "$runtime_dir/config.json")
export QWEN35_SELECTED_CACHE_ROUTER_AUDIT=$runtime_dir/runtime_audit.json
export QWEN35_SELECTED_CACHE_ROUTER_ACTION_THRESHOLD=0
export QWEN35_SELECTED_CACHE_ROUTER_MODE=routed
export QWEN35_SELECTED_CACHE_ROUTER_SUMMARY_PATH=$case_dir/telemetry/router-{pid}.json
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ "$phase" == infer ]]; then
  "$python_bin" "$OPENCOMPASS_ROOT/run.py" "$ROUTER_RULER_CONFIG" \
    --mode infer --work-dir "$work_dir" --dump-eval-details \
    --max-num-workers 1 --max-workers-per-gpu 1 > "$case_dir/logs/infer.log" 2>&1
else
  "$python_bin" "$OPENCOMPASS_ROOT/run.py" "$ROUTER_RULER_CONFIG" \
    --mode eval --reuse latest --work-dir "$work_dir" --dump-eval-details \
    --max-num-workers 1 --max-workers-per-gpu 1 > "$case_dir/logs/score.log" 2>&1
fi
"$python_bin" "$repo_dir/scripts/mark_ruler_phase.py" "$phase" "$case_dir"
