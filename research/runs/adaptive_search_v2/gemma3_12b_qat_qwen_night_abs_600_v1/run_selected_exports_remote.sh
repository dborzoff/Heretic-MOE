#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

repo=/workspace/heretic-gemma3/heretic-moe-qwen-night-abs
run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_qwen_night_abs_600_v1
recheck_root="${run_root}/finalist_recheck_64x1024_top5_v2"
export_root="${run_root}/selected_models"
heretic="${repo}/research/runs/adaptive_search_v2/gemma3_12b_qat_qwen_night_abs_600_v1/heretic-night-abs"

mkdir -p "${export_root}/balanced_t555" "${export_root}/max_t752"

run_export() {
  local device="$1"
  local recheck_trial="$2"
  local output="$3"
  local log="$4"

  (
    export CUDA_VISIBLE_DEVICES="${device}"
    cd "${recheck_root}"
    exec "${heretic}" \
      --parallel-workers 1 \
      --restore-trial-number "${recheck_trial}" \
      --checkpoint-action continue \
      --model-action save \
      --save-directory "${output}" \
      --export-strategy MERGE \
      --no-optimization-only
  ) >"${log}" 2>&1
}

run_export 0 2 "${export_root}/balanced_t555/model" "${export_root}/balanced_t555/export.log" &
balanced_pid=$!
run_export 1 1 "${export_root}/max_t752/model" "${export_root}/max_t752/export.log" &
max_pid=$!

status=0
wait "${balanced_pid}" || status=1
wait "${max_pid}" || status=1

if [[ "${status}" -ne 0 ]]; then
  echo "One or more selected-model exports failed; inspect per-model logs."
  exit "${status}"
fi

echo "Balanced T555 and Max T752 exports completed."
