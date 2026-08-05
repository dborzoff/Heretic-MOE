#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

repo=/workspace/heretic-gemma3/heretic-moe-qwen-night-abs
run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_qwen_night_abs_600_v1
cd "${repo}"

exec python research/scripts/finalist_recheck.py run \
  --output-dir "${run_root}/finalist_recheck_64x1024_top5_v2" \
  --heretic "${repo}/research/runs/adaptive_search_v2/gemma3_12b_qat_qwen_night_abs_600_v1/heretic-night-abs" \
  --devices 0 1
