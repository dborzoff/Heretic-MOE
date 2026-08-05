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

exec python research/scripts/run_adaptive_search.py \
  --base-config "${run_root}/config.toml" \
  --run-root "${run_root}/adaptive_600" \
  --heretic "${repo}/research/runs/adaptive_search_v2/gemma3_12b_qat_qwen_night_abs_600_v1/heretic-night-abs" \
  --exploration-trials 120 \
  --target-trials 661 \
  --random-device 0 \
  --sobol-device 1 \
  --continue-shared-only
