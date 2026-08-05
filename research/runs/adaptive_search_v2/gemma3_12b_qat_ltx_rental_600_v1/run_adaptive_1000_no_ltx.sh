#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

repo=/workspace/heretic-gemma3/heretic-moe
run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_no_ltx_1000_v2
cd "${repo}"

exec python research/scripts/run_adaptive_search.py \
  --base-config "${run_root}/base_config.toml" \
  --run-root "${run_root}/adaptive_1000" \
  --heretic /venv/main/bin/heretic \
  --exploration-trials 120 \
  --target-trials 1000 \
  --random-device 0 \
  --sobol-device 1 \
  --continue-shared-only
