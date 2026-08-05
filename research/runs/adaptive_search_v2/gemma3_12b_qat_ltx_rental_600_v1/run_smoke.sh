#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HERETIC_WORKER_LABEL="GPU 0 smoke"

run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_ltx_rental_600_v1
smoke_dir="${run_root}/smoke_gpu0"
mkdir -p "${smoke_dir}/checkpoints"
cp "${run_root}/base_config.toml" "${smoke_dir}/config.toml"
cd "${smoke_dir}"

exec heretic \
  --n-trials 1 \
  --n-startup-trials 1 \
  --startup-design random \
  --parallel-workers 1 \
  --study-checkpoint-dir="${smoke_dir}/checkpoints" \
  --trial-responses-file="${smoke_dir}/trial-responses.sqlite3"
