#!/usr/bin/env bash
set -euo pipefail

token_file=/tmp/hf_token_minimax_h3
log_file=/workspace/logs/minimax_fl2va_download.log
trap 'rm -f "$token_file"' EXIT

mkdir -p /workspace/logs /workspace/models/MiniMax-H3
exec > >(tee -a "$log_file") 2>&1

date --iso-8601=seconds
python3 /workspace/tools/download_minimax_h3_partition.py \
  --partition fl2va \
  --output /workspace/models/MiniMax-H3 \
  --token-file "$token_file"
date --iso-8601=seconds
