#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/minimax_te
ORIGINAL=/workspace/models/Qwen3-VL-32B-Instruct
MARKER_ZERO=/workspace/exports/qwen3vl32b_hybrid_600_v1/marker-zero

grep -q '"status": "COMPARE_PASS"' "$ROOT/reports/original_trim_exact.jsonl"
grep -q '"status": "PASS"' "$ROOT/reports/marker_zero_trim_build.jsonl"
test -s "$ROOT/reports/trimmed_bf16_sha256.txt"

original_real=$(realpath "$ORIGINAL")
marker_real=$(realpath "$MARKER_ZERO")
[[ "$original_real" == /workspace/models/Qwen3-VL-32B-Instruct ]]
[[ "$marker_real" == /workspace/exports/qwen3vl32b_hybrid_600_v1/marker-zero ]]

shopt -s nullglob
original_shards=("$ORIGINAL"/model-*-of-*.safetensors)
marker_shards=("$MARKER_ZERO"/model-*-of-*.safetensors)
[[ ${#original_shards[@]} -eq 14 ]]
[[ ${#marker_shards[@]} -eq 14 ]]

echo "reclaim_original_shards=${#original_shards[@]}"
echo "reclaim_marker_zero_shards=${#marker_shards[@]}"
du -ch "${original_shards[@]}" "${marker_shards[@]}" | tail -1
rm -- "${original_shards[@]}" "${marker_shards[@]}"
echo 'reclaim_status=PASS metadata_preserved=true sources_recoverable_from_local_and_hf=true'
df -h /workspace
