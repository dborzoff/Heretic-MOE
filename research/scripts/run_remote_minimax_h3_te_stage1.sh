#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/minimax_te
TEMPLATE_ROOT=/workspace/models/Comfy-Org-MiniMax-H3
TEMPLATE_REL=text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
TEMPLATE="$TEMPLATE_ROOT/$TEMPLATE_REL"
ORIGINAL=/workspace/models/Qwen3-VL-32B-Instruct
MARKER_ZERO=/workspace/exports/qwen3vl32b_hybrid_600_v1/marker-zero
MARKER_OUTPUT="$ROOT/builds/marker-zero/qwen3vl_32b_minimax_h3_bf16.safetensors"
BUILDER="$ROOT/tools/build_minimax_h3_encoder.py"

mkdir -p "$TEMPLATE_ROOT/text_encoders" "$ROOT/reports" "$(dirname "$MARKER_OUTPUT")"
printf '\033]0;MiniMax H3 TE - Stage 1\007'
echo '=== MINIMAX H3 TE STAGE 1: DOWNLOAD + EXACT TRIM VALIDATION ==='
date -Is
df -h /workspace

if [[ ! -f "$TEMPLATE" ]]; then
  echo 'stage=download_original_trim status=running'
  hf download Comfy-Org/MiniMax-H3 "$TEMPLATE_REL" --local-dir "$TEMPLATE_ROOT"
fi

echo "template_bytes=$(stat -c %s "$TEMPLATE")"
echo 'stage=compare_original_trim status=running'
python "$BUILDER" \
  --source-model "$ORIGINAL" \
  --template "$TEMPLATE" \
  --compare-template-source \
  | tee "$ROOT/reports/original_trim_exact.jsonl"
echo 'stage=compare_original_trim status=PASS'

template_bytes=$(stat -c %s "$TEMPLATE")
free_bytes=$(df --output=avail -B1 /workspace | tail -1 | tr -d ' ')
required_bytes=$((template_bytes + 5 * 1024 * 1024 * 1024))
if (( free_bytes < required_bytes )); then
  echo "stage=build_marker_zero status=BLOCKED free_bytes=$free_bytes required_bytes=$required_bytes"
  exit 3
fi

echo 'stage=build_marker_zero status=running'
python "$BUILDER" \
  --source-model "$MARKER_ZERO" \
  --template "$TEMPLATE" \
  --output "$MARKER_OUTPUT" \
  --overwrite \
  --verify \
  | tee "$ROOT/reports/marker_zero_trim_build.jsonl"
echo 'stage=build_marker_zero status=PASS'

df -h /workspace
date -Is
echo 'stage=minimax_h3_te_stage1 status=PASS'
