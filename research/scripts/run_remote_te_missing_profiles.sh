#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: run_remote_te_missing_profiles.sh VARIANT PROFILE...}
shift
(( $# > 0 )) || { echo "missing profiles"; exit 2; }
case "$variant" in balanced|marker-zero) ;; *) echo "unsupported variant=$variant"; exit 2;; esac

ROOT=/workspace
REPO=DmitryDB/Qwen3-VL-32B-Instruct-Heretic-Adaptive-v1
TOOLS="$ROOT/heretic-moe/research/scripts"
TE_ROOT="$ROOT/minimax_te"
RELEASE_ROOT="$ROOT/qwen3vl32b_release"
BUILD_ROOT="$RELEASE_ROOT/minimax_h3_text_encoders/$variant"
TRIMMED="$BUILD_ROOT/qwen3vl_32b_minimax_h3_bf16.safetensors"
REPORT_ROOT="$RELEASE_ROOT/reports/minimax_h3_text_encoders/$variant"
PROFILE_MATRIX="$TE_ROOT/tools/qwen_h3_quant_profiles.json"
COMFY_ROOT="$ROOT/ComfyUI"

mkdir -p "$REPORT_ROOT"
exec > >(tee -a "$REPORT_ROOT/recovery.log") 2>&1
echo "OPENAI CODEX | RECOVER MISSING TE PROFILES | variant=$variant profiles=$*"
echo "started=$(date --iso-8601=seconds)"
test -s "$TRIMMED"

for profile in "$@"; do
  report="$REPORT_ROOT/${profile}.upload.json"
  if [[ -s "$report" ]] && grep -q '"status": "PASS"' "$report"; then
    echo "stage=skip-already-uploaded variant=$variant profile=$profile"
    continue
  fi
  output="$BUILD_ROOT/Qwen3-VL-32B-Instruct-Heretic-${variant}-MiniMax-H3-TE-${profile}.safetensors"
  echo "stage=te-quant variant=$variant profile=$profile"
  CUDA_VISIBLE_DEVICES=0 python "$TOOLS/quantize_qwen_h3_te.py" \
    "$TRIMMED" "$output" \
    --profile "$profile" \
    --profile-matrix "$PROFILE_MATRIX" \
    --comfy-root "$COMFY_ROOT" \
    --tools-root "$TE_ROOT/tools" \
    --device cuda:0 \
    --overwrite \
    | tee "$REPORT_ROOT/${profile}.quantize.jsonl"
  python "$TOOLS/validate_qwen_h3_te_quant.py" \
    "$output" --profile "$profile" --report "$REPORT_ROOT/${profile}.validate.json"
  python "$TOOLS/upload_hf_release_file.py" \
    --repo "$REPO" \
    --file "$output" \
    --path-in-repo "minimax_h3_text_encoders/$variant/$(basename "$output")" \
    --commit-message "Add $variant MiniMax-H3 TE $profile" \
    --report "$report"
  rm -f -- "$output"
  echo "stage=te-profile-complete variant=$variant profile=$profile"
done

rm -f -- "$TRIMMED"
echo "finished=$(date --iso-8601=seconds) variant=$variant status=PASS"
