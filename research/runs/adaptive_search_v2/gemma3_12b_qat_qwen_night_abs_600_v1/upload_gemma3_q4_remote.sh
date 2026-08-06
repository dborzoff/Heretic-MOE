#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/heretic-gemma3
RUN="$ROOT/runs/gemma3_12b_qat_qwen_night_abs_600_v1"
RELEASE="$RUN/release"
REPO=DmitryDB/Gemma-3-12B-IT-QAT-Heretic-MOE-v1
UPLOADER="$ROOT/heretic-moe-qwen-night-abs/research/scripts/upload_hf_release_file.py"
PY=/venv/main/bin/python
TOKEN_FILE="$ROOT/.hf_token"
REPORTS="$RELEASE/research/upload_reports"

export HF_TOKEN
HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
export HF_XET_HIGH_PERFORMANCE=1
trap 'unset HF_TOKEN' EXIT
mkdir -p "$REPORTS"

for variant in balanced max; do
  label="${variant^}"
  source="$RELEASE/GGUF/$variant/Gemma-3-12B-IT-QAT-HereticMOE-${label}-Q4_0.gguf"
  target="GGUF/$variant/$(basename "$source")"
  echo "UPLOAD_START $target bytes=$(stat -c %s "$source")"
  "$PY" "$UPLOADER" \
    --repo "$REPO" \
    --file "$source" \
    --path-in-repo "$target" \
    --commit-message "Publish $target" \
    --report "$REPORTS/Q4_0-$variant.json"
  echo "UPLOAD_VERIFIED $target"
  rm -f "$source"
  echo "SOURCE_REMOVED $source"
done

echo "Q4_UPLOAD_COMPLETE"
