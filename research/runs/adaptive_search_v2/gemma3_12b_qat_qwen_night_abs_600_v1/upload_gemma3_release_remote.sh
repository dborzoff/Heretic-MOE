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

if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "Missing temporary HF token file: $TOKEN_FILE" >&2
  exit 2
fi

export HF_TOKEN
HF_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
export HF_XET_HIGH_PERFORMANCE=1
trap 'unset HF_TOKEN; rm -f "$TOKEN_FILE"' EXIT
mkdir -p "$REPORTS"
shopt -s nullglob

upload_one() {
  local source="$1"
  local target="$2"
  local slug
  slug="$(printf '%s' "$target" | tr '/ ' '__')"
  echo "UPLOAD_START $target bytes=$(stat -c %s "$source")"
  "$PY" "$UPLOADER" \
    --repo "$REPO" \
    --file "$source" \
    --path-in-repo "$target" \
    --commit-message "Publish $target" \
    --report "$REPORTS/${slug}.json"
  echo "UPLOAD_VERIFIED $target"
  rm -f "$source"
  echo "SOURCE_REMOVED $source"
}

# LTX text encoders are uploaded first because they are the primary release use.
for variant in balanced max; do
  for source in "$RELEASE/Text_Encoder/$variant"/*.safetensors; do
    upload_one "$source" "Text_Encoder/$variant/$(basename "$source")"
  done
done

# Director GGUF files and their matching importance matrices stay together.
for variant in balanced max; do
  for source in "$RELEASE/GGUF/$variant"/*.gguf "$RELEASE/GGUF/$variant"/*.imatrix; do
    upload_one "$source" "GGUF/$variant/$(basename "$source")"
  done
done

# Full Transformers masters are uploaded last.
for variant in balanced max; do
  if [[ "$variant" == balanced ]]; then
    source_dir="$RUN/selected_models/balanced_t555/model"
  else
    source_dir="$RUN/selected_models/max_t752/model"
  fi
  while IFS= read -r -d '' source; do
    upload_one "$source" "Transformers/$variant/$(basename "$source")"
  done < <(find "$source_dir" -maxdepth 1 -type f -print0 | sort -z)
done

echo "RELEASE_UPLOAD_COMPLETE"
