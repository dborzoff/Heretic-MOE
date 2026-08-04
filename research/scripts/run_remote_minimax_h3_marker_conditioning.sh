#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/minimax_te
COMFY=/workspace/ComfyUI
MODEL="$ROOT/builds/marker-zero/qwen3vl_32b_minimax_h3_bf16.safetensors"
REPO=/workspace/models/MiniMax-H3
INPUTS="$ROOT/minimax_h3_official_inputs"
OUT="$ROOT/reports/marker_zero_qwen_h3_official_corpus_scan.json"
LOG="$ROOT/reports/marker_zero_qwen_h3_official_corpus_scan.log"

printf '\033]0;MiniMax H3 TE - marker-zero conditioning\007'
echo '=== MINIMAX H3 TE STAGE 2: MARKER-ZERO OFFICIAL CONDITIONING SCAN ==='
date -Is

if [[ ! -d "$COMFY/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
if [[ ! -f "$ROOT/reports/comfy_requirements.ok" ]]; then
  python -m pip install -r "$COMFY/requirements.txt"
  python - <<'PY'
import comfy_kitchen
print("comfy_kitchen=PASS")
PY
  touch "$ROOT/reports/comfy_requirements.ok"
fi
if ! python -c 'import cv2' >/dev/null 2>&1; then
  python -m pip install opencv-python-headless
fi

if [[ ! -f "$REPO/scripts/readme/reproducible-768p-t2va-request.sh" ]]; then
  hf download MiniMaxAI/MiniMax-H3 \
    --include 'scripts/readme/*' 'FL2VA/text_encoder/*' \
    --exclude '*.safetensors' \
    --local-dir "$REPO"
fi
if [[ ! -f "$REPO/assets/fl2va-clay-fox-reference.png" ]]; then
  echo 'Downloading official MiniMax H3 reference assets required by the scan...'
  hf download MiniMaxAI/MiniMax-H3 \
    --include 'assets/*' \
    --local-dir "$REPO"
fi
if [[ -d "$INPUTS/assets" ]]; then
  echo 'Restoring missing frozen reference assets from the scan input bundle...'
  mkdir -p "$REPO/assets"
  cp -an "$INPUTS/assets/." "$REPO/assets/"
fi

test -s "$MODEL"
test -d "$INPUTS"
test -s "$ROOT/reports/original_qwen_h3_official_corpus_scan.json"
test -s "$REPO/assets/fl2va-clay-fox-reference.png"
test -s "$REPO/assets/character-action-reference.png"

export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="$COMFY${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/tools/qwen_h3_official_corpus_scan.py" \
  --model "$MODEL" \
  --repo "$REPO" \
  --inputs "$INPUTS" \
  --out "$OUT" \
  --threads 32 \
  2>&1 | tee "$LOG"

test -s "$OUT"
sha256sum "$OUT" "$LOG" | tee "$ROOT/reports/marker_zero_conditioning_sha256.txt"
date -Is
echo 'stage=marker_zero_conditioning status=PASS'
