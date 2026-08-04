#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/minimax_te
COMFY=/workspace/ComfyUI
MODEL="$ROOT/builds/marker-zero/qwen3vl_32b_minimax_h3_bf16.safetensors"
REPO=/workspace/models/Qwen3-VL-32B-Instruct
INPUTS="$ROOT/minimax_h3_official_inputs"
OUT="$ROOT/reports/marker_zero_qwen_h3_official_corpus_scan.json"
LOG="$ROOT/reports/marker_zero_qwen_h3_official_corpus_scan.log"

printf '\033]0;MiniMax H3 TE - marker-zero conditioning\007'
echo '=== MINIMAX H3 TE STAGE 2: MARKER-ZERO OFFICIAL CONDITIONING SCAN ==='
date -Is

if [[ ! -d "$COMFY/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
fi
python -m pip install -r "$COMFY/requirements.txt"

test -s "$MODEL"
test -d "$INPUTS"
test -s "$ROOT/reports/original_qwen_h3_official_corpus_scan.json"

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
