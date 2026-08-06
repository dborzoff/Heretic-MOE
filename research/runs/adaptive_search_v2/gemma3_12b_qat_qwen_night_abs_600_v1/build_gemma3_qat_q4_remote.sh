#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/heretic-gemma3
RUN="$ROOT/runs/gemma3_12b_qat_qwen_night_abs_600_v1"
LLAMA="$ROOT/toolchain/llama.cpp"
CONVERT="$ROOT/toolchain/quant-venv/bin/python"
QUANTIZE="$LLAMA/build/bin/llama-quantize"
IMATRIX_BIN="$LLAMA/build/bin/llama-imatrix"
CALIBRATION="$ROOT/heretic-moe-qwen-night-abs/src/heretic/data/perplexity_reference_v1.txt"
RELEASE="$RUN/release"

build_one() {
  local device="$1"
  local variant="$2"
  local label="$3"
  local source="$RUN/q4_sources/$variant"
  local output="$RELEASE/GGUF/$variant/Gemma-3-12B-IT-QAT-HereticMOE-${label}-Q4_0.gguf"
  local work="$RELEASE/tmp/q4-$variant"
  local f16="$work/Gemma-3-12B-IT-QAT-HereticMOE-${label}-F16.gguf"
  local report="$RELEASE/research/validation/${variant}-q4_0"
  local check="$work/load-check.imatrix"

  mkdir -p "$work" "$report"
  "$CONVERT" "$LLAMA/convert_hf_to_gguf.py" "$source" \
    --outtype f16 --outfile "$f16" >"$report/convert-f16.log" 2>&1
  "$QUANTIZE" "$f16" "$output" Q4_0 16 >"$report/q4_0.log" 2>&1
  (
    export CUDA_VISIBLE_DEVICES="$device"
    "$IMATRIX_BIN" -m "$output" -f "$CALIBRATION" -o "$check" \
      --output-format gguf --no-ppl -ngl 999 -c 512 -b 512 --chunks 1
  ) >"$report/load-check.log" 2>&1
  test -s "$check"
  sha256sum "$output" >"$report/sha256sums.txt"
  rm -f "$check" "$f16"
  rmdir "$work" 2>/dev/null || true
  echo "Q4_0_COMPLETE variant=$variant gpu=$device bytes=$(stat -c %s "$output")"
}

build_one 0 balanced Balanced >"$RELEASE/GGUF/balanced/q4_0-pipeline.log" 2>&1 &
pid_balanced=$!
build_one 1 max Max >"$RELEASE/GGUF/max/q4_0-pipeline.log" 2>&1 &
pid_max=$!

status=0
wait "$pid_balanced" || status=1
wait "$pid_max" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "One or more Q4_0 builds failed." >&2
  exit "$status"
fi

rm -rf "$RUN/q4_sources/balanced" "$RUN/q4_sources/max"
rmdir "$RUN/q4_sources" 2>/dev/null || true
echo "QAT_Q4_RELEASE_COMPLETE"
