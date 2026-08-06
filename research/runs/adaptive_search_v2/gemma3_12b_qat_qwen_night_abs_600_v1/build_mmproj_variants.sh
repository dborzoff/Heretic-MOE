#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/heretic-gemma3
RUN="$ROOT/runs/gemma3_12b_qat_qwen_night_abs_600_v1"
LLAMA="$ROOT/toolchain/llama.cpp"
PY=/venv/main/bin/python
OUT="$RUN/release/mmproj"

mkdir -p "$OUT/balanced" "$OUT/max" "$RUN/release/research/validation/mmproj"

build_one() {
  local variant="$1"
  local model="$2"
  local output="$OUT/$variant/Gemma-3-12B-Heretic-MOE-${variant^}-BF16.gguf"
  "$PY" "$LLAMA/convert_hf_to_gguf.py" \
    "$model" \
    --mmproj \
    --outtype bf16 \
    --outfile "$output" \
    >"$RUN/release/research/validation/mmproj/$variant.log" 2>&1
}

build_one balanced "$RUN/selected_models/balanced_t555/model" &
pid_balanced=$!
build_one max "$RUN/selected_models/max_t752/model" &
pid_max=$!
wait "$pid_balanced"
wait "$pid_max"

find "$OUT" -type f -name '*.gguf' -print0 | sort -z | xargs -0 sha256sum \
  >"$RUN/release/research/validation/mmproj/sha256sums.txt"

mapfile -t files < <(find "$OUT" -type f -name '*.gguf' | sort)
if [[ "${#files[@]}" -ne 2 ]]; then
  echo "Expected two mmproj outputs, found ${#files[@]}" >&2
  exit 1
fi

if cmp -s "${files[0]}" "${files[1]}"; then
  printf 'shared_identical=true\n' \
    >"$RUN/release/research/validation/mmproj/comparison.txt"
else
  printf 'shared_identical=false\n' \
    >"$RUN/release/research/validation/mmproj/comparison.txt"
  exit 2
fi

echo "MMProj variants complete and byte-identical."
