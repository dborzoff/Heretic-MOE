#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

repo=/workspace/heretic-gemma3/heretic-moe-qwen-night-abs
run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_qwen_night_abs_600_v1
selected="${run_root}/selected_models"
release="${run_root}/release"
toolchain=/workspace/heretic-gemma3/toolchain
quant_tool="${toolchain}/quant-venv/bin/convert-to-quant"
base_tokenizer=/workspace/heretic-gemma3/models/google__gemma-3-12b-it-qat-q4_0-unquantized/tokenizer.model

mkdir -p "${release}/Text_Encoder/balanced" "${release}/Text_Encoder/max" "${release}/research/validation"

build_bf16() {
  local variant="$1"
  local source="$2"
  local output="$3"
  local report="$4"
  if [[ -s "${output}" && -s "${report}" ]]; then
    echo "stage=bf16-skip variant=${variant}"
    return
  fi
  rm -f -- "${output}" "${output}.tmp" "${report}"
  echo "stage=bf16-start variant=${variant}"
  /venv/main/bin/python "${repo}/research/scripts/build_gemma3_ltx_te.py" \
    --src "${source}" \
    --dst "${output}" \
    --tokenizer-source "${base_tokenizer}" \
    --report "${report}"
  echo "stage=bf16-complete variant=${variant}"
}

balanced_bf16="${release}/Text_Encoder/balanced/Gemma-3-12B-IT-QAT-HereticMOE-Balanced-BF16.safetensors"
max_bf16="${release}/Text_Encoder/max/Gemma-3-12B-IT-QAT-HereticMOE-Max-BF16.safetensors"

build_bf16 balanced "${selected}/balanced_t555/model" "${balanced_bf16}" "${release}/research/validation/balanced-bf16.json" \
  >"${release}/Text_Encoder/balanced/bf16.log" 2>&1 &
balanced_build_pid=$!
build_bf16 max "${selected}/max_t752/model" "${max_bf16}" "${release}/research/validation/max-bf16.json" \
  >"${release}/Text_Encoder/max/bf16.log" 2>&1 &
max_build_pid=$!

status=0
wait "${balanced_build_pid}" || status=1
wait "${max_build_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "One or more BF16 TE builds failed."
  exit "${status}"
fi

echo "stage=wait-quant-toolchain"
for _ in $(seq 1 180); do
  [[ -x "${quant_tool}" ]] && break
  sleep 10
done
[[ -x "${quant_tool}" ]] || { echo "Quant toolchain did not become ready."; exit 4; }

quantize_variant() {
  local device="$1"
  local variant="$2"
  local label="$3"
  local input="$4"
  local output_dir="$5"
  local int8="${output_dir}/Gemma-3-12B-IT-QAT-HereticMOE-${label}-INT8-ConvRot.safetensors"
  local nvfp4="${output_dir}/Gemma-3-12B-IT-QAT-HereticMOE-${label}-NVFP4.safetensors"

  export CUDA_VISIBLE_DEVICES="${device}"
  if [[ ! -s "${int8}" ]]; then
    echo "stage=int8-start variant=${variant} gpu=${device}"
    "${toolchain}/quant-venv/bin/python" "${repo}/research/scripts/quantize_gemma3_ltx_te.py" \
      --tool "${quant_tool}" --input "${input}" --output "${int8}" \
      --format int8-convrot --device cuda \
      --report "${release}/research/validation/${variant}-int8-convrot.json"
    echo "stage=int8-complete variant=${variant} gpu=${device}"
  fi
  if [[ ! -s "${nvfp4}" ]]; then
    echo "stage=nvfp4-start variant=${variant} gpu=${device}"
    "${toolchain}/quant-venv/bin/python" "${repo}/research/scripts/quantize_gemma3_ltx_te.py" \
      --tool "${quant_tool}" --input "${input}" --output "${nvfp4}" \
      --format nvfp4 --device cuda \
      --report "${release}/research/validation/${variant}-nvfp4.json"
    echo "stage=nvfp4-complete variant=${variant} gpu=${device}"
  fi
}

quantize_variant 0 balanced Balanced "${balanced_bf16}" "${release}/Text_Encoder/balanced" \
  >"${release}/Text_Encoder/balanced/quantize.log" 2>&1 &
balanced_quant_pid=$!
quantize_variant 1 max Max "${max_bf16}" "${release}/Text_Encoder/max" \
  >"${release}/Text_Encoder/max/quantize.log" 2>&1 &
max_quant_pid=$!

status=0
wait "${balanced_quant_pid}" || status=1
wait "${max_quant_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "One or more TE quantization branches failed."
  exit "${status}"
fi

echo "All Gemma 3 TE variants completed."
