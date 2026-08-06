#!/bin/bash
set -euo pipefail

source /venv/main/bin/activate
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

repo=/workspace/heretic-gemma3/heretic-moe-qwen-night-abs
run_root=/workspace/heretic-gemma3/runs/gemma3_12b_qat_qwen_night_abs_600_v1
selected="${run_root}/selected_models"
release="${run_root}/release"
llama=/workspace/heretic-gemma3/toolchain/llama.cpp
quantize="${llama}/build/bin/llama-quantize"
imatrix_bin="${llama}/build/bin/llama-imatrix"
calibration="${repo}/src/heretic/data/perplexity_reference_v1.txt"
base=/workspace/heretic-gemma3/models/google__gemma-3-12b-it-qat-q4_0-unquantized

[[ -x "${quantize}" && -x "${imatrix_bin}" ]] || { echo "llama.cpp toolchain missing"; exit 3; }
[[ -s "${calibration}" ]] || { echo "calibration corpus missing"; exit 4; }
mkdir -p "${release}/GGUF/balanced" "${release}/GGUF/max" \
  "${release}/research/validation" "${release}/tmp"

augment_export() {
  local source="$1"
  for file in tokenizer.model added_tokens.json special_tokens_map.json preprocessor_config.json; do
    if [[ ! -e "${source}/${file}" && -e "${base}/${file}" ]]; then
      cp -- "${base}/${file}" "${source}/${file}"
    fi
  done
}

run_variant() {
  local device="$1"
  local variant="$2"
  local label="$3"
  local source="$4"
  local director_dir="${release}/GGUF/${variant}"
  local imatrix_dir="${release}/GGUF/${variant}"
  local work="${release}/tmp/${variant}"
  local f16="${work}/Gemma-3-12B-IT-QAT-HereticMOE-${label}-F16.gguf"
  local q8="${director_dir}/Gemma-3-12B-IT-QAT-HereticMOE-${label}-Q8_0.gguf"
  local iq4="${director_dir}/Gemma-3-12B-IT-QAT-HereticMOE-${label}-IQ4_XS.gguf"
  local matrix="${imatrix_dir}/Gemma-3-12B-IT-QAT-HereticMOE-${label}.imatrix"
  local report_dir="${release}/research/validation/${variant}-director"
  mkdir -p "${work}" "${report_dir}"

  augment_export "${source}"
  if [[ ! -s "${f16}" ]]; then
    echo "stage=f16-start variant=${variant}"
    /workspace/heretic-gemma3/toolchain/quant-venv/bin/python "${llama}/convert_hf_to_gguf.py" "${source}" \
      --outtype f16 --outfile "${f16}" >"${report_dir}/convert-f16.log" 2>&1
    echo "stage=f16-complete variant=${variant} bytes=$(stat -c %s "${f16}")"
  fi

  if [[ ! -s "${q8}" ]]; then
    "${quantize}" "${f16}" "${q8}" Q8_0 16 >"${report_dir}/q8.log" 2>&1 &
    q8_pid=$!
  else
    q8_pid=""
  fi
  if [[ ! -s "${matrix}" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="${device}"
      "${imatrix_bin}" -m "${f16}" -f "${calibration}" -o "${matrix}" \
        --output-format gguf --no-ppl -ngl 999 -c 512 -b 512 --chunks 200
    ) >"${report_dir}/imatrix.log" 2>&1 &
    matrix_pid=$!
  else
    matrix_pid=""
  fi
  [[ -z "${q8_pid}" ]] || wait "${q8_pid}"
  [[ -z "${matrix_pid}" ]] || wait "${matrix_pid}"
  [[ -s "${q8}" && -s "${matrix}" ]] || { echo "Q8 or imatrix missing for ${variant}"; exit 5; }
  echo "stage=q8-imatrix-complete variant=${variant} gpu=${device}"

  if [[ ! -s "${iq4}" ]]; then
    "${quantize}" --imatrix "${matrix}" "${f16}" "${iq4}" IQ4_XS 16 \
      >"${report_dir}/iq4.log" 2>&1
  fi
  [[ -s "${iq4}" ]] || { echo "IQ4 missing for ${variant}"; exit 6; }

  for model in "${q8}" "${iq4}"; do
    check="${work}/$(basename "${model}").check.imatrix"
    (
      export CUDA_VISIBLE_DEVICES="${device}"
      "${imatrix_bin}" -m "${model}" -f "${calibration}" -o "${check}" \
        --output-format gguf --no-ppl -ngl 999 -c 512 -b 512 --chunks 1
    ) >>"${report_dir}/load-check.log" 2>&1
    test -s "${check}"
    rm -f -- "${check}"
  done

  sha256sum "${q8}" "${iq4}" "${matrix}" >"${report_dir}/sha256sums.txt"
  rm -f -- "${f16}"
  rmdir "${work}" 2>/dev/null || true
  echo "stage=director-complete variant=${variant} gpu=${device}"
}

run_variant 0 balanced Balanced "${selected}/balanced_t555/model" \
  >"${release}/GGUF/balanced/pipeline.log" 2>&1 &
balanced_pid=$!
run_variant 1 max Max "${selected}/max_t752/model" \
  >"${release}/GGUF/max/pipeline.log" 2>&1 &
max_pid=$!

status=0
wait "${balanced_pid}" || status=1
wait "${max_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "One or more Director pipelines failed."
  exit "${status}"
fi

echo "All Gemma 3 Director variants completed."
