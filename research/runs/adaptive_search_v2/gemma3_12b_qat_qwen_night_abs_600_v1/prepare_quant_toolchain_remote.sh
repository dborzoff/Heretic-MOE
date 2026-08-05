#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
root=/workspace/heretic-gemma3/toolchain
llama="${root}/llama.cpp"
quant_venv="${root}/quant-venv"
mkdir -p "${root}"

echo "stage=llama-source"
if [[ ! -d "${llama}/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${llama}"
fi

echo "stage=llama-configure"
cmake -S "${llama}" -B "${llama}/build" -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "stage=llama-build"
cmake --build "${llama}/build" --target llama-quantize llama-imatrix -j 24

echo "stage=quant-venv"
if [[ ! -x "${quant_venv}/bin/python" ]]; then
  /venv/main/bin/python -m venv --system-site-packages "${quant_venv}"
fi
"${quant_venv}/bin/python" -m pip install --upgrade pip
"${quant_venv}/bin/python" -m pip install \
  'convert_to_quant==1.3.1' \
  'comfy-kitchen==0.2.26' \
  'sentencepiece==0.2.1'

echo "stage=verify"
[[ -x "${llama}/build/bin/llama-quantize" ]]
[[ -x "${llama}/build/bin/llama-imatrix" ]]
[[ -x "${quant_venv}/bin/convert-to-quant" ]]
git -C "${llama}" rev-parse HEAD
echo "Quantization toolchain ready."
