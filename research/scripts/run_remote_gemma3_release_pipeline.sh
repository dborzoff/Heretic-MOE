#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:?usage: run_remote_gemma3_release_pipeline.sh SOURCE VARIANT OUTPUT_ROOT CALIBRATION TE_REFERENCE}
VARIANT=${2:?missing VARIANT}
OUTPUT_ROOT=${3:?missing OUTPUT_ROOT}
CALIBRATION=${4:?missing CALIBRATION}
TE_REFERENCE=${5:?missing TE_REFERENCE}

ROOT=${WORKSPACE_ROOT:-/workspace/heretic-gemma3}
REPO=${HERETIC_REPO:-$ROOT/heretic-moe}
LLAMA=${LLAMA_ROOT:-/workspace/llama.cpp}
COMFY=${COMFY_ROOT:-/workspace/ComfyUI}
QUANT_TOOLS=${QUANT_TOOLS_ROOT:-/workspace/minimax_te/tools}
PYTHON=${PYTHON_BIN:-$REPO/.venv/bin/python}
TOOLS=$REPO/research/scripts
WORK=$OUTPUT_ROOT/$VARIANT
REPORT=$WORK/reports
GGUF=$WORK/full_model_gguf
TE=$WORK/ltx_text_encoders

F16=$GGUF/Gemma-3-12B-Heretic-MOE-$VARIANT-F16.gguf
Q8=$GGUF/Gemma-3-12B-Heretic-MOE-$VARIANT-Q8_0.gguf
IQ4=$GGUF/Gemma-3-12B-Heretic-MOE-$VARIANT-IQ4_XS.gguf
IMATRIX=$GGUF/Gemma-3-12B-Heretic-MOE-$VARIANT.imatrix
TE_BF16=$TE/Gemma-3-12B-Heretic-MOE-$VARIANT-LTX-TE-BF16.safetensors
TE_INT8=$TE/Gemma-3-12B-Heretic-MOE-$VARIANT-LTX-TE-INT8-ConvRot.safetensors
TE_NVFP4=$TE/Gemma-3-12B-Heretic-MOE-$VARIANT-LTX-TE-NVFP4.safetensors

mkdir -p "$REPORT" "$GGUF" "$TE"
exec > >(tee -a "$REPORT/pipeline.log") 2>&1

echo "OPENAI CODEX | GEMMA3 HERETIC RELEASE | variant=$VARIANT"
echo "started=$(date --iso-8601=seconds)"
for path in "$SOURCE" "$CALIBRATION" "$TE_REFERENCE"; do
  [[ -e "$path" ]] || { echo "missing=$path"; exit 2; }
done
for path in "$LLAMA/convert_hf_to_gguf.py" "$LLAMA/build/bin/llama-quantize" "$LLAMA/build/bin/llama-imatrix"; do
  [[ -e "$path" ]] || { echo "missing=$path"; exit 3; }
done

echo "stage=convert-f16"
[[ -s "$F16" ]] || "$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$SOURCE" --outtype f16 --outfile "$F16"

echo "stage=imatrix"
[[ -s "$IMATRIX" ]] || "$LLAMA/build/bin/llama-imatrix" \
  -m "$F16" -f "$CALIBRATION" -o "$IMATRIX" --output-format gguf \
  --no-ppl -ngl 999 -c 512 -b 512 --chunks 200

echo "stage=q8"
[[ -s "$Q8" ]] || "$LLAMA/build/bin/llama-quantize" "$F16" "$Q8" Q8_0

echo "stage=iq4"
[[ -s "$IQ4" ]] || "$LLAMA/build/bin/llama-quantize" --imatrix "$IMATRIX" "$F16" "$IQ4" IQ4_XS

echo "stage=te-bf16"
[[ -s "$TE_BF16" ]] || "$PYTHON" "$TOOLS/build_gemma3_ltx_te.py" \
  --source "$SOURCE" --output "$TE_BF16" --reference "$TE_REFERENCE"

echo "stage=te-int8-convrot"
[[ -s "$TE_INT8" ]] || CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TOOLS/quantize_gemma3_ltx_te.py" \
  "$TE_BF16" "$TE_INT8" --profile int8-convrot --comfy-root "$COMFY" \
  --tools-root "$QUANT_TOOLS" --device cuda:0

echo "stage=te-nvfp4"
[[ -s "$TE_NVFP4" ]] || CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$TOOLS/quantize_gemma3_ltx_te.py" \
  "$TE_BF16" "$TE_NVFP4" --profile nvfp4 --comfy-root "$COMFY" \
  --tools-root "$QUANT_TOOLS" --device cuda:0

sha256sum "$F16" "$IMATRIX" "$Q8" "$IQ4" "$TE_BF16" "$TE_INT8" "$TE_NVFP4" > "$REPORT/SHA256SUMS"
echo "status=PASS"
echo "finished=$(date --iso-8601=seconds)"
