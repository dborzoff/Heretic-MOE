---
license: gemma
base_model: google/gemma-3-12b-it-qat-q4_0-unquantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - gemma3
  - heretic-moe
  - ltx-video
  - text-encoder
  - gguf
  - comfyui
  - quantized
---

# Gemma 3 12B IT QAT — Heretic-MOE v1

Work-in-progress release of Gemma 3 12B IT QAT variants produced with
[Heretic-MOE](https://github.com/dborzoff/heretic-moe). The release is intended
for two related uses:

1. a full language model for prompt rewriting and directing;
2. a native ComfyUI text encoder for LTX workflows.

The repository is being prepared while the final adaptive search and validation
are still running. Files marked **Planned** are not available yet. A file will be
marked **Validated** only after its checksum, loader test, and task-specific
comparison have passed.

## Search finalists

| Release variant | Search role | Trial | Refusal geometry | Keyword flags | Absolute PPL drift | LTX conditioning drift | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Value-balanced | Best preservation/removal value | TBD | TBD | TBD | TBD | TBD | Search running |
| Max-removal | Stronger refusal removal within preservation limits | TBD | TBD | TBD | TBD | TBD | Search running |

The search metrics are selection signals, not final behavioral claims. Final
variants will also be evaluated with longer generations and blind semantic
judging.

## Full-model Director files

| File | Format | Importance matrix | Intended use | Status |
|---|---|---|---|---|
| `director/<variant>/Gemma-3-12B-Heretic-MOE-<variant>-Q8_0.gguf` | GGUF Q8_0 | Variant-specific matrix recorded separately | Higher-quality local directing and prompt rewriting | Planned |
| `director/<variant>/Gemma-3-12B-Heretic-MOE-<variant>-IQ4_XS.gguf` | GGUF IQ4_XS | Yes, variant-specific | Compact local directing and prompt rewriting | Planned |
| `imatrix/<variant>/Gemma-3-12B-Heretic-MOE-<variant>.imatrix` | llama.cpp importance matrix | N/A | Reproducible local GGUF quantization | Planned |

The intermediate F16 GGUF master is used during conversion. Whether it is
published will be decided after final file-size and storage checks.

## LTX text encoders

| File | Weight format | Compute target | Intended hardware | Status |
|---|---|---|---|---|
| `ltx_text_encoders/<variant>/Gemma-3-12B-Heretic-MOE-<variant>-LTX-TE-BF16.safetensors` | BF16 master | Native ComfyUI | High-memory GPU or offload | Planned |
| `ltx_text_encoders/<variant>/Gemma-3-12B-Heretic-MOE-<variant>-LTX-TE-INT8-ConvRot.safetensors` | INT8 ConvRot language matrices, preserved auxiliary tensors | Native ComfyUI INT8 | RTX 30/40/50 series; exact VRAM TBD | Planned |
| `ltx_text_encoders/<variant>/Gemma-3-12B-Heretic-MOE-<variant>-LTX-TE-NVFP4.safetensors` | Block-scaled NVFP4 language matrices, preserved auxiliary tensors | Native ComfyUI NVFP4 | RTX 50 series; exact VRAM TBD | Planned |

The text-encoder files retain the Gemma tokenizer and the tensor layout expected
by the stock ComfyUI Gemma loader. Quantization is performed from the exported
Heretic-MOE master, not from another quantized artifact.

## Hardware table

Measured values will replace the placeholders below after the final files are
loaded in real workflows.

| Variant | File size | Peak VRAM | System RAM | Hardware tested | Throughput | Status |
|---|---:|---:|---:|---|---:|---|
| Director Q8_0 | TBD | TBD | TBD | TBD | TBD | Pending |
| Director IQ4_XS | TBD | TBD | TBD | TBD | TBD | Pending |
| LTX TE BF16 | TBD | TBD | TBD | TBD | TBD | Pending |
| LTX TE INT8-ConvRot | TBD | TBD | TBD | TBD | TBD | Pending |
| LTX TE NVFP4 | TBD | TBD | TBD | RTX 50 series | TBD | Pending |

## Validation matrix

| Check | Value-balanced | Max-removal |
|---|---|---|
| Adaptive search complete | Pending | Pending |
| Export/reload equality | Pending | Pending |
| SHA-256 manifest | Pending | Pending |
| llama.cpp Q8_0 load | Pending | Pending |
| llama.cpp IQ4_XS load | Pending | Pending |
| Stock ComfyUI BF16 TE load | Pending | Pending |
| Stock ComfyUI INT8-ConvRot TE load | Pending | Pending |
| Stock ComfyUI NVFP4 TE load | Pending | Pending |
| LTX conditioning comparison | Pending | Pending |
| Long-generation behavioral test | Pending | Pending |
| Blind semantic review | Pending | Pending |

## Planned repository layout

```text
director/
  value-balanced/
  max-removal/
imatrix/
  value-balanced/
  max-removal/
ltx_text_encoders/
  value-balanced/
  max-removal/
reports/
  manifests/
  validation/
```

## Reproducibility

The release pipeline is maintained in the
[Heretic-MOE repository](https://github.com/dborzoff/heretic-moe):

- `research/scripts/build_gemma3_ltx_te.py`
- `research/scripts/quantize_gemma3_ltx_te.py`
- `research/scripts/run_remote_gemma3_release_pipeline.sh`

Every published model file will be accompanied by its source trial, conversion
configuration, SHA-256 checksum, and validation status.

## Current status

This repository is a release scaffold. Do not treat the current placeholders as
benchmark results or the planned filenames as completed artifacts.
