# Qwen3.6 Heretic-MOE v3 NVFP4 W4A16/W4A4 release design

## Goal

Publish two unambiguous NVFP4 families for both selected Heretic-MOE v3 variants:

- `Balanced-NVFP4_W4A16.safetensors`
- `Balanced-NVFP4_W4A4.safetensors`
- `Max-NVFP4_W4A16.safetensors`
- `Max-NVFP4_W4A4.safetensors`

The existing ambiguous `Balanced-NVFP4.safetensors` and `Max-NVFP4.safetensors` paths must not remain in the Hugging Face repository. "NVIDIA-exact W4A4" means the same quantization architecture and tensor schema as NVIDIA's reference Qwen3.6 NVFP4 checkpoint, not byte-identical weights: the Heretic-MOE Balanced and Max masters are intentionally different models.

## Source and quantization rules

Each quant is built independently from its corresponding single-file BF16 master. W4A4 is never derived from W4A16.

- **W4A16:** preserve the already-built artifacts. They use NVFP4 weights for the selected MoE/MLP linears, FP8 attention/KV handling, and do not quantize the corresponding MLP input activations to NVFP4.
- **W4A4:** reproduce NVIDIA's effective artifact schema: NVFP4 weights and NVFP4 input activations for the selected MoE/MLP linears, with the same FP8 attention/KV module policy as the reference checkpoint.
- Vision, embeddings, norms, shared non-quantized tensors, and the restored MTP block retain the reference policy. MTP remains excluded from the calibrated quantized module set.
- The same frozen, text-free calibration contract used for the current GPU quants is reused. No test prompt or response text is emitted to logs or reports.

## Hugging Face layout and migration

The current W4A16 LFS objects are renamed server-side to the explicit `_W4A16` paths. Hugging Face represents a path rename as one atomic repository commit that adds the existing LFS object at the new path and removes only the old tree path. The 23.4 GB payload is not uploaded again and its LFS SHA-256 does not change. No old-name aliases remain.

The new W4A4 artifacts are uploaded alongside W4A16 under the same variant folders. Upload is resumable and each file gets a manifest containing source master hash, recipe hash, ModelOpt commit, output size, output SHA-256, tensor counts, quantizer counts, and validation results.

The model card will explain:

- W4A16 is the quality-first NVFP4 option;
- W4A4 is the activation-quantized, Blackwell-oriented performance option;
- the two formats need loader support for their recorded ModelOpt quantization metadata;
- W4A4 matches NVIDIA's quantization scheme, while its weights remain the Heretic-MOE Balanced or Max weights.

## Verification gates

An artifact is not published as complete until all applicable checks pass:

1. Safetensors opens and every tensor has a finite dtype/shape/byte range.
2. Architecture fields match the official Qwen3.6-35B-A3B NVFP4 reference.
3. The normalized quantized-module map matches the reference: 291 modules, consisting of 161 NVFP4-policy modules and 130 FP8-policy modules.
4. W4A4 contains the reference activation-quantization schema, including the expected `input_scale` tensors and enabled input quantizers; W4A16 demonstrably does not masquerade as W4A4.
5. MTP is present in both variants and excluded from calibration according to the recorded policy.
6. Balanced and Max output hashes differ, proving that one variant was not accidentally copied over the other.
7. Hugging Face reports the exact four final NVFP4 paths, correct sizes and LFS SHA-256 values, and no old ambiguous NVFP4 paths.
8. The final card and research manifests describe the formats consistently.

## IQ2 completion retained in the same release

The ongoing GGUF work remains independent from the NVFP4 rename/build. Balanced and Max use `IQ2_XXS` for the ordinary model layers and `Q4_K` for the eleven MTP weights in `blk.40`, because an inference calibration imatrix cannot cover the inactive MTP head. The release records this exception and preserves the failed first attempt as diagnostic evidence. Both IQ2 files require `llama-gguf` validation, SHA-256 verification, and successful HF upload.

## Failure handling and cleanup

- Do not remove an old HF tree path before the same LFS object is verified at its explicit W4A16 destination.
- Do not label a file W4A4 if activation-quantization tensors or quantizers are missing.
- Preserve failed conversion logs and manifests, but never publish partial weight files.
- Keep the paid server until all required artifacts and metadata are verified on HF; then destroy the instance immediately.
