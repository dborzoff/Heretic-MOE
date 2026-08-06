# MiniMax-H3 Qwen3-VL-32B layer trace

## Goal

Measure which Qwen language layers and projection families materially affect MiniMax-H3 conditioning, then use that evidence to choose safe truncation and mixed-precision profiles. The test must distinguish text-only conditioning from visual DeepStack conditioning.

## Frozen official inputs

Official request examples are stored at:

`F:\AI\hf_originals\MiniMax__MiniMax-H3\HF\scripts\readme`

Use the reproducible 768p request structure for three independent modes:

1. T2VA: text only.
2. FL2VA/I2VA: text plus image conditioning.
3. Ref2VA: text plus reference video and audio metadata.

Prompt or answer text must never be printed to the console or copied into numeric reports.

## Architecture facts to verify at runtime

- The H3 language path executes Qwen language blocks 0 through 49.
- The H3 output is the unnormalized hidden state after block 49 (the 50th block).
- Language blocks 50 through 63, final language norm, and LM head do not contribute to H3 conditioning.
- The complete 27-block vision tower executes for image/video inputs.
- Vision DeepStack outputs from vision blocks 8, 16, and 24 are injected around the first three language blocks.
- Audio waveforms are handled by H3 AudioVAE; Qwen receives only presentation text/metadata for audio references.

## FL2VA VAE boundary

- `video_vae` is a separate pixel/video-latent codec, not part of Qwen. Its
  checkpoint is 10,415,548,320 bytes: 2,603,871,032 FP32 parameters, including
  roughly 180M encoder and 2.424B decoder parameters.
- Video latents have 24 channels. The configured compression is 16x spatially
  and 4x temporally; the decoder includes 36 ViT blocks. Tiling and temporal
  chunking are supported.
- `audio_vae` is also separate from Qwen. Its checkpoint is 605,429,308 bytes:
  151,326,585 FP32 parameters. It is a DAC/BigVGAN-style 32 kHz stereo codec
  with 32 latent channels.
- Qwen-only conditioning comparisons must run before either VAE so VAE drift
  cannot be mistaken for Qwen-layer influence. The VAEs are loaded only for
  end-to-end MiniMax validation of shortlisted interventions.

## Stage A: execution trace

Run one BF16 reference forward per input mode with hooks that record only numeric metadata:

- executed module names and counts;
- input/output shapes;
- per-layer hidden-state L2 norm;
- residual delta norm and cosine;
- separate statistics for text-token and vision-token positions;
- DeepStack injection positions and tensor shapes;
- peak VRAM/RAM and wall time.

This stage proves what executes, but it does not prove causal importance.

## Stage B: causal layer sweep

For every language block 0..49, run paired interventions against the untouched BF16 reference:

1. residual bypass of the whole block;
2. attention-output bypass only;
3. MLP-output bypass only.

For each intervention, measure change in the final unnormalized layer-50 conditioning:

- relative L2 error;
- cosine similarity;
- maximum absolute error;
- text-token error;
- vision-token error;
- error after H3's text-input projection when the projection weights are available.

Do not infer quality from activation magnitude alone. Rank layers by causal output change.

## Stage C: quantization sensitivity

For the most important and least important layers from Stage B, substitute one tensor family at a time:

- gate_proj;
- up_proj;
- down_proj;
- q_proj;
- k_proj;
- v_proj;
- o_proj.

Compare INT8 and W4 candidates against BF16. Rank each tensor by conditioning damage per GiB saved. Then evaluate complete mixed profiles because individual errors are not additive.

## Stage D: edited-model comparison

Run the untouched Qwen reference and each surviving Heretic finalist through the identical H3 presentation. Report:

- layerwise divergence from the original;
- final conditioning divergence;
- whether divergence concentrates in text or vision positions;
- whether DeepStack fusion changes disproportionately;
- empty/NaN/Inf checks.

This stage must use the finalist selected after the 136x2048 semantic judgment. Do not select a finalist from marker or PPL metrics alone.

## Stage E: end-to-end MiniMax validation

Only the best two or three conditioning profiles proceed to full MiniMax-H3 generation. Use fixed seeds and the official request modes, then compare:

- output completion and technical validity;
- conditioning and generation wall time;
- peak VRAM/RAM;
- visual/reference adherence;
- obvious temporal, audio, or identity regressions.

Full MiniMax generation is not required for every layer intervention.

## Compute plan

- Static checkpoint surgery and hash/shape verification can run locally in system RAM.
- Reference BF16 layer tracing is best run on the rented 96 GB GPU after the current Qwen semantic generation is complete and copied locally.
- Two local 24 GB GPUs do not automatically combine into a single 48 GB device in stock ComfyUI. Use explicit sharding/offload only if the remote 96 GB path is unavailable.
- Keep the current rental alive until the semantic archives, blind judge packet, and layer-trace inputs are safely copied and hashed.

## Outputs

All reports must be text-free:

- `execution_trace.json`
- `layer_causal_sweep.parquet`
- `projection_quant_sweep.parquet`
- `finalist_conditioning_comparison.json`
- `end_to_end_summary.json`
- `manifest.json` with source revisions and SHA-256 values
