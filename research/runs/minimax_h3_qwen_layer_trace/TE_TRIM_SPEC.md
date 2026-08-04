# MiniMax-H3 universal Qwen TE trim specification

## Scope

The target checkpoint must support all official MiniMax-H3 modes:

- FL2VA partition: T2VA and first/last-frame FL2VA.
- Ref2VA partition: image, video, and audio reference conditions.

FL2VA and Ref2VA use the same Qwen3-VL-32B text encoder. The 14 official
shards have identical Hugging Face LFS SHA-256 values in the root, FL2VA, and
Ref2VA subfolders. One universal TE checkpoint is sufficient.

## Runtime mode matrix

| Mode | Token embedding | LM blocks 0-49 | Vision blocks 0-26 | DeepStack mergers | Final vision merger |
|---|---:|---:|---:|---:|---:|
| T2VA text only | yes | yes | no | no | no |
| FL2VA image(s) | yes | yes | yes | yes | yes |
| Ref2VA image | yes | yes | yes | yes | yes |
| Ref2VA video | yes | yes | yes, once per two sampled frames | yes | yes |
| Ref2VA audio only | yes | yes | no | no | no |

Audio samples do not enter Qwen. Qwen receives the textual audio presentation
label; the audio signal is handled by the separate H3 AudioVAE path.

## Guaranteed static trim

The MiniMax conditioning output is the unnormalized hidden state after the
50th language block (zero-based block index 49). The following tensors are
guaranteed unused by MiniMax-H3 and can be removed without an ablation:

- `model.language_model.layers.50.*` through
  `model.language_model.layers.63.*`
- `model.language_model.norm.weight`
- `lm_head.weight`

Original BF16 tensor bytes: 66,714,780,128 (62.1330 GiB).

- Removed language blocks 50-63: 13,652,753,408 bytes (12.7151 GiB).
- Removed LM head: 1,555,824,640 bytes (1.4490 GiB).
- Removed final norm: 10,240 bytes.
- Total guaranteed removal: 15,208,588,288 bytes (14.1641 GiB, 22.796%).
- Universal all-mode BF16 TE: 51,506,191,840 tensor bytes (47.9689 GiB).

## Required tensor families

Always keep:

- `model.language_model.embed_tokens.weight`
- `model.language_model.layers.0.*` through
  `model.language_model.layers.49.*`

Each language block contains 11 tensors:

- `input_layernorm.weight`
- `post_attention_layernorm.weight`
- `self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`
- `self_attn.q_norm.weight`, `k_norm.weight`
- `mlp.gate_proj.weight`, `up_proj.weight`, `down_proj.weight`

For a universal FL2VA plus Ref2VA checkpoint also keep:

- `model.visual.patch_embed.*`
- `model.visual.pos_embed.weight`
- `model.visual.blocks.0.*` through `model.visual.blocks.26.*`
- `model.visual.deepstack_merger_list.0.*` through
  `model.visual.deepstack_merger_list.2.*`
- `model.visual.merger.*`

The vision tower is sequential. DeepStack features are collected after vision
blocks 8, 16, and 24 and injected into language blocks 0, 1, and 2. Therefore
none of the 27 vision blocks can be deleted statically for an all-mode file.

## Further reduction requires measurement

Trimming alone does not fit the universal BF16 TE on a 24 GiB GPU. The next
safe reduction is quantization, not deleting more blocks blindly.

Measurement order:

1. Establish an exact BF16 conditioning reference for T2VA, FL2VA image,
   Ref2VA image, Ref2VA video, and Ref2VA audio-only.
2. Quantize one tensor family at a time and compare layer-49 conditioning:
   relative L2, cosine distance, maximum error, and text/vision-token splits.
3. Test LM `gate_proj`/`up_proj` first, then Q/K/V, then `down_proj`/`o_proj`.
4. Test vision blocks and all four vision support groups independently.
5. Run full MiniMax generation only for shortlisted profiles with fixed inputs
   and seeds.

Deleting or bypassing any block 0-49 is experimental model pruning. It is not
part of the guaranteed static trim and must pass all five mode gates.

## Release profiles

All three profiles retain every runtime path and support both FL2VA and
Ref2VA. Sizes below are tensor-payload estimates before quantization scales,
alignment, and file metadata.

### Universal (24 GiB GPUs)

- Token embedding and full vision tower: BF16.
- LM Q/K/V/O and `down_proj`: INT8.
- LM `gate_proj` and `up_proj`: W4.
- Estimated tensor payload: 19.16 GiB.
- Target packaged size: at most 20 GiB.

### Compact (16 GiB GPUs)

- Full vision tower: BF16.
- Token embedding: INT8.
- LM `o_proj`: INT8.
- LM Q/K/V and all MLP projections: W4.
- Estimated tensor payload: 14.16 GiB.
- Target packaged size: at most 14.7 GiB, leaving runtime headroom.

### UltraCompact (12 GiB GPUs)

- Vision tower: INT8.
- Token embedding and LM Q/K/V/O: W4.
- Large LM `gate_proj`, `up_proj`, and `down_proj`: approximately 3-bit.
- Estimated tensor payload: 9.98 GiB before group-scale overhead.
- Target packaged size: at most 10.8 GiB.

UltraCompact is an experimental profile. It must not be released from weight
error alone; it needs conditioning tests in all five modes and end-to-end
FL2VA/Ref2VA validation.
