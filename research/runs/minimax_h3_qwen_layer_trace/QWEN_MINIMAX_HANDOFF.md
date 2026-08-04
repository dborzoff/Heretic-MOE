# Qwen3-VL-32B and MiniMax-H3 handoff

This report contains paths, hashes, and aggregate measurements only. It does
not contain evaluation prompts or model responses.

## Qwen finalist validation

The frozen 400 x 512 BF16 perplexity report is:

`research/results/qwen3vl32b_hybrid_600_v1_full_136x2048/metadata/ppl_400x512.json`

Status: PASS. Each model was evaluated on 204,400 target tokens.

| System | Perplexity | Relative to original |
|---|---:|---:|
| original | 8.4323614303 | 0.0000% |
| max-removal | 8.4281915121 | -0.0495% |
| balanced | 8.4267780241 | -0.0662% |
| marker-zero | 8.4345372996 | +0.0258% |

All four 136 x 2048 response archives are complete and have zero empty
prompt/answer rows. SHA-256:

- original: `5a16691c747b83cb990fd4ee28d107c06fd00099db848b6fa45eee96a60fe524`
- max-removal: `ff9c9258f349bc24effd36a6b2d756975d98995302f6b98c0d31fb8e17c45e34`
- balanced: `bb0a77fa07cb929db1ab796674fed24bdd2a3b5f3be6ba9c85662e27f7f8a2a5`
- marker-zero: `bb4927054245757a7e158e0f7969c9bec624fd3ab2b5c1257ef8b7c97e6a1d2e`

The blind semantic packet is frozen at:

`research/results/qwen3vl32b_hybrid_600_v1_full_ninelabel_rejudge/build_v1/judge_packet/manifest.json`

- systems: 4
- rows per system: 136
- expected/unique rows: 544/544
- prompt identity: true
- empty prompt or answer: 0
- judge input SHA-256:
  `10d0b78e4249966ea61ccc227642f928797ad179b2b43419210ebdd7f00e364e`
- judge manifest SHA-256:
  `b7f7c24e94b2720290c231b4b1c520393707251057448fc84b8bac6003968456`
- status: PASS

No semantic winner is declared until blind judging and private-map verification
are complete.

## MiniMax-H3 repository and runtime

The official repository is present at:

`F:/AI/hf_originals/MiniMax__MiniMax-H3/HF`

The root repository index and README were refreshed on 2026-08-04. Official
Qwen text-encoder shards are byte-identical to
`Qwen/Qwen3-VL-32B-Instruct` and to the FL2VA/Ref2VA copies. FL2VA and Ref2VA
share one Qwen TE but use different generator-transformer weights.

The current rental server contains the FL2VA modular components at:

`/workspace/models/MiniMax-H3`

- generator transformer: 61.729 GiB
- video VAE: 9.700 GiB
- audio VAE: 0.564 GiB
- total partial snapshot: 72.063 GiB
- Qwen weights: intentionally not duplicated

The official Diffusers `minimax-h3` branch is installed at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`. Transformer, video VAE, and
audio VAE config/meta-device smoke tests pass.

## Test decision

MiniMax conditioning must be compared on both original and Heretic Qwen. The
original is the numerical reference. The Heretic candidate is selected only
after blind semantic judging; using all finalists in full MiniMax generation
would multiply an expensive test before the selection question is resolved.

Recommended order:

1. Original BF16 conditioning reference in all five modes.
2. BF16 conditioning for the semantic Heretic winner.
3. Universal, Compact, and UltraCompact quantization sweeps on the better base.
4. Full FL2VA and Ref2VA generation only for shortlisted profiles.
