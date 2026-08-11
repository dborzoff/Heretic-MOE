# EN/RU language-refusal diagnostic design

## Goal

Determine whether the same canonical SAFE and UNSAFE requests trigger materially
different refusal behavior in English and Russian, before multilingual prompts
are allowed to influence Heretic-MOE direction construction or optimization.

## Scope

- Primary model: original
  `F:/AI/hf_originals/LLM/google__gemma-4-E4B-it` in BF16. This model has
  already completed a local Heretic-MOE search on the target RTX 4090 setup.
- Confirmation model: original `F:/AI/hf_originals/LLM/Qwen__Qwen3.5-9B`
  in BF16, used only for behavioral discordances and a stratified concordant
  control sample unless the primary result is inconclusive.
- Inputs: 1,400 EN SAFE, 1,400 EN UNSAFE, and their aligned 1,400-row RU
  translations for each direction.
- Generation: greedy, thinking disabled, maximum 256 new tokens. Any response
  stopped by the token limit is rerun once with a 512-token limit.
- Console and machine reports are text-free. Prompt and response text is stored
  only in the local response archive.

BF16 is mandatory for the operative result. A GGUF run may be used as a speed
probe but cannot establish residual geometry and is not a substitute.

## Analysis

Rows are joined by `canonical_id`; translations are never treated as independent
samples. Report paired EN/RU transitions separately for SAFE, UNSAFE, and each
category:

- delivered in both languages;
- refusal/evasion in both languages;
- delivered in EN but refusal/evasion in RU;
- refusal/evasion in EN but delivered in RU;
- unclear because either response is truncated or cannot be classified.

The first pass uses deterministic mathematical features only as a radar. Every
discordant pair and a deterministic stratified sample of concordant pairs is
prepared for blind semantic audit. No language is added to the optimization
objective from radar labels alone.

If behavior differs materially, first repeat discordant pairs plus a stratified
concordant control on Qwen3.5-9B BF16. Then measure per-layer EN/RU residual
directions on Gemma-4 E4B and aligned prompt IDs. Compare direction cosine, norm
ratio, and paired projection signs by layer. Only a cross-model behavioral
difference with supporting residual evidence is sufficient to add Russian to
the search objective.

## Gates

- Exact 1:1 canonical coverage and category agreement between EN and RU.
- No empty prompts, duplicate row IDs, or prompt text in console output.
- Model/tokenizer/config hashes and generation settings frozen in a manifest.
- All generated rows present; capped rows either rerun at 512 or marked unclear.
- Final conclusion includes paired rates with canonical-ID bootstrap intervals.
- If RU SAFE is incomplete, run EN/RU UNSAFE first and append SAFE without
  changing model or generation settings.
