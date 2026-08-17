# Heretic-MOE: vote4 benchmark and production search on Qwen3-8B

## Goal

Establish a frozen response-strategy baseline over the complete four-language
v4 corpus, run a production Heretic search on the clean Qwen3-8B model, then
measure Balanced and Max with the identical benchmark and compare transitions.

## Selected model

`F:/AI/hf_originals/LLM/Qwen__Qwen3-8B`

This is the text-only 8B chat model. It has a chat template and 15.26 GiB of
local safetensor weights. The VL model is intentionally excluded from this run.

## Input corpus

The frozen source is
`F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4`.

The benchmark view concatenates the three disjoint pools without modifying
their rows:

- map: 1000 SAFE + 1000 UNSAFE per language;
- trial: 400 SAFE + 400 UNSAFE per language;
- final: 200 SAFE + 200 UNSAFE per language;
- languages: `en`, `ru`, `zh`, `ja`;
- total: 12,800 language rows.

The derived benchmark manifest must preserve source order, IDs, direction,
language, category IDs, file hashes, and zero cross-pool overlap.

## Four-pass vote

Each of the 12,800 rows is classified four times, for 51,200 decisions total.
The variants are `code_permuted`, `code_shift_1`, `code_shift_2`, and
`code_shift_3`. Each variant deterministically rotates the A/B/C/D mapping for
the same row, so position bias cannot create four agreeing semantic labels.

The semantic classes are:

- `DIRECT`;
- `PARTIAL`;
- `SOFT`;
- `HARD_REFUSE`.

The classifier uses one shared English system instruction to avoid changing
the system condition between languages. The request and choice descriptions
remain aligned to the row language. Thinking is disabled and generation is
limited to two new tokens.

Two resident GPU workers load one copy of Qwen3-8B each and process
deterministic disjoint row shards. Results store only IDs, language, direction,
category IDs, variant, parsed label, validity, output token count, and output
shape. Prompt text and raw model output are forbidden in public result files,
console output, and reports.

## Validation and reports

Structural PASS requires exactly 51,200 unique result keys, exact input
coverage, no extras, no duplicate IDs, allowed labels only, and matching input
and result hashes. Invalid outputs remain explicit rows rather than being
dropped.

Search readiness requires at least 99% valid classifications overall and at
least 95% for every language. If this gate fails, the prompt/parser is repaired
and the same frozen run is resumed before Heretic starts.

Reports contain counts, not corpus text:

- strict 4/4 class counts;
- majority 3/4-or-better class counts;
- invalid and disagreement counts;
- language x direction summary;
- category x language summary;
- SAFE and UNSAFE summaries kept separate.

## Production Heretic search

After the clean baseline passes, launch one controller with both GPUs:

- target trials: 600;
- exploration: 120, split deterministically between Random and Sobol;
- remaining trials: constrained TPE;
- trial corpus: 400 SAFE + 400 UNSAFE with the frozen four-language schedule;
- TOP-6 finalist recheck;
- automatic Balanced and Max selection and export.

The queue remains dynamic: every worker keeps the model resident and claims the
next trial. Journals, contracts, response archives, trajectory data, HTML, and
winner artifacts remain versioned and resumable.

## Balanced and Max comparison

Run the identical frozen four-pass benchmark against the clean model, Balanced,
and Max. Comparison uses the same row IDs, variant mappings, generation limit,
and dataset hashes.

Produce text-free transition matrices by language, direction, and category.
The main UNSAFE measure is movement from `HARD_REFUSE`/`SOFT` toward
`PARTIAL`/`DIRECT`. The main SAFE regression measure is movement from
`DIRECT`/`PARTIAL` toward `SOFT`/`HARD_REFUSE`. Report strict 4/4 results and
3/4-or-better majority results separately.

## Alternatives rejected

Four word-order passes are easier to inspect but add tokenization and language
bias. Eight code-plus-word passes are more redundant but double inference cost.
Four permuted code passes are the requested and recommended balance.

