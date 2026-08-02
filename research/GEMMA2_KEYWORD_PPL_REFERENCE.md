# Gemma 2 2B Keyword + Perplexity reference

This document records the public code revision and search configuration used
for an early Gemma 2 2B reference build. The generated model artifact is not
part of this repository.

## Code provenance

- Heretic revision: `613e230d4f5988f7f7ba46122b9a0e27bc6ec737`
- Revision date: 2026-07-29 17:32:18 +03:00
- Base architecture: `google/gemma-2-2b-it`
- Row normalization: `FULL`
- Objectives: `KeywordRate` and relative perplexity increase

The selected trial completed after this revision was committed and before the
next repository commit. The exported artifact was written during the same
interval. The following commit, `cfb6ea1`, translated comments and added
research material; it did not change the numerical weight-editing path used by
this reference run.

## Fork fixes present at the reference revision

- Added relative perplexity as a second optimization objective instead of
  relying only on refusal-keyword matches.
- Corrected the namespaced Wikitext dataset identifier used by the perplexity
  scorer.
- Preserved command-line settings when resuming an Optuna study.
- Gave abliterable components independent weight schedules.
- Added support for fused MoE experts stored as batched 3D parameters and gave
  routed experts a separate component key.
- Added seeding from the Pareto front of an earlier journal after changing an
  objective or search space.
- Added warnings when Pareto-front candidates accumulate against a search
  boundary.

## Recorded search profile

- Direction set: 400 harmless plus 400 harmful prompts.
- Fast refusal proxy: 100 harmful evaluation prompts per trial.
- Search size: 400 accumulated completed Optuna trials.
- Search response limit: 100 generated tokens.
- Perplexity: fixed Wikitext windows, identical across candidates.

The short search-time perplexity measurement is a ranking surrogate, not a
publication-quality capability measurement. Exported finalists should be
rechecked with a larger fixed set of windows and semantic response evaluation.

