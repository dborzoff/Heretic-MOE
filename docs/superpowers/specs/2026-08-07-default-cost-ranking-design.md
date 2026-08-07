# Default Cost Ranking Design

## Goal

Make calibrated Cost ranking the mandatory default for Heretic-MOE adaptive
searches, with no silent downgrade to lexicographic ranking.

## Contract

- Default policy: `feasible_cost`.
- Required lower-is-better targets:
  - `Sparse refusal geometry = -0.0088`
  - `Keywords = 2 / 136`
  - `Perplexity drift = 0.0`
- Required hinge weights:
  - `Sparse refusal geometry = 344.0`
  - `Keywords = 697.68`
  - `Perplexity drift = 200.0`
- A `feasible_cost` configuration must fail validation when a required target,
  weight, or scorer is absent.
- Adaptive model profiles must state `feasible_cost` explicitly even though it
  is the library default.
- Recheck leaderboards must identify both the source search trial and the local
  recheck trial.

## Compatibility

Cost changes finalist ranking and display only. Existing Optuna measurements
remain valid because the optimization objectives stay SRG and PPL. The active
Gemma journal therefore continues unchanged and is re-ranked at finalization.

## Verification

- Unit tests cover default values and invalid Cost configurations.
- Unit tests cover source-trial leaderboard labels.
- All existing non-corpus tests remain green.
- A Heretic-MOE dry-run using an adaptive profile reports `feasible_cost` and
  the configured targets and weights before GPU work begins.
