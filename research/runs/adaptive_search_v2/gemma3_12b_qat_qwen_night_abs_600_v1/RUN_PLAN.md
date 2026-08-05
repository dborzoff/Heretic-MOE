# Gemma 3 12B QAT: Qwen-night search with absolute PPL drift

- Code baseline: `33f8727` (`feasible_cost` requires explicit targets and weights).
- Only algorithm change: perplexity uses absolute relative drift.
- Search objectives: sparse refusal geometry and absolute PPL drift measured on
  24 fixed windows of 512 tokens.
- Constraint: absolute PPL drift at most `0.005`.
- Finalist diagnostics: keyword rate.
- Prefix: reuse the first 120 Random/Sobol trials after numeric migration.
- Adaptive stage: two-worker multivariate TPE, trials 120 through 599.
- LTX conditioning is not loaded and does not influence sampling or ranking.
- Trial responses are retained in SQLite; console and reports remain text-free.
