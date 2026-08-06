# Adaptive two-GPU search pipeline

`research/scripts/run_adaptive_search.py` is the single controller for the
Random/Sobol exploration, multivariate-TPE continuation, finalist recheck, and
two-model export.

Heretic-MOE 1.5 completes the full workflow by default: TOP-5 candidates are
remeasured with 64 x 1,024-token PPL windows, distinct `Balanced` and `Max`
winners are selected, and both models are exported with SHA-256 manifests. Use
`--search-only` when only the reusable journal and response archive are needed.

For a 600-trial run with `--exploration-trials 120` it performs:

1. GPU 0 evaluates 60 Random trials in `random_branch/`.
2. GPU 1 evaluates 60 scrambled-Sobol trials in `sobol_branch/`.
3. The controller validates both study contracts and creates one shared journal.
   Its first 120 trials are round-robin: Random gets even numbers and Sobol gets
   odd numbers.
4. Two TPE workers share the merged journal and evaluate 240 trials each. Optuna
   assigns these numbers atomically because completion order is asynchronous.
5. The final journal contains exactly 600 trials.
6. Both GPUs remeasure a constraint-aware, Pareto-diverse TOP-5 at higher
   fidelity, so preservation and maximum-removal regions are both represented.
7. The controller exports `Balanced` and `Max` without interactive prompts.

Sobol covers continuous parameters. The always-present categorical
`direction_scope` axis is stratified deterministically (alternating choices),
so a 60-trial branch evaluates it exactly 30/30 without Optuna's independent
RandomSampler fallback. Any unknown future categorical fallback still emits a
warning instead of being hidden.

The merge copies Optuna metadata only. It does not load a model or regenerate
answers. Every generated answer is written during evaluation to the run root's
single `trial-responses.sqlite3`. Branch-local trial numbers are mapped to the
same even/odd global numbering before the journal merge.

## Run layout

```text
<run-root>/
  adaptive_run_manifest.json
  trial-responses.sqlite3
  random_branch/
    config.toml
    checkpoints/*.jsonl
  sobol_branch/
    config.toml
    checkpoints/*.jsonl
  shared_tpe/
    config.toml
    checkpoints/*.jsonl
    checkpoints/*.jsonl.merge.json
```

Generated stage configs are immutable except that the shared `n_trials` target
may be raised explicitly. Re-running the controller resumes existing journals;
it never overwrites a different stage configuration.

## Portable perplexity corpus

The Perplexity scorer defaults to
`builtin://perplexity-reference-v1`. The 241,748-byte canonical LF corpus is
packaged under `src/heretic/data/` and verified by SHA-256 before use. No machine-specific
`F:/AI/llamacpp/ppl_test.txt` path or Hugging Face download is required.

Model and prompt-dataset paths remain explicit inputs because they differ
between local and rented servers. Copy the repository and data bundle, update
those three paths in the base config, then keep the pipeline code unchanged.

## Example

```powershell
F:\AI\heretic_env\Scripts\python.exe research\scripts\run_adaptive_search.py `
  --base-config research\configs\adaptive_search\gemma2_sparse_geometry.toml `
  --run-root research\runs\adaptive_search_v2\gemma2_sparse_geometry_dual_600_v1 `
  --heretic F:\AI\heretic_env\Scripts\heretic.exe `
  --exploration-trials 120 `
  --target-trials 600 `
  --random-device 0 `
  --sobol-device 1
```

The default is one visible controller console. Both GPU processes inherit its
output and also write their shared run artifacts. For diagnostics only,
`--visible-worker-windows` opens a separately titled console and stage-local log
for every active GPU process.

To extend a completed shared study without repeating exploration or merge, add
`--continue-shared-only` and raise `--target-trials`.
