# Dynamic multi-GPU search pipeline

`hereticMOE` is the public one-command controller for Random/Sobol
exploration, multivariate-TPE continuation, finalist recheck, and two-model
export. It detects available NVIDIA devices and starts one persistent worker per
selected GPU.

Heretic-MOE 1.5 completes the full workflow by default: TOP-5 candidates are
remeasured with 64 x 1,024-token PPL windows, distinct `Balanced` and `Max`
winners are selected, and both models are exported with SHA-256 manifests. Use
`--search-only` when only the reusable journal and response archive are needed.

For a 600-trial run with `--exploration-trials 120` it performs:

1. The supervisor creates one durable queue and one shared Optuna journal.
2. The first 120 numbered tasks alternate Random and scrambled Sobol.
3. A barrier prevents TPE from starting before all 120 exploration tasks finish.
4. The same model-resident workers then claim multivariate-TPE tasks. A faster
   GPU claims more work instead of waiting at a fixed per-device budget.
5. The queue completes exactly 600 work permits. A recovered process crash can
   leave an additional failed audit row in the journal, but never drops a task.
6. Both GPUs remeasure a constraint-aware, Pareto-diverse TOP-5 at higher
   fidelity, so preservation and maximum-removal regions are both represented.
7. The controller exports `Balanced` and `Max` without interactive prompts.

The device list is deduplicated. The same scheduler supports one, two, or many
GPUs. One GPU means one resident model; six GPUs mean six workers pulling from
the same queue. A run-root lock rejects a second supervisor before it can launch
duplicate workers.

Sobol covers continuous parameters. The always-present categorical
`direction_scope` axis is stratified deterministically (alternating choices),
so the 120-trial prefix evaluates it exactly 60/60 without Optuna's independent
RandomSampler fallback. Any unknown future categorical fallback still emits a
warning instead of being hidden.

There is no branch merge in the dynamic path. Every generated answer is written
during evaluation to the run root's single `trial-responses.sqlite3` and linked
to its immutable global Optuna trial number.

## Run layout

```text
<run-root>/
  adaptive_run_manifest.json
  trial-work-queue-600.sqlite3
  trial-responses.sqlite3
  shared_tpe/
    config.toml
    checkpoints/*.jsonl
```

The run manifest captures launch-time SHA-256 values for the controller and
`hereticMOE` executable plus the Git revision and dirty state when available.
These values are frozen at process start and do not change if the checkout is
updated while a long run is still active.

Generated stage configs are immutable except that the shared `n_trials` target
may be raised explicitly. Re-running the controller resumes existing journals;
it never overwrites a different stage configuration.

## Portable perplexity corpus

The Perplexity scorer defaults to
`builtin://perplexity-reference-v1`. The 241,748-byte canonical LF corpus is
packaged under `src/heretic/data/` and verified by SHA-256 before use. No machine-specific
`F:/AI/llamacpp/ppl_test.txt` path or Hugging Face download is required.

Model and prompt-dataset paths remain explicit inputs because they differ
between local and rented servers. For a portable run, pass `--data-root` with
the four-file adaptive-search bundle; the controller writes those resolved
paths into the effective run-local TOML and records the root in the run
manifest.

## Example

```powershell
F:\AI\heretic_env\Scripts\hereticMOE.exe `
  --base-config research\configs\adaptive_search\gemma2_sparse_geometry.toml `
  --model F:\AI\hf_originals\my_model `
  --data-root F:\AI\HereticMoe\runtime_data\adaptive_search_v2 `
  --run-root research\runs\adaptive_search_v2\gemma2_sparse_geometry_dual_600_v1 `
  --exploration-trials 120 `
  --n-trials 600
```

The default is one visible console. Worker output is prefixed with the physical
GPU index while all workers write the shared run artifacts. Use `--devices
0,1,2` for an explicit selection; the default `auto` mode applies the free-memory
gate and `--max-workers` can cap the pool.

To extend a completed shared study without repeating exploration, add
`--continue-shared-only` and raise `--n-trials`.
