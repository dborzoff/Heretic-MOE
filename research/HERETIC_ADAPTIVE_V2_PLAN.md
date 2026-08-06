# Heretic Adaptive v2 implementation plan

## Objective

Improve refusal removal without trading away general model quality. The search
must remain resumable, auditable, and bounded by user-selected budgets. Prompt
and response text stays outside the Optuna journal and public reports.

## Baseline retained

The completed 600-trial Ministral study remains immutable. It is not appended
after objective or search-space changes. Its Pareto points may seed a new study,
where they are re-evaluated under the new contract.

## Search contract

### 1. Feasibility before preference

Scorers may define upper or lower feasibility bounds. A common configuration is
an upper bound on relative perplexity increase. The sampler learns feasibility,
and the selection menu ranks feasible trials before infeasible trials.

Within the feasible set, selection is lexicographic:

1. the configured primary behavioral objective;
2. remaining optimization objectives;
3. normalized constraint slack;
4. immutable trial number.

This makes a low-damage model with a small residual refusal rate preferable to
an unconstrained zero-refusal model that exceeds the accepted quality cost.

### 2. Branched exploration

The published workflow becomes a first-class orchestrator:

1. Random startup branch;
2. multivariate TPE continuation of that branch;
3. independent scrambled-Sobol startup branch;
4. multivariate TPE continuation of that branch;
5. deterministic merge into a shared study;
6. shared multivariate TPE continuation to the requested total budget.

Each stage has an exact trial count, a separate journal, a recorded seed, and a
resume-safe manifest. A merged study never silently mixes incompatible scorer,
constraint, or search-space contracts.

### 3. Multi-fidelity promotion

Multi-objective Optuna trials cannot use the standard `trial.report` pruning
interface. Adaptive v2 therefore uses explicit studies rather than pretending
that ordinary pruning is available:

- discovery study: small prompt/PPL budgets;
- promotion: a spread of feasible Pareto points is enqueued into a new study;
- confirmation study: larger prompt/PPL budgets and the same parameter space;
- finalist validation: export/reload, long generation, semantic labels, and a
  larger exact perplexity pass.

Old scores are never copied across fidelity levels. Parameters are copied and
measured again.

### 4. Conditional component search

Optional component gates turn whole edit components on or off. When a component
is disabled, its curve parameters are absent rather than sampled and ignored.
Grouped multivariate TPE handles the resulting conditional subspaces. The
legacy fixed-space mode remains available for compatibility and A/B tests.

### 5. Text-free trial telemetry

Every completed trial records:

- scorer and baseline scalar values;
- constraint values and feasibility;
- component parameters and enabled state;
- per-component/per-layer realized edit norms and edited parameter counts;
- runtime and peak allocated/reserved CUDA memory;
- model/config/dataset hashes in the run manifest;
- optional scorer-specific dry diagnostics, never prompt or response text.

The public journal contract must remain safe to inspect without revealing the
corpus.

## Direction estimation

The default difference-of-means direction remains available. Adaptive v2 adds
diagnostics before changing the estimator:

- balanced group weights when group metadata is explicitly supplied;
- deterministic bootstrap directions;
- adjacent-layer cosine stability and bootstrap agreement;
- a prior map for search, never a hard causal claim.

A rank-k direction is allowed only after more than one stable independent
direction is observed. Covariance/oblique projectors, routers, gates, MTP, and
intra-layer point trees remain research-only until they show a cross-model
causal benefit.

## Dense and fused normalization

The fused routed-expert path gains the same row-normalization choices as dense
output projections. Its FULL path performs the exact operation directly on each
expert tensor in bounded chunks:

1. save original row norms;
2. normalize rows;
3. apply the directional projection;
4. renormalize edited rows;
5. restore original row norms.

Search reports use realized relative edit norm, not `max_weight`, to compare
edit strength across dense and fused architectures.

## Validation gates

1. Unit tests for configuration, constraints, conditional sampling, merging,
   promotion, telemetry, exact restoration, and fused normalization.
2. Deterministic toy studies for Random/Sobol branches and shared continuation.
3. Static tensor mechanics tests for dense/fused equivalence where the
   operations are mathematically equivalent.
4. Short model smoke test on one dense checkpoint.
5. Cross-family confirmation on Ministral, Qwen, and Gemma.
6. Only then create a new seeded study and extend the validated search budget
   toward 1,000 trials.

## Speed benchmark contract

Model throughput is measured with a GPU crossover:

- every candidate runs on both physical GPUs;
- warmup is excluded;
- prefill and decode timing are reported separately;
- generated tokens per second is primary, rows per second is secondary;
- peak VRAM and background VRAM are recorded;
- response-generation and judge-classification benchmarks are separate.

