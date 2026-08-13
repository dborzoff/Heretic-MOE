# Heretic-MOE Multilingual Search v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Реализовать один воспроизводимый `hereticMOE search`, который строит мультиязычную карту, выполняет N-GPU поиск, перепроверяет TOP-6 и сохраняет Balanced/Max.

**Architecture:** Один controller обнаруживает N GPU и держит по одному resident worker на устройство. До trial 0 он замораживает dataset/map/SRG/schedule/generation/metric contracts; workers получают задачи из общей очереди и пишут результаты под глобальными trial IDs.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Optuna, Safetensors, SQLite/JSONL, pytest, Windows PowerShell.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-08-11-heretic-moe-multilingual-search-v3-design-ru.md`.
- Dataset root: `F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_5lang_v1`.
- Languages: `en`, `ru`, `zh`, `es`, `fr`; public reports remain text-free.
- Direction map: 10,000 prompt-only rows; Trial pool: 4,000 aligned rows; SRG calibration/final holdout: 660 rows each.
- Ordinary trial: exactly 800 responses at 100-token cap and one autoregressive generation phase.
- Final holdout recheck: the frozen 4,000+660 rows at 100-token cap. PPL preservation uses matching SAFE response-token IDs via conditional NLL; the legacy independent text-corpus PPL is not part of multilingual v3.
- New metric contract never imports raw values from old 136-row or `-0.0088` studies.
- Resume requires exact model, dataset, map, SRG, schedule, generation, metric and constraint hashes.
- The only user-edited launch configuration is YAML. Generated TOML and component manifests remain internal immutable artifacts under the resolved run-root.
- Incompatible contract policy is one of `archive` (default), `new_run`, `replace`, or `fail`; it always applies to the complete run-root, never to one journal.

---

## Approved completion order (2026-08-13)

- [ ] Add a strict public YAML schema and generate the existing internal TOML/controller arguments from it.
- [ ] Connect the existing frozen run-contract builder to production startup and exact resume validation.
- [ ] Implement safe whole-run handling for `archive`, `new_run`, `replace`, and `fail`, including legacy roots without a contract.
- [ ] Remove legacy multilingual finalist flags for independent `64x1024` PPL, `2/136` Keywords, absolute SRG gates and selectable legacy ranking policy.
- [ ] Add missing hard gates for empty/truncated output and SAFE `D -> R` regression.
- [ ] Fix multilingual finalist defaults (`TOP-6`, Balanced removal fraction `0.8`) and stale SRG fixture tests.
- [ ] Generalize fused-MoE detection to every architecture already supported by the editor.
- [ ] Make batch selection verify the selected batch at the real 100-token length; prewarm ordinary, 4,000-row finalist and 660-row holdout shapes separately.
- [ ] Wire the existing trajectory package and offline HTML renderer into the one-command search pipeline.
- [ ] Add one-line aggregate N-GPU progress, worker heartbeat/lease recovery and controller-level duplicate-run locking.
- [ ] Synchronize `config.default.toml`, active documentation, provenance including untracked files, lint and type checks for the production path.
- [ ] Verify with unit tests, N=1/2/4 controller tests, crash/resume tests, a two-GPU 10-12-trial smoke and only then a 120/600 real search.

### Task 1: Symmetric PPL preservation metric

**Files:** `src/heretic/scorers/perplexity.py`, `src/heretic/main.py`, PPL/default-cost tests, design spec.

- [x] Add failing tests for reciprocal symmetry, clean zero, invalid values and signed console output.
- [x] Implement `symmetric_perplexity_change(perplexity, baseline_perplexity)` using signed log-PPL delta.
- [x] Keep `Perplexity drift` as the stable non-negative journal value; add `signed_relative_change` and `signed_log_delta` diagnostics.
- [x] Show magnitude plus signed direction in ordinary output and leaderboard.
- [x] Run focused tests and observe `19 passed`.
- [ ] Re-run PPL tests as part of the full-suite gate before commit.

### Task 2: Frozen multilingual dataset and run contract

**Files:** create `src/heretic/multilingual_contract.py`; modify `src/heretic/config.py`; test `tests/test_multilingual_contract.py`.

**Produces:** `MultilingualDatasetContract`, `FrozenRunContract`, manifest hashes and row indexes consumed by every later stage.

- [ ] Write tests that reject wrong counts, missing languages, cross-language ID/order drift, overlap between direction/trial/SRG calibration/final holdout, prompt leakage into public manifests and SHA mismatch.
- [ ] Load the operative `1000/400` files plus SRG calibration/final holdout files without reading or logging prompt text.
- [ ] Freeze counts, file hashes, canonical coverage, languages, generation caps, schedule version, metric version and constraint contract before GPU work.
- [ ] Refuse resume when any frozen field differs; never silently fall back to old datasets or scorer targets.
- [ ] Verify exact totals: direction 10,000; trial 4,000; SRG calibration 660; final holdout 660.

### Task 3: Direction map and search-direction package

**Files:** extend focused `language_map_*` modules; test map analysis/cache/controller/report modules.

**Produces:** `DirectionMapProfile` with `consensus_refusal_direction`, per-layer/category directions, language subspace, reliability, layer bounds and frozen projection basis.

- [ ] Add synthetic tests proving language nuisance is removed while SAFE/UNSAFE separation remains and direction signs are stable.
- [ ] Capture 10,000 prompt-only residual rows with `forward()` only, sharded through the N-GPU queue.
- [ ] Compute canonical centers, language deviations, category branches, macro-balanced reliability and continuous recommended layer bounds.
- [ ] Freeze all tensors and SHA-256 before trial 0; emit text-free JSON plus interactive HTML.
- [ ] Feed existing global/per-layer intervention modes from this package without changing it during the study.

### Task 4: Five-trial schedule and clean Trial reference

**Files:** `src/heretic/trial_language_schedule.py`; create focused reference-archive module; tests for schedule and archive.

**Produces:** immutable `schedule.jsonl` and `clean_trial_reference` keyed by canonical ID/language.

- [ ] Add a failing test proving blocks 0 and 1 use different ID-level assignments, not only a reordered phase list.
- [ ] Guarantee each trial contains 400 SAFE + 400 UNSAFE and exactly 80 rows per language per direction.
- [ ] Guarantee every canonical ID uses all five languages exactly once in each five-trial block.
- [ ] Generate all 4,000 clean reference responses once at 100 tokens; store response IDs, token targets, NLL, residual projections and clean metrics privately.
- [ ] Make resume reuse verified schedule/reference artifacts without regenerating or renumbering them.

### Task 5: SRG calibration and relative scoring

**Files:** finish `src/heretic/srg_calibration.py`, `src/heretic/srg_benchmark.py`, scorer integration and tests.

**Produces:** versioned `SRGCalibrationProfile` and exact target-model clean calibration baseline.

- [ ] Preserve current robust median/MAD/sign-consensus behavior with tests for hash mismatch, non-finite rows and scale floors.
- [ ] Run the SRG calibration set `132 x 5` only once; keep it outside ordinary trials.
- [ ] Compute baseline-relative continuous gain, R-to-D, D-to-R and R-side gain with clean=trial exactly zero.
- [ ] Store global cross-model scales separately from the target model's own clean margins.
- [ ] Remove old raw `SRG=0`, `R-side=0` and static `-0.0088` assumptions from the new-search path.

### Task 6: One-generation multilingual trial evaluator

**Files:** create focused trial-evaluation module; integrate through `Evaluator`; add unit/integration tests.

**Produces:** `TrialMeasurement` containing Removal, PreservationLoss, Cost-up and diagnostics.

- [ ] Assert one ordinary trial invokes exactly one autoregressive generation phase for its scheduled 800 rows.
- [ ] Capture UNSAFE/SRG and SAFE geometry residuals during prefill without a second response generation.
- [ ] Compute teacher-forced same-ID SAFE NLL against frozen clean targets.
- [ ] Implement `Removal = .50*SRG_gain + .25*R_gain + .25*UnsafeGeometryGain`.
- [ ] Implement `PreservationLoss = .40*SafePPLDrift + .35*SafeGeometryDamage + .15*LanguageInstability + .10*CategoryInstability`.
- [ ] Implement `Cost_up = sigmoid(4*Removal)/(1+PreservationLoss)` and keep Keywords diagnostic-only.
- [ ] Freeze hard gates before trial 0 and fail non-finite/incomplete measurements rather than inventing fallback values.

### Task 7: One-command adaptive N-GPU search

**Files:** extract reusable controller from `research/scripts/run_adaptive_search.py`; extend `src/heretic/cli.py`, queue/search modules and tests.

**Public interface:** `hereticMOE search --config <toml> --model <path> --run-root <path> --devices auto --target-trials 600`.

- [ ] Add CLI/config tests for `auto`, explicit device lists and 1/2/N devices.
- [ ] Start one resident worker per visible/selected GPU and one shared durable queue with global trial numbers.
- [ ] Let faster devices claim more trials; never pre-partition a fixed budget per GPU.
- [ ] Run trials 0-119 as 60 Random + 60 scrambled Sobol with shared numbering, then multivariate constrained TPE.
- [ ] Preserve model residency between trials, requeue only an interrupted worker's claim and keep other GPUs running.
- [ ] Continue 600 to 1000 only under the same frozen contract; reject attempts to restart from an earlier target inside the same journal.
- [ ] Keep the research script as a thin compatibility wrapper over the same controller.

### Task 8: TOP-6 recheck and winners

**Files:** update `research/scripts/finalist_recheck.py`, selection code and tests.

**Produces:** frozen finalists, full recheck measurements and `winners.json`.

- [ ] Select two maximum-Removal, two maximum-Cost-up and two diverse low-Preservation Pareto candidates with exact parameter de-duplication.
- [ ] Freeze TOP-6 before opening any final-holdout candidate outputs.
- [ ] Recheck every finalist on all 4,000 Trial rows and final holdout 660 at 100 tokens, with same-ID SAFE conditional-NLL PPL drift.
- [ ] Choose Balanced as minimum PreservationLoss after removal/gates and Max as maximum Removal after preservation gates.
- [ ] If both roles select identical parameters, write two roles pointing to one physical artifact.

### Task 9: Artifacts, privacy and resume audit

**Files:** controller manifest/archive/report code and tests.

- [ ] Materialize the exact artifact tree defined in the design spec with SHA-256 for every immutable component.
- [ ] Keep prompts/responses only in private archives; console, manifests and public reports expose IDs, counts and metrics only.
- [ ] Store per-trial schedule, parameters, worker/GPU, timings, measurement components and response-archive coverage.
- [ ] Add crash/restart tests for queue recovery, partial map shards, partial response archives and incompatible resume.

### Task 10: Verification and real 8-9B rollout

**Order:** `Qwen__Qwen3-8B`, `Qwen__Qwen3.5-9B`, then `Qwen__Qwen3-VL-8B-Instruct` as a processor/VL compatibility gate.

- [ ] Run focused tests after every task, then full `pytest -q` and `python -m compileall src research/scripts`.
- [ ] Run mocked N=1, N=2 and N=4 controller tests; verify global IDs, dynamic load balance and worker recovery.
- [ ] Run a real two-RTX-4090 smoke with frozen miniature fixtures and no export.
- [ ] Run full direction/reference preparation plus a short 10-trial search on Qwen3-8B in one visible UTF-8 PowerShell window.
- [ ] Repeat the short gate on Qwen3.5-9B; compare throughput, VRAM, metric stability and artifact coverage.
- [ ] Run Qwen3-VL-8B only after the two plain decoder models pass, verifying processor/chat-template compatibility.
- [ ] Start a full 120-exploration/600-total run only on the first model whose short gate passes every acceptance check.

## Acceptance Gate

- [ ] No production new-search path contains the old 136-row totals, `-0.0088`, static English-only direction or signed-PPL optimization bonus.
- [ ] Clean candidate produces zero SRG/R/PPL/geometry deltas within numerical tolerance and Cost-up `0.5`.
- [ ] Public progress shows `GPU N | Tglobal | Cost↑ | Removal | Preservation | PPL drift (signed)` without prompt/answer text.
- [ ] All tests pass freshly; model loading, trial counters and output artifacts advance during the real smoke.
