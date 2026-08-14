# Throughput-Aware Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the fastest safe generation batch, remove repeated successful NLL cache purges, expose per-stage timings, and prove the improvement against the frozen 45.6-second Qwen3-VL-4B baseline.

**Architecture:** Keep the frozen multilingual metrics and 800-row/100-token trial contract unchanged. Extend the existing memory autotuner with a short real-throughput sweep over safe candidates, then retain CUDA allocations through the full teacher-forced NLL pass and release them once at its boundary. Preserve durable response archives; measure disk time before changing persistence semantics.

**Tech Stack:** Python 3.12, PyTorch, Transformers, pytest, Ruff, Optuna journal/SQLite queue.

## Global Constraints

- Do not inspect corpus prompt or response text.
- Keep 400 SAFE + 400 UNSAFE and 100 generated tokens per search trial.
- Keep at least 10% or 2 GiB VRAM reserve, whichever is larger.
- Do not change objective, constraints, schedule, trial IDs, or journal contents.
- Resume the existing 74/600 queue only after one isolated benchmark proves correctness.

---

### Task 1: Throughput-aware generation batch

**Files:**
- Modify: `src/heretic/model.py`
- Modify: `src/heretic/generation_batch_cache.py`
- Test: `tests/test_prompt_token_cache.py`
- Test: `tests/test_generation_batch_cache.py`

**Interfaces:**
- Consumes: `Model.validate_generation_batch_size(...)` and the existing memory-safe candidate.
- Produces: validation records containing `elapsed_seconds`, `generated_tokens`, and `tokens_per_second`; `Model.autotune_generation_batch_size(...)` selects the highest-throughput safe candidate.

- [ ] Add a failing test with safe batches 64/96/128 whose measured throughput peaks at 96; assert that 96 is selected instead of memory-maximum 128.
- [ ] Run the focused test and confirm it fails because selection currently uses only maximum safe VRAM.
- [ ] Time exact 100-token validation after CUDA synchronization and sweep deterministic 50%, 75%, and 100% safe candidates, aligned to configured granularity.
- [ ] Persist throughput fields in the generation-batch cache record and key it by the existing model/GPU/mode contract.
- [ ] Run focused tests and Ruff; commit the independently passing change.

### Task 2: Conditional NLL allocation reuse

**Files:**
- Modify: `src/heretic/model.py`
- Test: `tests/test_teacher_forced_nll.py`

**Interfaces:**
- Consumes: cached prompt IDs and frozen clean target token IDs.
- Produces: identical per-row conditional NLL values while releasing transient CUDA cache once after all successful batches and immediately after OOM/backoff.

- [ ] Replace the old test expecting one cache purge per successful batch with a failing test expecting exactly one purge after the complete successful NLL call.
- [ ] Add a failing test asserting combined prompt+target lengths are processed descending.
- [ ] Run both tests and confirm expected failures against the current ascending/per-batch implementation.
- [ ] Process longest sequences first so smaller later tensors reuse allocator blocks; delete batch tensors normally and purge once in `finally` after successful completion.
- [ ] Preserve immediate purge for OOM/headroom backoff and exact row-order restoration.
- [ ] Run the NLL suite and Ruff; commit the independently passing change.

### Task 3: Stage timings and visible progress

**Files:**
- Modify: `src/heretic/multilingual_trial_evaluator.py`
- Modify: `src/heretic/main.py`
- Test: `tests/test_multilingual_trial_evaluator.py`
- Test: `tests/test_main_prompt_contract.py`

**Interfaces:**
- Produces: text-free timing diagnostics for `generation`, `srg_metrics`, `conditional_nll`, `geometry_metrics`, and `archive_write`, plus compact aggregate progress.

- [ ] Add failing tests asserting finite nonnegative stage timings without serializing prompts/responses into public diagnostics.
- [ ] Time existing stage boundaries with `time.perf_counter()`; preserve metric values and private archive SHA.
- [ ] Print one compact supervisor line with global completed/target, worker balance, elapsed, rate, and ETA.
- [ ] Run focused tests and Ruff; commit the independently passing change.

### Task 4: Verification and real benchmark

**Files:**
- Modify only if verification exposes a tested defect.

**Interfaces:**
- Consumes: the stopped corrected run and its immutable baseline evidence.
- Produces: one comparable full 800-row/100-token trial timing and a resume/no-resume decision.

- [ ] Run focused suites for batch selection, prompt cache, NLL, multilingual evaluator, and supervisor progress.
- [ ] Run the complete pytest suite and Ruff on changed production/tests.
- [ ] Start one isolated two-GPU benchmark without mutating the source 74/600 journal; record selected batch, tokens/s, 10% headroom, stage timings, and total trial duration.
- [ ] Compare against baseline median 45.6 seconds per GPU trial; resume only if output contracts pass and no stage regresses materially.
- [ ] Resume the existing queue from 74/600; allow the stale claimed row to be recovered by the durable queue and verify two active workers before unattended execution.
