# Default Cost Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make calibrated Cost ranking the validated default for Heretic-MOE adaptive searches and preserve source trial identity during recheck.

**Architecture:** Put the canonical targets and weights in `heretic.config`, validate the complete scorer contract at Settings construction, and keep adaptive TOML profiles explicit. Add a small display-label helper so normal trials and recheck trials share one tested formatting path.

**Tech Stack:** Python 3.12, Pydantic Settings, Optuna, pytest, TOML.

## Global Constraints

- Do not read prompt/response corpora.
- Do not interrupt the active Gemma search.
- Do not silently fall back from Cost ranking.
- Existing journals remain immutable.

---

### Task 1: Default Cost contract

**Files:**
- Modify: `src/heretic/config.py`
- Modify: `research/configs/adaptive_search/*.toml`
- Test: `tests/test_default_cost_contract.py`

- [ ] Write tests asserting the default policy, calibrated maps, and validation failures.
- [ ] Run the focused tests and verify they fail because the defaults are still lexicographic.
- [ ] Add canonical constants, default factories, and fail-fast validation.
- [ ] Set every adaptive profile to `feasible_cost` with the canonical maps.
- [ ] Run the focused tests and verify they pass.

### Task 2: Source trial identity

**Files:**
- Modify: `src/heretic/main.py`
- Test: `tests/test_recheck_trial_labels.py`

- [ ] Write a test asserting `T962 (recheck T2)` for a transferred finalist.
- [ ] Run the focused test and verify the existing local-only label fails.
- [ ] Add one label helper and use it in both leaderboard render paths.
- [ ] Run the focused test and verify it passes.

### Task 3: End-to-end verification

**Files:**
- Verify: `src/heretic/config.py`
- Verify: `src/heretic/main.py`
- Verify: adaptive TOML profiles

- [ ] Run all non-corpus unit tests.
- [ ] Run a dry-run against an adaptive profile and verify the Cost contract.
- [ ] Confirm the active Gemma queue is still advancing.
- [ ] Commit the verified implementation.
