# Recheck-Only Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a resumable TOP-6 high-fidelity recheck mode that never exports model weights.

**Architecture:** Keep the existing search and finalist-recheck pipeline. Add one explicit post-search mode at the controller boundary, pass an export decision into finalization, and stop immediately after validated winners when recheck-only is selected.

**Tech Stack:** Python, argparse, unittest, existing Heretic-MOE controller and finalist recheck tools.

---

### Task 1: Lock the CLI contract with tests

**Files:**
- Modify: `tests/test_adaptive_search_controller.py`
- Test: `tests/test_adaptive_search_controller.py`

1. Add tests for default/finalize, recheck-only, and search-only parsing.
2. Assert `--recheck-only` is mutually exclusive with export and search-only flags.
3. Assert the default finalist count is six.
4. Run the focused test module and confirm the new tests fail before production changes.

### Task 2: Implement the post-search mode

**Files:**
- Modify: `research/scripts/run_adaptive_search.py`

1. Replace coupled boolean parsing with an explicit three-way post-search mode while preserving the existing flags.
2. Record the selected mode in the run manifest.
3. Let the existing finalization function receive `export_models`.
4. After the validated winner report is loaded, return before all export-directory and model-assembly operations when `export_models` is false.
5. Write `recheck_complete` instead of `release_complete` for that path.

### Task 3: Verify behavior and regressions

**Files:**
- Test: `tests/test_adaptive_search_controller.py`
- Test: `tests/`

1. Run the focused controller tests.
2. Run the full unit-test suite.
3. Run CLI help and a dry-run check without touching model payloads.
4. Inspect the diff for accidental prompt/response access and unrelated changes.

### Task 4: Commit and execute the current recheck

**Files:**
- Commit the specification, plan, tests, and controller change.
- Reuse: `F:/AI/hf_originals/heretic_out/research/searches/Qwen__Qwen3-0.6B-Base/HereticMOE-v1.5-search-only`

1. Let the active search reach exactly 600 trials without restart.
2. Resume the same run root at target 600 in recheck-only mode with TOP 6 and PPL 64x1024.
3. Keep the recheck in a visible foreground PowerShell window with advancing counters.
4. Validate finalist coverage, winner report, hashes, and absence of model exports.
5. Report only numerical/path/hash metadata; do not inspect prompt or response texts.
