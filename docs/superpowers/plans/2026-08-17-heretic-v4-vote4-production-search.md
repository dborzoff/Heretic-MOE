# Heretic v4 Vote4 Production Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark all 12,800 v4 rows with four permuted short-label votes, run a 600-trial Qwen3-8B Heretic search, and compare original, Balanced, and Max with the identical benchmark.

**Architecture:** A derived schema-v1 benchmark view concatenates the frozen map, trial, and final pools without changing their records. A reusable `vote4` command shards one model across all requested GPUs and writes only text-free result rows and aggregate reports. The existing v4 search pipeline then produces finalists; the same vote contract evaluates exported models and a comparison module builds transition matrices.

**Tech Stack:** Python 3.12, PyTorch/Transformers through the existing Heretic `Model`, JSONL manifests, pytest, Ruff, Optuna dynamic GPU queue.

## Global Constraints

- Model: `F:/AI/hf_originals/LLM/Qwen__Qwen3-8B`.
- Languages: exactly `en`, `ru`, `zh`, `ja`.
- Source corpus remains byte-for-byte frozen and must never be rewritten.
- Public outputs and console logs must never contain prompt text or raw model output.
- Vote variants: `code_permuted`, `code_shift_1`, `code_shift_2`, `code_shift_3`.
- `max_new_tokens=2`, thinking disabled, English classifier system prompt.
- Search: 600 total trials, 120 exploration trials, two GPUs, TOP-6 recheck, Balanced/Max export.
- Existing unrelated dirty classification and translation files must be preserved.

---

### Task 1: Materialize the frozen 12,800-row benchmark view

**Files:**
- Create: `src/heretic/four_language_vote_dataset.py`
- Create: `tests/test_four_language_vote_dataset.py`

**Interfaces:**
- Consumes: v4 `manifest.json` and its 24 map/trial/final JSONL files.
- Produces: `materialize_vote_dataset(source_root, output_root) -> dict[str, object]` and a schema-v1 manifest accepted by `load_classification_rows`.

- [ ] **Step 1: Write a failing test for deterministic concatenation and hashes**

```python
manifest = materialize_vote_dataset(source, output)
assert manifest["directions"] == {"safe": 1600, "unsafe": 1600}
assert manifest["languages"] == ["en", "ru", "zh", "ja"]
assert manifest["rows"] == 12800
assert len(manifest["files"]) == 8
```

- [ ] **Step 2: Run the focused test and confirm it fails because the module is absent**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_four_language_vote_dataset.py`

- [ ] **Step 3: Implement verified map → trial → final concatenation**

Validate the source schema/status, every source file SHA-256, row count,
language/direction fields, per-pool aligned ID order, and uniqueness across all
three pools. Write eight files named `vote_{language}_{direction}_1600.jsonl`
atomically plus one schema-v1 manifest.

- [ ] **Step 4: Verify the focused tests and the real corpus mechanically**

Run the test above, materialize under
`F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4/vote4`,
then load it with `load_classification_rows` without printing row text.

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/four_language_vote_dataset.py tests/test_four_language_vote_dataset.py
git commit -m "feat: build frozen vote4 dataset view"
```

### Task 2: Add a four-pass, N-GPU vote command

**Files:**
- Modify: `src/heretic/self_classification_cli.py`
- Modify: `src/heretic/self_classification_worker.py`
- Modify: `tests/test_self_classification_cli.py`
- Modify: `tests/test_self_classification_worker.py`

**Interfaces:**
- Consumes: schema-v1 benchmark manifest, one model path, ordered GPU IDs.
- Produces: `hereticMOE self-classify vote4` with exact coverage and resumable per-worker parts.

- [ ] **Step 1: Write failing CLI and worker contract tests**

```python
result = main(["vote4", "--dataset-manifest", str(manifest),
               "--model", str(model), "--output-dir", str(output),
               "--devices", "0,1", "--dry-run"])
assert result["rows"] == 12800
assert result["variants"] == 4
assert result["tasks"] == 51200
assert result["workers"] == 2
assert result["max_new_tokens"] == 2
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_self_classification_cli.py tests/test_self_classification_worker.py`

- [ ] **Step 3: Implement `vote4`**

Add one parser branch and one controller that creates one deterministic shard per
device, uses the four code variants, `system_mode="english"`, and
`max_new_tokens=2`. Merge parts atomically and require exactly 51,200 unique
keys. Keep raw output private to the in-memory parser and prohibit it in files.

- [ ] **Step 4: Add visible text-free progress**

Emit worker/device, completed/total, active batch size, valid/invalid counts,
elapsed time, and ETA. Do not emit prompts, decoded output, or row text.

- [ ] **Step 5: Verify focused tests and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_self_classification_cli.py tests/test_self_classification_worker.py
git add src/heretic/self_classification_cli.py src/heretic/self_classification_worker.py tests/test_self_classification_cli.py tests/test_self_classification_worker.py
git commit -m "feat: add four-pass gpu vote benchmark"
```

### Task 3: Produce vote and transition reports

**Files:**
- Modify: `src/heretic/self_classification_report.py`
- Modify: `tests/test_self_classification_report.py`

**Interfaces:**
- Consumes: text-free vote rows from one or more model IDs.
- Produces: per-model vote reports and original/Balanced/Max transition reports.

- [ ] **Step 1: Write failing aggregation tests**

Test strict 4/4, majority 3/4, invalids, language x direction,
category x language, and model-to-model transitions. Assert that SAFE and UNSAFE
are never combined into one success rate.

- [ ] **Step 2: Run the focused report tests and confirm RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_self_classification_report.py`

- [ ] **Step 3: Implement text-free reports**

Write JSON and HTML summaries containing counts and percentages only. For
UNSAFE, report HARD/SOFT → PARTIAL/DIRECT transitions. For SAFE, report
DIRECT/PARTIAL → SOFT/HARD regressions. Include 4/4 and 3/4 tables separately.

- [ ] **Step 4: Verify and commit**

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_self_classification_report.py
git add src/heretic/self_classification_report.py tests/test_self_classification_report.py
git commit -m "feat: compare vote4 model transitions"
```

### Task 4: Add the production Qwen3-8B search profile

**Files:**
- Create: `config.heretic_moe_4lang_v4_qwen3_8b.yaml`
- Modify: `tests/test_public_config_references.py`

**Interfaces:**
- Consumes: the existing v4 dataset and production controller.
- Produces: one frozen 600/120 search configuration with automatic TOP-6 and export.

- [ ] **Step 1: Write a failing config-contract test**

Assert model path, target/exploration counts, `post_search: export`, four
languages, v4 counts, 100-token generation, two-role export, and dynamic devices.

- [ ] **Step 2: Run the config test and confirm RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_public_config_references.py`

- [ ] **Step 3: Add the YAML profile and verify dry-run**

Run:

```powershell
F:/AI/heretic_env/Scripts/hereticMOE.exe --config config.heretic_moe_4lang_v4_qwen3_8b.yaml --dry-run
```

- [ ] **Step 4: Commit**

```powershell
git add config.heretic_moe_4lang_v4_qwen3_8b.yaml tests/test_public_config_references.py
git commit -m "feat: add qwen3 8b production search profile"
```

### Task 5: Verify the complete code path

**Files:** No production edits unless a failing test exposes a defect.

- [ ] **Step 1: Run the full CPU suite**

Run: `.venv/Scripts/python.exe -m pytest -q`

- [ ] **Step 2: Run Ruff and diff checks**

Run Ruff on every changed Python file, then `git diff --check`.

- [ ] **Step 3: Run a two-GPU vote4 smoke**

Use a deterministic small subset only to validate two resident workers,
coverage, merge, resume, batch backoff, and text-free output.

- [ ] **Step 4: Re-run focused tests after any GPU fix and commit it separately**

### Task 6: Generate the clean Qwen3-8B baseline

**Files:** Runtime artifacts only under `heretic_out/research/results`.

- [ ] **Step 1: Launch vote4 on all 12,800 rows in a visible window**

Use GPUs 0 and 1 and preserve resumable worker parts.

- [ ] **Step 2: Monitor numeric progress and GPU telemetry**

Report coverage, valid/invalid counts, batch size, utilization, VRAM, and ETA;
do not inspect row text or raw outputs.

- [ ] **Step 3: Verify 51,200/51,200 and readiness gates**

Require overall valid rate ≥99% and every language ≥95%. If not, repair only
the prompt/parser contract, resume missing/invalid keys, and repeat validation.

- [ ] **Step 4: Freeze baseline hashes and reports**

### Task 7: Run the production Heretic search

**Files:** Runtime artifacts only under a new versioned search root.

- [ ] **Step 1: Launch the 600/120 profile in one visible controller window**

- [ ] **Step 2: Monitor map, clean reference, Random/Sobol, constrained TPE, TOP-6, recheck, and export**

- [ ] **Step 3: Verify queue/journal integrity and winner provenance**

Require 600 unique COMPLETE trials, no failed claims, six rechecked finalists,
valid winners, immutable hashes, and actual Balanced/Max export artifacts.

### Task 8: Benchmark Balanced and Max and compare

**Files:** Runtime artifacts only under the frozen vote4 result root.

- [ ] **Step 1: Run the same vote4 contract on Balanced and Max**

- [ ] **Step 2: Verify exact 51,200-row coverage for each model**

- [ ] **Step 3: Build original/Balanced/Max transition reports**

- [ ] **Step 4: Deliver paths, hashes, category/language tables, and a concise conclusion**

