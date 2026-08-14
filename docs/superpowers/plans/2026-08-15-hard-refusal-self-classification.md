# Multilingual Self-Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a text-private `hereticMOE self-classify` pipeline that compares three deterministic response-strategy prompts on Qwen3.5-9B, selects the strongest format, and applies it to local chat models on all available GPUs.

**Architecture:** Pure contract/template/parser code is isolated from model inference. A controller validates the existing PolyGuard manifest, materializes only allowed metadata in a private runtime job, and launches one resident worker per GPU/model. Atomic JSONL checkpoints feed a separate aggregation/report module, so inference can resume without loading prompt text into reports.

**Tech Stack:** Python 3.12, PyTorch/Transformers through the existing `heretic.model.Model`, JSONL/CSV, stdlib HTML, pytest, Ruff.

## Global Constraints

- Languages are exactly `en`, `ru`, `zh`, `ko`.
- Classes are exactly `DIRECT`, `PARTIAL`, `SOFT`, `HARD_REFUSE`; `INVALID` is parser state only.
- Prompt and raw model-output text may exist only in worker memory and must never be written to console, JSONL, CSV, HTML, or manifests.
- Pilot covers 720 canonical IDs in four languages and all three variants: 8,640 classifications.
- Full benchmark uses the winning variant on every compatible local chat model that fits one RTX 4090.
- Inference is deterministic and resumable by `(model_id, row_id, variant)`.

---

### Task 1: Classification contracts, localized templates, and strict parsers

**Files:**
- Create: `src/heretic/self_classification.py`
- Test: `tests/test_self_classification.py`

**Interfaces:**
- Produces: `BehaviorClass`, `PromptVariant`, `ClassificationInput`, `ClassificationResult`, `render_classifier_prompt(row, variant)`, `parse_classification_output(text, row_id, variant)`, and `permuted_code_map(row_id)`.
- Consumes: only scalar metadata and prompt text supplied in memory by the caller.

- [ ] **Step 1: Write failing tests for exact classes and technical INVALID handling**

  Cover exact phrase, numeric, and code parsing; reject extra prose; prove all permutations are deterministic bijections; assert serialized results have no `prompt`, `response`, or `raw_output` field.

- [ ] **Step 2: Run the focused test and confirm RED**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification.py`

- [ ] **Step 3: Implement immutable enums/dataclasses, four localized wrappers, and strict parsers**

  The phrase variant emits one localized sentence, number emits `1`–`4`, and code emits `A`–`D`. Use SHA-256 of `row_id` to rotate the code-to-class order. Keep technical parse failure in `valid=false, classification=null`.

- [ ] **Step 4: Run focused tests and Ruff**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification.py`

  Run: `.venv\Scripts\ruff.exe check src/heretic/self_classification.py tests/test_self_classification.py`

- [ ] **Step 5: Commit**

  Commit: `feat: add multilingual self-classification contracts`

### Task 2: Verified PolyGuard input loader and text-free checkpoints

**Files:**
- Create: `src/heretic/self_classification_data.py`
- Test: `tests/test_self_classification_data.py`

**Interfaces:**
- Consumes: schema-v1 PolyGuard `manifest.json`, selected languages, and optional completed-result keys.
- Produces: `load_classification_rows(manifest_path, languages) -> list[ClassificationInput]`, `append_result_atomic(path, result)`, `load_completed_keys(path)`, and `verify_result_coverage(...)`.

- [ ] **Step 1: Write failing tests using synthetic manifests and prompt sentinels**

  Verify hashes, exact aligned-ID coverage, language filtering, category/direction preservation, atomic resume, duplicate rejection, and absence of prompt sentinels from every output artifact.

- [ ] **Step 2: Run the focused test and confirm RED**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification_data.py`

- [ ] **Step 3: Implement strict manifest loading and one-record-per-line checkpointing**

  Reuse `get_file_sha256`; keep prompt only on `ClassificationInput`; write a minimal public result dictionary from `ClassificationResult`. Flush and `fsync` every completed batch, then validate the just-written tail.

- [ ] **Step 4: Run focused tests and Ruff**

  Run both new test modules, then Ruff both production modules and tests.

- [ ] **Step 5: Commit**

  Commit: `feat: add verified self-classification data flow`

### Task 3: Resident GPU inference worker and adaptive batching

**Files:**
- Create: `src/heretic/self_classification_worker.py`
- Test: `tests/test_self_classification_worker.py`

**Interfaces:**
- Consumes: private job JSON, local model path, one visible CUDA device, pending classification rows.
- Produces: text-free batch events and atomic `ClassificationResult` records.
- Depends on: `Model.prepare_prompt_cache`, `Model.get_responses_batched`, and Task 1 parsers.

- [ ] **Step 1: Write failing worker tests with a fake model**

  Assert deterministic prompt order, resume skipping, OOM batch halving, no raw text in events/results, and correct output-token counts.

- [ ] **Step 2: Run focused worker tests and confirm RED**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification_worker.py`

- [ ] **Step 3: Implement model loading and inference loop**

  Build `Settings` with no quantization, BF16, greedy generation, prompt cache, 10% VRAM headroom, and maximum output length derived from the variant. Set `CUDA_VISIBLE_DEVICES` before importing torch/model code. On OOM halve batch and retry without rebuilding the model.

- [ ] **Step 4: Verify tests and lint**

  Run all three focused modules and Ruff their files.

- [ ] **Step 5: Commit**

  Commit: `feat: add resumable self-classification worker`

### Task 4: Pilot scoring, model aggregation, and HTML reporting

**Files:**
- Create: `src/heretic/self_classification_report.py`
- Test: `tests/test_self_classification_report.py`

**Interfaces:**
- Consumes: text-free result rows plus model byte sizes.
- Produces: `select_prompt_variant(rows)`, `write_pilot_reports(...)`, `write_model_reports(...)`, JSON/CSV summaries, and standalone HTML.

- [ ] **Step 1: Write failing aggregation tests**

  Construct synthetic rows that prove lexicographic selection: valid rate, majority agreement, translation agreement, then shorter-output tie-break. Verify per-language/category/direction/model totals reconcile exactly and HTML contains no sentinels.

- [ ] **Step 2: Run the report test and confirm RED**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification_report.py`

- [ ] **Step 3: Implement deterministic statistics and a compact filterable HTML table**

  Include class distributions, variant/model agreement matrices, language flip rates, invalid rates, and category/direction pivots. Embed only aggregated statistics and text-free row metadata.

- [ ] **Step 4: Verify tests and lint**

  Run all four focused modules and Ruff their files.

- [ ] **Step 5: Commit**

  Commit: `feat: report self-classification benchmark`

### Task 5: Public CLI, two-GPU model queue, pilot, and full run

**Files:**
- Create: `src/heretic/self_classification_cli.py`
- Modify: `src/heretic/cli.py`
- Test: `tests/test_self_classification_cli.py`

**Interfaces:**
- Adds: `hereticMOE self-classify pilot`, `hereticMOE self-classify run-models`, and internal `worker`.
- Consumes: dataset manifest, output root, pilot model, model root, device list.
- Produces: private jobs, worker processes, final reports, and a hash manifest.

- [ ] **Step 1: Write failing CLI tests**

  Verify command dispatch, dry-run counts of `2880 × 3`, model compatibility filtering, one-model-per-device scheduling, failed-model isolation, winner reuse, and manifest hashes.

- [ ] **Step 2: Run CLI tests and confirm RED**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification_cli.py`

- [ ] **Step 3: Implement controller and CLI routing**

  Use subprocess workers with one resident model per physical GPU. Pilot first; require a verified pilot winner before `run-models`; discover only local chat models below the supplied root; store failure reasons without prompt text.

- [ ] **Step 4: Run focused and full verification**

  Run all five new test modules, the full pytest suite, Ruff on the production path, and `self-classify pilot --dry-run`. Confirm exact 8,640 pilot tasks and zero prompt fields in artifacts.

- [ ] **Step 5: Commit**

  Commit: `feat: add self-classification benchmark pipeline`

- [ ] **Step 6: Run the real pilot and model benchmark**

  Pilot model: `F:\AI\hf_originals\LLM\Qwen__Qwen3.5-9B`.

  Dataset manifest: `F:\AI\hf_originals\heretic_out\research\datasets\polyguard_language_geometry_v1\manifest.json`.

  Model root: `F:\AI\hf_originals\LLM`.

  Output root: `F:\AI\hf_originals\heretic_out\research\results\polyguard_self_classification_v1`.

  Use devices `0,1`, verify no overlapping geometry/ComfyUI job before launch, keep one visible foreground controller window, and validate final hashes/counts before reporting completion.
