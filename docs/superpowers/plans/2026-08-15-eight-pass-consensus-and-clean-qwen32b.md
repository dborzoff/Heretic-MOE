# Eight-Pass Consensus and Clean Qwen32B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean Qwen3-VL-32B IQ4_XS control and produce text-private eight-pass behavior classifications whose strict consensus excludes Heretic and Qwen3-0.6B-Base.

**Architecture:** Extend the existing self-classification contracts with four cyclic A-D mappings and four cyclic full-word option orders. Run a metadata-only format diagnostic on Ministral before the full queue, convert the clean 32B source through the existing qfabrik GGUF/imatrix DAG, then aggregate original-model consensus while reporting Heretic only as a comparison.

**Tech Stack:** Python 3.12, PyTorch/Transformers, llama.cpp, DDB-qfabrik, JSONL, pytest, Ruff.

## Global Constraints

- Prompt text and raw model output never appear in console, JSONL, CSV, HTML, or manifests.
- Languages remain exactly `en`, `ru`, `zh`, `ko`; diagnostic comparison uses aligned `en`, `ru`, `ko` IDs.
- Variants are four code shifts and four word-order shifts; generation is deterministic with thinking disabled.
- `Qwen3-0.6B-Base` is excluded from reruns and all panels.
- `Qwen3-VL-32B_HereticMOE-B-IQ4_XS.gguf` runs all variants but never votes.
- Clean `Qwen3-VL-32B-Instruct-IQ4_XS.gguf` votes after GGUF hash, structure, and load checks pass.
- No quantization fallback and no imatrix reuse from Heretic.

---

### Task 1: Diagnose Ministral format failures without raw text

**Files:**
- Modify: `src/heretic/self_classification.py`
- Modify: `src/heretic/self_classification_worker.py`
- Test: `tests/test_self_classification.py`
- Test: `tests/test_self_classification_worker.py`

**Interfaces:**
- Produces: optional `system_mode: localized|english`, explicit generation limit, and `classify_output_shape(text, expected_outputs) -> str`.
- Output shapes: `exact`, `quoted`, `json_single_label`, `label_punctuation`, `label_plus_text`, `unknown`; no raw text is serialized.

- [ ] **Step 1: Write failing tests**

  Assert English meta-instructions can wrap RU/KO rows, generation limits are honored, and representative synthetic outputs map to shape names without retaining their contents.

- [ ] **Step 2: Run RED tests**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification.py tests/test_self_classification_worker.py`

- [ ] **Step 3: Implement the minimal diagnostic controls**

  Keep legacy defaults unchanged; job JSON may opt into the English system and a positive token limit. Add only shape metadata to diagnostic summaries, never to ordinary result rows unless it is a fixed allow-listed shape string.

- [ ] **Step 4: Run a Ministral diagnostic matrix**

  Use the same aligned EN/RU/KO IDs for `4,8,16,32` tokens and localized/English system modes. Report validity, EOS/cap behavior, and output-shape counts only.

- [ ] **Step 5: Commit**

  Commit: `feat: diagnose self-classification format failures`

### Task 2: Add four code shifts and four word-order shifts

**Files:**
- Modify: `src/heretic/self_classification.py`
- Modify: `src/heretic/self_classification_worker.py`
- Modify: `src/heretic/self_classification_cli.py`
- Test: `tests/test_self_classification.py`
- Test: `tests/test_self_classification_worker.py`
- Test: `tests/test_self_classification_cli.py`

**Interfaces:**
- Produces variants `code_permuted`, `code_shift_1`, `code_shift_2`, `code_shift_3`, `word_order_0`, `word_order_1`, `word_order_2`, `word_order_3`.
- `code_permuted` remains byte-compatible with the completed shift-0 archive.

- [ ] **Step 1: Write failing permutation tests**

  For every class, assert the four code shifts map it once to each of A-D. Assert the four word variants preserve output words while rotating option positions once through all four slots.

- [ ] **Step 2: Run RED tests**

  Run the three focused modules and confirm the new enum members are absent.

- [ ] **Step 3: Implement rendering and parsing**

  Use one English system contract for new variants, rotate only deterministic metadata, set the token limit from the diagnostic result, and preserve strict exact parsing.

- [ ] **Step 4: Verify focused tests and Ruff**

  Run focused pytest and Ruff on all modified production/test files.

- [ ] **Step 5: Commit**

  Commit: `feat: add eight-pass classification variants`

### Task 3: Add strict consensus aggregation and exclusions

**Files:**
- Modify: `src/heretic/self_classification_report.py`
- Modify: `src/heretic/self_classification_cli.py`
- Test: `tests/test_self_classification_report.py`
- Test: `tests/test_self_classification_cli.py`

**Interfaces:**
- Produces: per-model `code_4of4`, `word_4of4`, `strict_8of8`; clean-panel unanimous groups; separate Heretic comparison; text-free JSON/CSV/HTML.

- [ ] **Step 1: Write failing aggregation tests**

  Synthetic rows prove that one disagreement or INVALID removes an ID from strict consensus, Heretic cannot change membership, and Qwen3-0.6B historical rows are ignored.

- [ ] **Step 2: Run RED tests**

  Run: `F:\AI\heretic_env\Scripts\python.exe -m pytest -q tests/test_self_classification_report.py tests/test_self_classification_cli.py`

- [ ] **Step 3: Implement exact aggregation**

  Do not fill class/category shortages by relaxing thresholds. Record zero coverage explicitly and keep provenance hashes for every source result file.

- [ ] **Step 4: Verify focused tests and Ruff**

  Confirm reports contain no prompt/response fields or sentinel text.

- [ ] **Step 5: Commit**

  Commit: `feat: aggregate strict classification consensus`

### Task 4: Build clean Qwen3-VL-32B IQ4_XS

**Files:**
- Create outside repo: `F:/AI/hf_originals/heretic_out/research/quant_jobs/qwen3vl32b-clean-iq4-xs.json`
- Produce: `F:/AI/ComfyUI_windows_portable/ComfyUI/models/LLM/Qwen3-VL-32B_Instruct-IQ4_XS.gguf`

**Interfaces:**
- Consumes source `F:/AI/hf_originals/LLM/Qwen__Qwen3-VL-32B-Instruct`, generic verified GGUF recipe, and a new model-specific imatrix run.
- Produces validated IQ4_XS plus execution manifest and SHA-256.

- [ ] **Step 1: Materialize and dry-run the explicit qfabrik job**

  Request only `GGUF-IQ4_XS`, target role `llm`, profile `GGUF-Full-Imatrix`, and the installed llama.cpp toolchain. Dry-run must report no fallback and sufficient disk.

- [ ] **Step 2: Execute in a visible window**

  Convert HF to F16 GGUF, compute a fresh imatrix on GPU, and quantize IQ4_XS. Do not run classification concurrently with imatrix.

- [ ] **Step 3: Validate and install**

  Verify GGUF structure/architecture/file type, SHA-256, and successful `llama-server` load. Move the validated final file to the requested LLM directory; delete only verified build intermediates after the final hash is recorded.

### Task 5: Execute the full eight-pass panel

**Files:**
- Produce outside repo: `F:/AI/hf_originals/heretic_out/research/results/polyguard_self_classification_v3/**`

**Interfaces:**
- Consumes the existing shift-0 rows where fingerprints match, seven new passes for existing eligible models, and eight passes for clean Qwen32B.
- Produces strict consensus groups and Heretic comparison.

- [ ] **Step 1: Run dry-run and exact task accounting**

  Expected new classifications: 305,280. Refuse resume if dataset/model/variant hashes differ.

- [ ] **Step 2: Launch one visible two-GPU controller**

  Keep one resident model per GPU, dynamically take the next model-pass job, and checkpoint atomically by `(model_id,row_id,variant)`.

- [ ] **Step 3: Validate final artifacts**

  Verify exact coverage, no duplicate keys, allow-listed labels/shapes, zero prohibited text fields, source/result hashes, and reconciliation of detailed/aggregate counts.

- [ ] **Step 4: Run project verification**

  Run full pytest, Ruff on the production path, and `git diff --check` before the final commit/report.

