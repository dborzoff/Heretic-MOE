# Heretic-MOE Four-Language Corpus v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Materialize the frozen 1000/400/200 SAFE+UNSAFE four-language corpus, make it the Heretic-MOE runtime contract, and verify it on two GPUs with an 8B or 9B model.

**Architecture:** A deterministic text-private materializer ranks already completed consensus results, allocates category-stratified disjoint memberships, and writes aligned JSONL plus hash manifests. The multilingual loader consumes map, trial, and bidirectional final files directly. Finalist evaluation computes removal only on UNSAFE rows and preservation geometry on SAFE rows.

**Tech Stack:** Python 3.12, PyTorch, Pydantic, JSONL/SHA-256, pytest, two CUDA GPUs.

## Global Constraints

- Never print prompt or response text.
- Preserve source prompt bytes and canonical translation alignment.
- Frozen languages are en, ru, zh, ja.
- Pool sizes per language and direction are map 1000, trial 400, final 200.
- Map/trial/final canonical IDs are disjoint within each direction.
- All public manifests and reports are text-free.
- Existing dirty self-classification and translation files are preserved.

---

### Task 1: Deterministic v4 materializer

**Files:**
- Create: src/heretic/four_language_corpus.py
- Create: tests/test_four_language_corpus.py

**Interfaces:**
- Consumes the frozen SAFE selection manifest, fullstats dataset manifest, and fullstats consensus rows.
- Produces build_four_language_corpus(output_root, sources) and a text-free root manifest.

- [ ] Write synthetic failing tests for exact pool sizes, category allocation, ranking, four-language order, disjoint IDs, and prohibited public fields.
- [ ] Run the focused test and confirm RED.
- [ ] Implement largest-remainder category quotas and deterministic within-category ranking.
- [ ] Implement aligned JSONL/category-index writing through atomic staging.
- [ ] Implement mechanical verification and SHA-256 manifests.
- [ ] Run the focused test and confirm GREEN.

### Task 2: Materialize the real corpus

**Files:**
- Create: F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_4lang_v4/

**Interfaces:**
- Uses build_four_language_corpus from Task 1.
- Produces map/trial/final files, category indexes, reserve memberships, and manifest.json.

- [ ] Run the materializer against the completed 912 SAFE and 1664 UNSAFE consensus artifacts.
- [ ] Verify 12,800 total language cells, exact counts, source hashes, and zero cross-pool ID overlap.
- [ ] Verify category totals and identical canonical order across languages.
- [ ] Load all JSONL without emitting corpus text.

### Task 3: Public YAML and multilingual dataset contract

**Files:**
- Modify: src/heretic/launch_config.py
- Modify: src/heretic/config.py
- Modify: src/heretic/multilingual_contract.py
- Modify: tests/test_launch_config.py
- Modify: tests/test_config.py
- Modify: tests/test_multilingual_contract.py

**Interfaces:**
- DataSettings exposes languages, direction_rows_per_cell, trial_rows_per_cell, and final_rows_per_cell.
- MultilingualDatasetBundle final_rows include both SAFE and UNSAFE direction metadata.

- [ ] Write failing config tests for en/ru/zh/ja and 1000/400/200.
- [ ] Write failing loader tests for final_{lang}_{safe|unsafe}_200.jsonl and cross-pool disjointness.
- [ ] Run focused tests and confirm RED.
- [ ] Implement YAML-to-internal-settings propagation and replace the frozen five-language validator.
- [ ] Load final rows as aligned directional rows and bump dataset contract schema.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Bidirectional finalist holdout

**Files:**
- Modify: src/heretic/multilingual_final_holdout.py
- Modify: src/heretic/multilingual_final_holdout_cli.py
- Modify: src/heretic/multilingual_runtime.py
- Modify: src/heretic/multilingual_prepare_cli.py
- Modify: src/heretic/multilingual_reference_worker.py
- Modify: tests/test_multilingual_final_holdout.py
- Modify: tests/test_multilingual_runtime.py

**Interfaces:**
- Clean and candidate archives preserve row direction in the row contract.
- Final public metrics expose UNSAFE removal/SRG and SAFE preservation geometry separately.

- [ ] Write failing tests proving SAFE rows are not included in SRG removal and are included in safe geometry damage.
- [ ] Write failing archive/hash tests proving direction changes invalidate resume.
- [ ] Run focused tests and confirm RED.
- [ ] Propagate final_rows_per_cell through preparation, workers, and finalist runtime.
- [ ] Evaluate SRG and R-side on UNSAFE indices only; evaluate geometry on the full directional row set.
- [ ] Add text-free per-direction counts and preservation diagnostics.
- [ ] Run focused tests and confirm GREEN.

### Task 5: Profile, static verification, and cleanup

**Files:**
- Create: config.heretic_moe_4lang_v4.yaml
- Modify: README.md

**Interfaces:**
- One public YAML launches the 4-language v4 pipeline.

- [ ] Add the real dataset paths and 1000/400/200 contract to the YAML.
- [ ] Run Ruff on changed production and test files.
- [ ] Run all focused multilingual/config/finalist tests.
- [ ] Run the complete CPU pytest suite.
- [ ] Remove only generated failed-run scratch artifacts; preserve research evidence and user changes.

### Task 6: Two-GPU 8B/9B smoke

**Files:**
- Create: F:/AI/hf_originals/heretic_out/research/searches/4lang_v4_smoke/

**Interfaces:**
- Uses config.heretic_moe_4lang_v4.yaml with model and run-root CLI overrides.

- [ ] Run dry-run and prove both GPUs, all three pools, counts, hashes, and frozen run contract.
- [ ] Run preparation/map/reference on an available 8B or 9B local model.
- [ ] Run a short two-GPU search and bidirectional finalist evaluation.
- [ ] Verify text-free reports, live worker distribution, resume safety, and final metrics.
- [ ] Send the morning Telegram report with paths, timings, category counts, test results, and open GPU-only caveats.
