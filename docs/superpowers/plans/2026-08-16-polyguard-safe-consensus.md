# PolyGuard SAFE Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 400–600 four-language SAFE prompts with the highest reproducible DIRECT-response rate across the existing twelve-model, eight-variant panel.

**Architecture:** Reuse the completed 12-model self-classification controller. First classify every English PolyGuard row jointly labelled `safe` and `unharmful`, exclude exact overlaps with the frozen Heretic corpus, then classify only the English TOP-600 in RU/ZH/JA and rank by the weakest language before the global rate.

**Tech Stack:** Python 3.12, PyArrow, Heretic `self_classification_cli`, JSONL/SHA-256 artifacts, two CUDA workers.

## Global Constraints

- Never print or store prompt/response text in logs, reports, checkpoints, or public result rows.
- Source prompts remain byte-identical to the downloaded PolyGuard parquet.
- English eligibility requires `prompt_label=safe` and `prompt_harm_label=unharmful`.
- Exact normalized overlap with the frozen 1400 SAFE and 1664 UNSAFE English corpus is forbidden.
- Use exactly the existing 12 models and 8 prompt variants.
- Final ranking order is minimum per-language DIRECT rate, global DIRECT rate, fully consistent model-language cells, invalid count, canonical ID.

---

### Task 1: Materialize the verified English candidate packet

**Files:**
- Create: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard_safe_en_stage_v1/manifest.json`
- Create: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard_safe_en_stage_v1/direction_en_safe_strict.jsonl`
- Create: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard_safe_en_stage_v1/direction_en_unsafe_strict.jsonl`

- [ ] Read only metadata plus the prompt column required to copy/hash rows; emit no text.
- [ ] Keep all English rows jointly labelled safe and unharmful.
- [ ] Reject normalized exact overlaps against the frozen English SAFE/UNSAFE files.
- [ ] Write a schema-v1 manifest with counts and SHA-256 values.
- [ ] Load the packet through `load_classification_rows` and verify expected coverage.

### Task 2: Run and verify English consensus

**Files:**
- Create: `F:/AI/hf_originals/heretic_out/research/results/polyguard_safe_en_consensus_v1/consensus/`
- Create: `F:/AI/hf_originals/heretic_out/research/results/polyguard_safe_en_consensus_v1.console.log`

- [ ] Launch `python -m heretic.self_classification_cli consensus` in a visible PowerShell window with devices 0 and 1, language `en`, batch size 32, the frozen 12-model list, and the eight consensus variants.
- [ ] Verify exact result coverage, zero prohibited text fields, manifest PASS, and all output hashes.
- [ ] Rank canonical IDs by DIRECT count across the expected 96 English classifications, then invalid count and canonical ID.
- [ ] Freeze the deterministic English TOP-600 membership with scores and source/result hashes.

### Task 3: Materialize and classify aligned RU/ZH/JA rows

**Files:**
- Create: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard_safe_top600_4lang_v1/`
- Create: `F:/AI/hf_originals/heretic_out/research/results/polyguard_safe_top600_3lang_consensus_v1/consensus/`

- [ ] Copy the PolyGuard RU/ZH/JA translations matching the frozen English TOP-600 IDs in identical order.
- [ ] Verify exact four-language ID coverage, source labels, non-empty prompts, and SHA-256 values.
- [ ] Run the same 12 models and 8 variants for languages `ru,zh,ja` in a visible two-GPU window.
- [ ] Verify exact result coverage and zero prohibited text fields.

### Task 4: Select and publish the final SAFE set

**Files:**
- Create: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard_safe_final_4lang_v1/`
- Create: `F:/AI/hf_originals/heretic_out/research/results/polyguard_safe_final_4lang_v1/selection_manifest.json`
- Create: `F:/AI/hf_originals/heretic_out/research/results/polyguard_safe_final_4lang_v1/report.html`

- [ ] Merge the EN and RU/ZH/JA text-free results by model, canonical ID, language, and variant.
- [ ] Compute per-language DIRECT rates, minimum language rate, global rate, fully consistent cells, and invalid count.
- [ ] If 400–600 IDs are 100% DIRECT, keep all; if more than 600, keep the best 600; if fewer than 400, keep the best 400 and record the cutoff.
- [ ] Materialize identical final membership in EN/RU/ZH/JA source files.
- [ ] Verify hashes, counts, ID order, disjointness from frozen old data, and absence of prompt/response fields in reports.
- [ ] Keep every non-selected ID available as an untouched validation pool.
