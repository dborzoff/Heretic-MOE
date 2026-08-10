# Qwen3.6 Heretic-MoE-v3 Hugging Face Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one English root model card for the Balanced and Max roles of the Qwen3.6-35B-A3B Heretic-MoE-v3 release.

**Architecture:** Stage one `README.md` outside the source checkout while the build runs. The card exposes one Balanced/Max table and switches from an explicit build-status statement to verified trials, metrics, files, sizes, and hashes only after the release manifests pass.

**Tech Stack:** Hugging Face model-card Markdown/YAML and PowerShell/Python mechanical validation.

## Global Constraints

- The exact release title is `Qwen3.6-35B-A3B Heretic-MoE-v3`.
- All public prose is English.
- There is exactly one model card for both Balanced and Max.
- If both roles select one physical checkpoint, Max is an alias and weights are not duplicated.
- No evaluation prompt or generated response text is published.
- Unsupported GPU formats are omitted rather than inferred.

---

### Task 1: Stage the build-in-progress card

**Files:**
- Modify: `F:/AI/tmp/qwen36_hf_card/README.md`

**Interfaces:**
- Consumes: the approved card design and the upstream model identifier `Qwen/Qwen3.6-35B-A3B`.
- Produces: one upload-ready root `README.md` whose factual final fields remain explicitly in progress.

- [ ] **Step 1: Run the pre-change invariant check**

```powershell
$card = Get-Content -Raw F:\AI\tmp\qwen36_hf_card\README.md
if ($card -notmatch '(?m)^# Qwen3\.6-35B-A3B Heretic-MoE-v3$') { exit 1 }
```

Expected: FAIL because the old heading does not use the approved exact release title.

- [ ] **Step 2: Update the staged card**

Use one H1 title, one Balanced/Max table, an explicit `Build in progress` gate, the GitHub link, the search/recheck contract, artifact families, compatibility limits, reproducibility evidence, and license boundary. Do not create nested model cards.

- [ ] **Step 3: Validate the staged card**

```powershell
$card = Get-Content -Raw F:\AI\tmp\qwen36_hf_card\README.md
if (($card | Select-String '(?m)^---$' -AllMatches).Matches.Count -lt 2) { exit 1 }
if (($card | Select-String '(?m)^# ' -AllMatches).Matches.Count -ne 1) { exit 1 }
if ($card -notmatch '(?m)^# Qwen3\.6-35B-A3B Heretic-MoE-v3$') { exit 1 }
if ($card -notmatch 'https://github\.com/dborzoff/Heretic-MOE') { exit 1 }
if (($card | Select-String '(?m)^\| Variant \|' -AllMatches).Matches.Count -ne 1) { exit 1 }
if ($card -match '\b(T[B]D|TO[D]O)\b') { exit 1 }
```

Expected: exit code 0.

### Task 2: Finalize the same card from verified release evidence

**Files:**
- Modify: `F:/AI/tmp/qwen36_hf_card/README.md`

**Interfaces:**
- Consumes: PASS winner/export/qfabrik manifests, uploaded HF paths, local SHA-256 values, file sizes, and pinned tool revisions.
- Produces: the final root model card uploaded to `DmitryDB/Qwen3.6-35B-A3B-Heretic-MOE-v3`.

- [ ] **Step 1: Verify evidence before editing**

Require exact Balanced/Max role resolution, PASS export manifests, PASS qfabrik manifest, local artifact size/hash matches, and remote path existence. A missing requirement keeps the card in build-in-progress state.

- [ ] **Step 2: Replace only provisional release state**

Add exact trial identifiers, recheck metrics, alias state, artifact paths, bytes, SHA-256 values, Heretic-MOE commit, llama.cpp commit, qfabrik version, and validation status. Remove `Build in progress` only when all published rows are verified.

- [ ] **Step 3: Re-run the invariant check and upload**

Run the Task 1 validator, verify no public prompt/response fields are present, upload `README.md` to the repository root, and read the remote file back before declaring the card published.
