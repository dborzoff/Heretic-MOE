# Multimodel SRG v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a language-conditioned sparse refusal geometry bank from nine systems across Ministral, Qwen3.6, Gemma 3, and Qwen3.5 without contaminating search or final holdouts.

**Architecture:** A separate 300-prompt calibration split feeds a resume-safe response archive generator. Transformers checkpoints and GGUF files share the same archive schema. Two blind judges produce nine-label decisions, a generic builder converts agreed/adjudicated labels into balanced per-language prototype banks, and the scorer macro-averages independent English, Russian, and Chinese geometries.

**Tech Stack:** Python 3.12, PyTorch/Transformers, llama.cpp server b8840, NumPy, SciPy, scikit-learn, pytest, JSONL, Agent Chat.

## Global Constraints

- Calibration languages are exactly `en`, `ru`, and `zh`; `zh` is the Chinese language code.
- Final calibration size is exactly 100 prompts per language and 10 prompts per category, 300 total.
- Calibration prompts have zero exact or near-duplicate overlap with direction, search, old adaptive-search data, and final holdouts.
- Generation is greedy, thinking-off where supported, and capped at 128 new tokens.
- Prompt and response text never appears in console, manifests, Agent Chat, or test output.
- Frozen `final234` and Qwen v3 responses remain evaluation-only.
- Qwen3.6 v2 Balanced/Max use one IQ4_XS file each; multiple quants of identical weights are not separate systems.
- Gemma Heretic uses `GGUF/max/Gemma-3-12B-IT-QAT-HereticMOE-Max-Q4_0.gguf` at repository revision `e6575173e81b9a4b7152ce89af897291f2fc602f`.
- No new search starts and no scorer default changes unless all release gates pass.
- Coding work is committed in small independently testable commits.

---

### Task 1: Freeze the independent SRG calibration split

**Files:**
- Create externally: `F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_3lang_v1/stage_a/srg_calibration_candidates.jsonl`
- Create externally: `F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_3lang_v1/final/srg_calibration_unsafe.jsonl`
- Modify externally: `F:/AI/hf_originals/heretic_out/research/datasets/heretic_moe_3lang_v1/final/manifest.json`

**Interfaces:**
- Consumes: Stage A exclusions and the ten-category taxonomy already sent to `C_judge`.
- Produces: 300-row JSONL with `base_id`, `language`, `category`, `difficulty`, and `prompt` fields.

- [ ] **Step 1: Extend the Stage A contract**

Send `C_judge` a text-free follow-up requiring 120 additional calibration candidates per language, 12 per category, disjoint from every other split.

- [ ] **Step 2: Extend the Stage B contract**

Send `C_judge2` a blind-audit follow-up requiring exact final quotas of 100 per language and 10 per category, with no quota weakening.

- [ ] **Step 3: Verify the frozen output mechanically**

Run a text-free verifier that asserts 300 rows, three languages, 100 rows per language, ten rows per language/category cell, unique IDs, non-empty prompts, and the overlap matrix reported as zero.

- [ ] **Step 4: Record immutable evidence**

Record the final JSONL SHA-256 and manifest SHA-256 in the SRG v2 run manifest. Do not copy prompt text into the repository.

### Task 2: Add GGUF generation to the existing response archive tool

**Files:**
- Modify: `research/scripts/generate_response_archives.py`
- Modify: `tests/test_response_archive.py`

**Interfaces:**
- Consumes: `--model LABEL=PATH`, where PATH is either a Transformers directory or a `.gguf` file.
- Produces: the existing `<label>.responses.jsonl` schema and `manifest.json`, with `backend`, `model_sha256`, and text-free runtime fields.

- [ ] **Step 1: Write failing model-spec tests**

Add tests equivalent to:

```python
def test_parse_model_accepts_gguf(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    spec = parse_model(f"qwen={model}")
    assert spec.label == "qwen"
    assert spec.backend == "gguf"


def test_parse_model_keeps_transformers_directory(tmp_path):
    model = tmp_path / "hf"
    model.mkdir()
    spec = parse_model(f"gemma={model}")
    assert spec.backend == "transformers"
```

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_response_archive.py -q`

Expected: FAIL because the current parser accepts directories only and has no backend field.

- [ ] **Step 3: Introduce a typed model specification**

Add:

```python
@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    backend: Literal["transformers", "gguf"]
```

Make `parse_model()` select `gguf` for a `.gguf` file and `transformers` for a directory, rejecting every other path type.

- [ ] **Step 4: Write failing request/manifest tests**

Mock the llama.cpp HTTP endpoint and assert that the request uses one user message, `temperature=0`, `max_tokens=128`, and does not log prompt content. Assert that GGUF manifests use the whole-file SHA-256 instead of `config.json`.

- [ ] **Step 5: Implement the llama.cpp backend**

Start the configured `llama-server.exe` with `--model`, `--host 127.0.0.1`, an assigned port, `--ctx-size 512`, `--parallel 1`, `--gpu-layers 999`, `--split-mode none`, `--main-gpu`, and `--log-disable`. Poll `/health`, call `/v1/chat/completions`, append each completed row immediately, and terminate only the child server started by this invocation.

- [ ] **Step 6: Preserve resume and text-free behavior**

Reuse `load_completed()`, preserve exact prompt order, include `generated_tokens` and `hit_token_cap`, and print only model label, numeric progress, rates, and paths.

- [ ] **Step 7: Run focused and full CPU-safe tests**

Run:

```powershell
F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_response_archive.py -q
F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_sparse_refusal_geometry.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the generator backend**

```powershell
git add research/scripts/generate_response_archives.py tests/test_response_archive.py
git commit -m "feat: generate response archives from GGUF models"
```

### Task 3: Preserve judge disagreement and build consensus inputs

**Files:**
- Modify: `research/scripts/compare_blind_judges.py`
- Create: `tests/test_compare_blind_judges.py`

**Interfaces:**
- Consumes: two complete blind nine-label JSONL files with identical `(system_code, id)` keys.
- Produces: text-free comparison report, `agreed_labels.jsonl`, and `resolution_queue.jsonl` containing keys and both labels but no prompt/answer text.

- [ ] **Step 1: Write failing consensus-output tests**

Use four synthetic keys: two exact agreements, one soft-family disagreement, and one delivered/refusal boundary inversion. Assert that only exact agreements enter `agreed_labels.jsonl` and both disagreements enter the resolution queue.

- [ ] **Step 2: Run the test and observe failure**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_compare_blind_judges.py -q`

Expected: FAIL because the current script only writes an aggregate report.

- [ ] **Step 3: Add explicit output arguments**

Add `--agreed-output` and `--resolution-output`; refuse to overwrite either path; preserve `system_code`, `id`, agreed label, and both confidences without corpus text.

- [ ] **Step 4: Run the new test and existing suite**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_compare_blind_judges.py tests/test_response_archive.py -q`

Expected: PASS.

- [ ] **Step 5: Commit consensus handling**

```powershell
git add research/scripts/compare_blind_judges.py tests/test_compare_blind_judges.py
git commit -m "feat: preserve blind judge consensus and disagreements"
```

### Task 4: Generalize the prototype-bank builder

**Files:**
- Modify: `research/scripts/build_sparse_geometry_prototypes.py`
- Create: `tests/test_build_sparse_geometry_prototypes.py`

**Interfaces:**
- Consumes: calibration prompts, canonical response archives, canonical labels, and a system metadata JSON mapping each system to `model_family`.
- Produces: `prototypes.en.jsonl`, `prototypes.ru.jsonl`, `prototypes.zh.jsonl`, and a pinned v2 `manifest.json`.

- [ ] **Step 1: Write failing validation tests**

Create synthetic archives and assert rejection of prompt mismatch, language mismatch, duplicate `(system,id)`, holdout hash overlap, missing family metadata, and labels outside the nine-label schema.

- [ ] **Step 2: Run the tests and observe failure**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_build_sparse_geometry_prototypes.py -q`

Expected: FAIL because the current builder is hard-coded to Ministral paths and six systems.

- [ ] **Step 3: Replace hard-coded paths with CLI arguments**

Implement repeated `--archive LABEL=PATH`, plus `--prompts`, `--labels`, `--systems`, `--output-dir`, and repeated `--exclude-manifest` arguments.

- [ ] **Step 4: Implement nine-label projection and weights**

Write rows with `id`, `base_id`, `language`, `system`, `model_family`, `prompt`, `answer`, `label`, and `weight`. Map `comply` to delivered weight 1.0, `comply_degraded` to delivered weight 0.75, soft labels to soft weight 1.0, and `refuse_policy` to refuse weight 1.0. Keep partial/other counts in the manifest but exclude them from the main three-class files unless adjudicated.

- [ ] **Step 5: Equalize family/class contribution**

Normalize weights so each present `(language, model_family, label)` cell sums to 1.0. Record raw and effective counts in the manifest.

- [ ] **Step 6: Run tests and a text-free dry build**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_build_sparse_geometry_prototypes.py -q`

Expected: PASS with no prompt strings in captured stdout.

- [ ] **Step 7: Commit the generic builder**

```powershell
git add research/scripts/build_sparse_geometry_prototypes.py tests/test_build_sparse_geometry_prototypes.py
git commit -m "feat: build balanced multilingual SRG banks"
```

### Task 5: Add language-conditioned weighted SRG scoring

**Files:**
- Modify: `src/heretic/scorers/sparse_refusal_geometry.py`
- Modify: `tests/test_sparse_refusal_geometry.py`

**Interfaces:**
- Consumes: the v2 manifest path through the existing `prototypes` setting and prompt-language values from the configured local JSONL.
- Produces: macro-average SRG, aggregate R-side, and per-language class margins in diagnostics.

- [ ] **Step 1: Write failing manifest/language tests**

Use three tiny language banks with distinct vocabularies. Assert that English queries use only the English vectorizers, Russian only Russian, and Chinese only Chinese; assert failure on an unsupported or missing language.

- [ ] **Step 2: Write failing weighted-centroid tests**

Give one family ten duplicate rows and another family one row with equal effective total weight. Assert that the centroid matches equal family contribution rather than raw row count.

- [ ] **Step 3: Run the focused tests and observe failure**

Run: `F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_sparse_refusal_geometry.py -q`

Expected: FAIL because bank v1 is a single unweighted JSONL and prompt metadata is discarded.

- [ ] **Step 4: Add v2 manifest loading while retaining v1 read support**

Detect `.json` versus `.jsonl`. For a v2 manifest, verify all bank hashes, load one bank per language, and fit independent char, word, pair, and centroid geometry objects.

- [ ] **Step 5: Read and align local prompt languages**

For JSONL prompt sources, read only `language` and the configured prompt column, apply the same split slice, and validate exact prompt text alignment without emitting text. Reject non-local sources for v2 multilingual mode.

- [ ] **Step 6: Implement weighted top-k and centroids**

Use prototype weights for centroid means and normalize per-class top-k weights before averaging. Preserve same-prompt exclusion by matching stable `base_id`, not positional integer alone.

- [ ] **Step 7: Aggregate and report diagnostics**

Macro-average the three language mean margins. Report `per_language.{lang}.rows`, `mean_margin`, `positive_rate`, direct-refusal margin, and soft margin. Keep prompt/response arrays out of diagnostics.

- [ ] **Step 8: Run scorer and full target tests**

Run:

```powershell
F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_sparse_refusal_geometry.py -q
F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_response_archive.py tests/test_compare_blind_judges.py tests/test_build_sparse_geometry_prototypes.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit scorer v2**

```powershell
git add src/heretic/scorers/sparse_refusal_geometry.py tests/test_sparse_refusal_geometry.py
git commit -m "feat: score multilingual weighted refusal geometry"
```

### Task 6: Acquire pinned local GGUF sources and generate archives

**Files:**
- Create externally: `F:/AI/hf_originals/heretic_out/srg_sources/`
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/responses/`
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/systems.json`

**Interfaces:**
- Consumes: the frozen calibration JSONL and pinned model files.
- Produces: nine aligned 300-row response archives and one text-free manifest per run.

- [ ] **Step 1: Download pinned Heretic GGUF files**

Download Qwen v2 Balanced IQ4_XS (17.86 GiB), Qwen v2 Max IQ4_XS (17.86 GiB), and Gemma Heretic Max Q4_0 (6.41 GiB) with repository SHA pins and resume enabled. Verify file sizes and SHA-256 locally.

- [ ] **Step 2: Prepare the original controls**

Use local Ministral BF16, local Qwen3.5-9B, local Gemma original control, and convert the local 67-GiB Qwen3.6 original master to one IQ4_XS through the existing qfabrik recipe. Record that Qwen3.6 comparisons all use IQ4-class inference.

- [ ] **Step 3: Prepare selected Heretic controls**

Reproduce/export Ministral trial 150 from its pinned journal. Use the existing selected/published Qwen3.5 Heretic source. Verify model configuration and weight hashes before generation.

- [ ] **Step 4: Run a text-free smoke test**

Generate only three calibration IDs per system. Require nine archives, 27 total rows, no empty answers, correct IDs/languages, and no corpus text in console logs.

- [ ] **Step 5: Generate the small local families**

Run Ministral, Gemma, and Qwen3.5 systems first, one model per GPU where possible, with cap 128 and immediate resume-safe writes.

- [ ] **Step 6: Generate Qwen v2 IQ4 in parallel**

Run Balanced on GPU 0 and Max on GPU 1 using independent llama-server ports. If GPU 0 remains occupied by ComfyUI, run sequentially on GPU 1 rather than evicting the user's process.

- [ ] **Step 7: Generate original Qwen3.6**

Run the pinned original IQ4 control on a free GPU under the identical prompt and generation contract.

- [ ] **Step 8: Validate all archives**

Require exactly 300 rows and unique IDs per system, byte-identical prompt fields across systems, zero empty responses, recorded token-cap counts, pinned model hashes, and a PASS manifest.

### Task 7: Blind judge, adjudicate, and build bank v2

**Files:**
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/blind_judge1/`
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/blind_judge2/`
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/bank_v2/`

**Interfaces:**
- Consumes: nine aligned response archives.
- Produces: two blind label files, agreed labels, adjudicated labels, three prototype banks, and a pinned manifest.

- [ ] **Step 1: Build independent blind packets**

Use different deterministic seeds for each judge and keep both private maps outside judge-facing directories.

- [ ] **Step 2: Send complete contracts to both judges**

Require solo judging without subagents, nine labels, exact coverage, no source-model inference, and text-free Agent Chat reports.

- [ ] **Step 3: Validate and compare both outputs**

Require 2,700 rows from each judge. Produce exact/coarse agreement, critical boundary inversions, agreed labels, and a resolution queue.

- [ ] **Step 4: Adjudicate every disagreement**

Do not use majority logic with only two judges. Obtain an explicit final label for each disagreement and preserve the original two labels.

- [ ] **Step 5: Build bank v2**

Run the generic builder, verify per-language/per-family/per-class counts, and pin all input/output hashes.

### Task 8: Validate against bank v1 and frozen Qwen v3

**Files:**
- Create: `research/scripts/evaluate_sparse_geometry_bank.py`
- Create: `tests/test_evaluate_sparse_geometry_bank.py`
- Create externally: `F:/AI/hf_originals/heretic_out/research/results/srg_v2/evaluation.json`

**Interfaces:**
- Consumes: bank v1, bank v2, frozen labeled validation archives, and model-family metadata.
- Produces: text-free row metrics, system-ranking metrics, per-language results, and a release verdict.

- [ ] **Step 1: Write failing grouped-evaluation tests**

Assert prompt-ID grouping, leave-one-family-out grouping, and rejection of any overlap between bank and validation IDs.

- [ ] **Step 2: Implement deterministic evaluation**

Report per-row ROC-AUC/PR-AUC, system-level Spearman/Pearson against two-judge refusalish rate, per-language ordering, and repeated-score maximum absolute difference.

- [ ] **Step 3: Run the release evaluation**

Compare bank v1 and bank v2, then score frozen Qwen v2/v3 `final234` outputs without adding them to training.

- [ ] **Step 4: Apply the hard release gate**

PASS only if bank v2 improves system-level Spearman, ranks v3 more refusalish than v2, has no per-language ordering reversal, and remains deterministic. Otherwise retain bank v1 and record FAIL.

- [ ] **Step 5: Run the full target suite**

Run:

```powershell
F:/AI/heretic_env/Scripts/python.exe -m pytest tests/test_response_archive.py tests/test_compare_blind_judges.py tests/test_build_sparse_geometry_prototypes.py tests/test_sparse_refusal_geometry.py tests/test_evaluate_sparse_geometry_bank.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit evaluation tooling**

```powershell
git add research/scripts/evaluate_sparse_geometry_bank.py tests/test_evaluate_sparse_geometry_bank.py
git commit -m "feat: validate multimodel SRG banks"
```

- [ ] **Step 7: Change the default only after PASS**

Update the pinned SRG manifest/hash in the next search configuration, run a three-prompt smoke test, and commit the configuration separately. On FAIL, make no default change.

