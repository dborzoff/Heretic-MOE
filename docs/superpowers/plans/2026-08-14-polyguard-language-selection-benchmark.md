# PolyGuard 17-Language Selection Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one text-private 12,240-row PolyGuard geometry benchmark on Qwen3.5-9B, Gemma-4-E4B, and Ministral-3-3B, then recommend 4-6 representative languages with deterministic cross-model k-medoids.

**Architecture:** A deterministic materializer converts the frozen PolyGuard parquet into a manifest plus private per-language JSONL files. The existing geometry runner learns an asymmetric per-direction input contract, emits one automatic 3D package per model, and writes full-space language distance evidence. A separate pure comparison step averages three verified distance matrices and runs fixed-English k-medoids without loading any model again.

**Tech Stack:** Python 3.12, PyTorch, pandas/pyarrow, safetensors, existing Heretic-MOE multi-GPU range queue, dependency-free HTML/JavaScript, pytest, Ruff.

## Global Constraints

- Source parquet: `F:/AI/hf_originals/2026-08-14_safety_bases/ToxicityPrompts__PolyGuardPrompts/data/test-00000-of-00001.parquet`.
- Dataset: 482 SAFE canonical IDs plus 238 UNSAFE canonical IDs across exactly 17 aligned languages, for 12,240 rows.
- SAFE requires `prompt_harm_label=unharmful`, `response_refusal_label=compliance`, and both agreement fields equal to `3`.
- UNSAFE requires `prompt_harm_label=harmful`, `response_refusal_label=refusal`, and both agreement fields equal to `3`.
- Public artifacts and console output contain no prompt, response, answer, question, or text fields.
- Existing balanced five-language `rows_per_cell` inputs remain backward compatible.
- Capture uses one generated token and every selected CUDA device through the existing dynamic range queue.
- Models run sequentially; no two model-level runs overlap.
- Original SAFE points are blue; original UNSAFE points are red.
- Every applicable S1-S14 category is retained and filterable; benign SAFE rows use `benign`.
- English is a mandatory medoid; candidates `k=4,5,6` are all reported.

---

### Task 1: Deterministic PolyGuard dataset materializer

**Files:**
- Create: `src/heretic/polyguard_language_dataset.py`
- Modify: `src/heretic/language_map_cli.py`
- Create: `tests/test_polyguard_language_dataset.py`
- Modify: `tests/test_language_map_cli.py`

**Interfaces:**
- Produces: `materialize_polyguard_language_dataset(source: Path, output_dir: Path) -> dict[str, object]`.
- Produces CLI: `hereticMOE geometry-map prepare-polyguard --source PARQUET --output-dir DIR`.
- Produces manifest fields `schema_version`, `status`, `source_sha256`, `languages`, `directions`, `rows`, `files`, and `selection_contract`.

- [ ] **Step 1: Write failing tests for strict selection and 17-way alignment**

Create synthetic records with two languages and assert the production selector keeps only unanimous harmful/refusal and unanimous unharmful/compliance IDs, preserves canonical order, maps language names to stable codes, retains all category codes, and emits no response column.

```python
manifest = materialize_polyguard_language_dataset(source, output)
assert manifest["directions"] == {"safe": 1, "unsafe": 1}
assert manifest["rows"] == 4
assert read_jsonl(output / "direction_en_unsafe_strict.jsonl")[0]["category_ids"] == ["S1", "S9"]
assert "response" not in read_jsonl(output / "direction_en_unsafe_strict.jsonl")[0]
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest -q tests/test_polyguard_language_dataset.py tests/test_language_map_cli.py`

Expected: collection or import failure because the materializer and subcommand do not exist.

- [ ] **Step 3: Implement the pure selector and atomic materialization**

Read only the required parquet columns. Map the 17 source language names to `en,hi,fr,it,de,pt,th,es,cs,sv,zh,ar,nl,ko,pl,ru,ja`. Validate exactly one row per `(id, language)`, exact ID order across languages, the production counts 482/238, and 12,240 total rows. Write private JSONL atomically and a text-free manifest with SHA-256 for every file.

- [ ] **Step 4: Add the `prepare-polyguard` CLI and text-free result**

The terminal result contains only status, language count, canonical counts, total rows, manifest path, and hashes.

- [ ] **Step 5: Run focused tests and lint**

Run: `python -m pytest -q tests/test_polyguard_language_dataset.py tests/test_language_map_cli.py`

Run: `python -m ruff check src/heretic/polyguard_language_dataset.py src/heretic/language_map_cli.py tests/test_polyguard_language_dataset.py tests/test_language_map_cli.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/heretic/polyguard_language_dataset.py src/heretic/language_map_cli.py tests/test_polyguard_language_dataset.py tests/test_language_map_cli.py
git commit -m "feat: materialize strict PolyGuard language benchmark"
```

### Task 2: Manifest-driven asymmetric geometry input

**Files:**
- Modify: `src/heretic/language_map_data.py`
- Modify: `src/heretic/language_map_cli.py`
- Modify: `src/heretic/language_map_projection.py`
- Modify: `tests/test_language_map_data.py`
- Modify: `tests/test_language_map_cli.py`
- Modify: `tests/test_language_map_projection.py`

**Interfaces:**
- Changes: `GeometryRow.category_ids: tuple[str, ...]` while retaining `category_id`.
- Changes: `load_aligned_corpus(..., expected_per_cell: int | Mapping[Direction, int] | None)`.
- Produces CLI input: `--dataset-manifest PATH`, mutually exclusive with legacy explicit inputs.

- [ ] **Step 1: Write failing asymmetric and multi-category tests**

Assert SAFE can contain two aligned IDs per language while UNSAFE contains one, missing translations fail, category drift fails, and public projection indexes preserve `direction_class` plus `category_ids`.

```python
rows = load_aligned_corpus(files, ("en", "ru"), {"safe": 2, "unsafe": 1})
assert len(rows) == 6
assert rows[-1].category_ids == ("S1", "S9")
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest -q tests/test_language_map_data.py tests/test_language_map_cli.py tests/test_language_map_projection.py`

- [ ] **Step 3: Generalize validation without weakening legacy contracts**

When an integer is supplied, preserve the exact old balanced behavior. When a mapping is supplied, validate each direction count independently and exact cross-language order within that direction. When `None` is supplied by a verified manifest, derive counts from the manifest and reject any file/count/hash mismatch.

- [ ] **Step 4: Preserve complete category metadata in public indexes**

Keep `category_id` as the primary tag and add sorted unique `category_ids`. Reject sensitive keys recursively before writing public artifacts.

- [ ] **Step 5: Run focused tests, old geometry suite, and lint**

Run: `python -m pytest -q tests/test_language_map_data.py tests/test_language_map_cli.py tests/test_language_map_projection.py tests/test_language_map_cache.py tests/test_language_map_parallel.py`

Run: `python -m ruff check src/heretic/language_map_data.py src/heretic/language_map_cli.py src/heretic/language_map_projection.py tests/test_language_map_data.py tests/test_language_map_cli.py tests/test_language_map_projection.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/heretic/language_map_data.py src/heretic/language_map_cli.py src/heretic/language_map_projection.py tests/test_language_map_data.py tests/test_language_map_cli.py tests/test_language_map_projection.py
git commit -m "feat: support asymmetric multilingual geometry manifests"
```

### Task 3: Per-model language distance evidence and k-medoids

**Files:**
- Create: `src/heretic/language_selection.py`
- Modify: `src/heretic/language_map_analysis.py`
- Modify: `src/heretic/language_map_cli.py`
- Create: `tests/test_language_selection.py`
- Modify: `tests/test_language_map_analysis.py`
- Modify: `tests/test_language_map_cli.py`

**Interfaces:**
- Produces: `build_language_distance_report(index: list[dict[str, object]], residuals: Tensor) -> dict[str, object]`.
- Produces: `select_cross_model_medoids(reports: Mapping[str, dict[str, object]], anchor: str = "en", candidates: tuple[int, ...] = (4, 5, 6)) -> dict[str, object]`.
- Produces CLI: `geometry-map compare-languages --analysis NAME=PATH ... --output-dir DIR`.

- [ ] **Step 1: Write synthetic failing tests for paired distances**

Construct four languages where EN and ES have identical direction vectors, RU and ZH are distinct, categories are balanced, and every layer is finite. Assert EN/ES distance is zero, matrix symmetry and diagonal invariants hold, and category interactions affect only the 0.15 component.

- [ ] **Step 2: Write failing deterministic k-medoids tests**

Assert English remains a medoid for every `k`, identical inputs produce byte-identical JSON, every language belongs to exactly one cluster, and tie-breaking uses manifest language order.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `python -m pytest -q tests/test_language_selection.py tests/test_language_map_analysis.py tests/test_language_map_cli.py`

- [ ] **Step 4: Implement per-layer full-space evidence**

For each language and layer calculate SAFE-to-UNSAFE direction, language offset from the paired multilingual centroid, and category-conditioned interaction. Convert each component to a `[0,1]` pair distance. Combine components with weights `0.60/0.25/0.15`; weight layers by SAFE/UNSAFE separation and record every intermediate aggregate without residual vectors.

- [ ] **Step 5: Implement fixed-anchor PAM and cross-model aggregation**

Validate identical language order in all reports, average the three distance matrices, run deterministic PAM for `k=4,5,6` without allowing English to leave the medoid set, calculate silhouette for each model and the aggregate, and recommend the maximum mean silhouette with smaller-`k` tie-breaking.

- [ ] **Step 6: Integrate analysis and comparison CLI artifacts**

Each model writes `language_distances.json`. Cross-model comparison writes `recommended_languages.json`, `distance_matrix.json`, and a text-free summary.

- [ ] **Step 7: Run tests and lint**

Run: `python -m pytest -q tests/test_language_selection.py tests/test_language_map_analysis.py tests/test_language_map_cli.py`

Run: `python -m ruff check src/heretic/language_selection.py src/heretic/language_map_analysis.py src/heretic/language_map_cli.py tests/test_language_selection.py tests/test_language_map_analysis.py tests/test_language_map_cli.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/heretic/language_selection.py src/heretic/language_map_analysis.py src/heretic/language_map_cli.py tests/test_language_selection.py tests/test_language_map_analysis.py tests/test_language_map_cli.py
git commit -m "feat: rank representative languages from geometry"
```

### Task 4: Automatic 3D reports, generated filters, and comparison view

**Files:**
- Modify: `src/heretic/language_map_report.py`
- Modify: `src/heretic/language_map_projection.py`
- Modify: `src/heretic/language_map_cli.py`
- Create: `src/heretic/language_selection_report.py`
- Modify: `tests/test_language_map_report.py`
- Modify: `tests/test_language_map_projection.py`
- Create: `tests/test_language_selection_report.py`

**Interfaces:**
- Produces: `write_language_selection_report(recommendation: Mapping[str, object], output_path: Path) -> dict[str, object]`.
- Changes `geometry-map run` to create `analysis/geometry_3d/report.html` automatically after verified capture.

- [ ] **Step 1: Write failing HTML safety and filter tests**

Assert report metadata discovers all 17 languages from `base_index`, category checkboxes include every `category_ids` value, SAFE legend/canvas color is `#4d9cff`, UNSAFE is `#ff5364`, and no sensitive key or synthetic prompt marker appears in HTML.

- [ ] **Step 2: Write failing automatic-render and comparison-view tests**

Assert a successful `run` creates the immutable 3D package without a journal and that the comparison HTML embeds only numeric matrices, clusters, medoids, language codes, and hashes.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `python -m pytest -q tests/test_language_map_report.py tests/test_language_map_projection.py tests/test_language_selection_report.py tests/test_language_map_cli.py`

- [ ] **Step 4: Implement generated multi-category filters and explicit labels**

Visibility succeeds when a row has at least one selected category. Replace `Original group A/B` copy with `SAFE/UNSAFE`, retain blue/red colors, and generate all language/category controls from metadata.

- [ ] **Step 5: Automatically freeze the pooled per-layer PCA package**

After analysis, create `analysis/geometry_3d` from the verified cache, render the offline HTML, and include its path/hash in the stage result. Never fit a separate basis per language.

- [ ] **Step 6: Implement the cross-model heatmap and cluster report**

Render the aggregate distance heatmap, `k=4,5,6` cluster memberships, recommended medoids, silhouette values, and per-model stability. Keep the report dependency-free and text-free.

- [ ] **Step 7: Run focused tests, complete geometry suite, and lint**

Run: `python -m pytest -q tests/test_language_map_*.py tests/test_language_selection*.py`

Run: `python -m ruff check src/heretic/language_map_*.py src/heretic/language_selection*.py tests/test_language_map_*.py tests/test_language_selection*.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add src/heretic/language_map_report.py src/heretic/language_map_projection.py src/heretic/language_map_cli.py src/heretic/language_selection_report.py tests/test_language_map_report.py tests/test_language_map_projection.py tests/test_language_selection_report.py
git commit -m "feat: render multilingual selection reports"
```

### Task 5: Materialize, verify, run three models, and compare

**Files:**
- Runtime output: `F:/AI/hf_originals/heretic_out/research/datasets/polyguard17_strict_v1/`
- Runtime output: `F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_qwen35_9b_v1/`
- Runtime output: `F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_gemma4_e4b_v1/`
- Runtime output: `F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_ministral3_3b_v1/`
- Runtime output: `F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_cross_model_v1/`

**Interfaces:**
- Consumes all commands created by Tasks 1-4.
- Produces three verified caches, three 3D reports, and one cross-model recommendation.

- [ ] **Step 1: Run the complete CPU test and source-quality gates**

Run: `python -m pytest -q`

Run: `python -m ruff check src/heretic tests`

Run: `git diff --check`

Expected: all tests pass; Ruff and diff check pass.

- [ ] **Step 2: Materialize the frozen 12,240-row dataset**

```powershell
python -m heretic.language_map_cli prepare-polyguard --source 'F:/AI/hf_originals/2026-08-14_safety_bases/ToxicityPrompts__PolyGuardPrompts/data/test-00000-of-00001.parquet' --output-dir 'F:/AI/hf_originals/heretic_out/research/datasets/polyguard17_strict_v1'
```

Verify manifest status, 17 languages, SAFE=482, UNSAFE=238, total=12,240, exact file hashes, and no public sensitive keys.

- [ ] **Step 3: Check for existing GPU workers and run Qwen visibly on both GPUs**

Use one foreground console and `--devices auto`; do not start if another Heretic or model process owns either selected GPU.

```powershell
python -m heretic.language_map_cli run --dataset-manifest 'F:/AI/hf_originals/heretic_out/research/datasets/polyguard17_strict_v1/manifest.json' --model 'F:/AI/hf_originals/LLM/Qwen__Qwen3.5-9B' --output-dir 'F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_qwen35_9b_v1' --devices auto
```

- [ ] **Step 4: Verify Qwen completion before starting Gemma**

Require `rows=12240`, cache `status=PASS`, finite residuals, complete HTML hash, and both workers complete. Only then start Gemma with the identical command shape and its own model/output paths.

- [ ] **Step 5: Verify Gemma completion before starting Ministral**

Require the same gates, then run Ministral sequentially with both GPUs.

- [ ] **Step 6: Compare all three verified analyses**

```powershell
python -m heretic.language_map_cli compare-languages --analysis qwen='F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_qwen35_9b_v1/analysis/language_distances.json' --analysis gemma='F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_gemma4_e4b_v1/analysis/language_distances.json' --analysis ministral='F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_ministral3_3b_v1/analysis/language_distances.json' --output-dir 'F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/polyguard17_cross_model_v1'
```

- [ ] **Step 7: Inspect evidence without reading corpus text**

Open each 3D HTML and the cross-model report. Check SAFE/UNSAFE colors, automatic 17-language filters, category filters, EN/ES placement, k=4/5/6 alternatives, and reported medoids. Record only counts, codes, distances, hashes, and selected languages.

- [ ] **Step 8: Commit the final production changes if runtime gates pass**

Do not commit runtime caches or private JSONL data. Commit only source/tests/docs still pending, and report exact artifact paths and verification evidence.
