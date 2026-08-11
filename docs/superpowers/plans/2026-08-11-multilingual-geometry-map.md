# Multilingual Geometry Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a neutral `hereticMOE geometry-map` diagnostic that measures every aligned prompt once, caches per-layer BF16-model residual geometry, and evaluates language/prompt subsets without another model invocation.

**Architecture:** A strict JSONL loader creates a text-free aligned row index. A capture module calls the existing Heretic first-generated-token residual path in bounded batches and writes an immutable tensor cache. A separate analyzer operates only on that cache to calculate fuzzy language/category/direction maps and virtual-corpus reconstruction error. The CLI dispatches before the normal multi-GPU search supervisor, so no Optuna study or export can start accidentally.

Multi-GPU capture uses one resident model process per selected GPU and a durable
SQLite queue of global row ranges. Workers produce independent atomic parts;
the controller verifies and merges them in canonical order. Dynamic claiming
lets faster GPUs complete more ranges without reloading either model.

**Tech Stack:** Python 3.12, PyTorch, safetensors, NumPy, Pydantic/Heretic Settings, pytest.

## Global Constraints

- Operative train is 1,200 SAFE plus 1,200 UNSAFE canonical prompts in EN/RU/ZH/ES/FR: 12,000 measured rows.
- Test is 200 SAFE plus 200 UNSAFE canonical prompts; its operative language assignment is selected from the train map.
- Each model/prompt row is measured once; every virtual corpus is calculated from cache.
- Original unmodified BF16 checkpoints only.
- Console, manifests, JSON reports, and HTML are prompt/response-text-free.
- Exact canonical IDs, direction, category, and five-way language coverage are hard gates.
- Categories are nested within direction, not crossed as common SAFE/UNSAFE factor levels.
- Start with Gemma-4-E4B; do not make a general language-policy claim from one architecture.

---

### Task 1: Aligned multilingual corpus contract

**Files:**
- Create: `src/heretic/language_map_data.py`
- Create: `tests/test_language_map_data.py`

**Interfaces:**
- Produces `LanguageFile(language: str, direction: Literal["safe", "unsafe"], path: Path)`.
- Produces `GeometryRow(canonical_id, row_id, language, direction, category_id, prompt, source_path, source_line)`.
- Produces `load_aligned_corpus(files, expected_languages, expected_per_cell) -> list[GeometryRow]`.
- Produces `text_free_row_index(rows) -> list[dict[str, object]]` with no `prompt` key.

- [ ] **Step 1: Write failing loader tests**

```python
def test_loads_balanced_aligned_rows_without_leaking_prompt(tmp_path):
    files = write_two_language_safe_unsafe_fixture(tmp_path)
    rows = load_aligned_corpus(files, ("en", "ru"), expected_per_cell=2)
    assert len(rows) == 8
    assert "prompt" not in text_free_row_index(rows)[0]

def test_rejects_missing_translation_and_category_drift(tmp_path):
    files = write_fixture_with_missing_ru_and_category_drift(tmp_path)
    with pytest.raises(ValueError, match="canonical coverage|category drift"):
        load_aligned_corpus(files, ("en", "ru"), expected_per_cell=2)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_language_map_data.py -q`

Expected: import failure because `heretic.language_map_data` does not exist.

- [ ] **Step 3: Implement strict JSONL parsing and alignment**

Implementation requirements:

```python
@dataclass(frozen=True)
class GeometryRow:
    canonical_id: str
    row_id: str
    language: str
    direction: Literal["safe", "unsafe"]
    category_id: str
    prompt: str
    source_path: Path
    source_line: int
```

Reject blank prompts, duplicate row IDs, duplicate `(direction, language,
canonical_id)`, metadata drift across translations, missing/extra canonical IDs,
unexpected languages/directions, and wrong cell counts. Preserve input order as
SAFE languages in requested language order followed by UNSAFE languages.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_language_map_data.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/language_map_data.py tests/test_language_map_data.py
git commit -m "feat: validate multilingual geometry corpus"
```

---

### Task 2: One-time residual cache capture

**Files:**
- Create: `src/heretic/language_map_cache.py`
- Create: `tests/test_language_map_cache.py`
- Modify: `src/heretic/model.py`

**Interfaces:**
- Consumes `GeometryRow` from Task 1.
- Produces `capture_residual_cache(model, rows, batch_size, output_dir, progress) -> CacheManifest`.
- Produces `load_residual_cache(output_dir) -> tuple[list[dict], Tensor, dict]`.
- Adds `Model.iter_residual_batches(prompts, batch_size)` so capture does not duplicate generation semantics.

- [ ] **Step 1: Write failing cache tests with a fake model**

```python
class FakeResidualModel:
    def iter_residual_batches(self, prompts, batch_size):
        for start in range(0, len(prompts), batch_size):
            count = min(batch_size, len(prompts) - start)
            yield torch.arange(count * 2 * 3).reshape(count, 2, 3).float()

def test_capture_calls_each_row_once_and_round_trips(tmp_path):
    manifest = capture_residual_cache(FakeResidualModel(), rows, 2, tmp_path)
    index, residuals, loaded = load_residual_cache(tmp_path)
    assert manifest.rows == len(rows) == residuals.shape[0]
    assert loaded["measurement_position"] == "first_generated_token"
    assert all("prompt" not in row for row in index)
```

Also test interruption safety: final filenames must not exist until the tensor,
row index, and manifest have all been validated.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_language_map_cache.py -q`

Expected: import failure because `heretic.language_map_cache` does not exist.

- [ ] **Step 3: Add the model batch iterator**

```python
def iter_residual_batches(self, prompts, batch_size):
    for batch in batchify(prompts, batch_size):
        yield self.get_residuals(batch)
```

Keep `get_residuals_batched` behavior by concatenating this iterator. This makes
the diagnostic reuse the exact operative Heretic measurement position.

- [ ] **Step 4: Implement atomic cache writing**

Convert each `GeometryRow.prompt` to `Prompt(system=system_prompt, user=prompt)`.
Accumulate CPU float32 batches in deterministic row order, concatenate once, and
write `residuals.safetensors.tmp`, `row_index.jsonl.tmp`, and `manifest.json.tmp`.
Verify finite values, shape `[rows, layers, hidden]`, counts, hashes, and no text
keys before atomic rename. The manifest records model/config/tokenizer hashes,
input hashes, dtype, measurement position, shape, and code commit.

- [ ] **Step 5: Run focused and existing residual tests**

Run: `pytest tests/test_language_map_cache.py tests/test_model_residuals.py -q`

If `tests/test_model_residuals.py` does not exist, run:
`pytest tests -q -k "residual or language_map_cache"`.

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/heretic/model.py src/heretic/language_map_cache.py tests/test_language_map_cache.py
git commit -m "feat: cache multilingual residual geometry"
```

---

### Task 3: Cached fuzzy geometry and virtual-corpus analysis

**Files:**
- Create: `src/heretic/language_map_analysis.py`
- Create: `tests/test_language_map_analysis.py`

**Interfaces:**
- Produces `analyze_geometry(index, residuals, seed=42) -> GeometryReport`.
- Produces `virtual_policy_indices(index, policy, seed=42) -> ndarray`.
- Produces `write_geometry_reports(report, output_dir)`.

- [ ] **Step 1: Write failing synthetic-geometry tests**

```python
def test_redundant_language_has_zero_marginal_direction_loss():
    index, residuals = redundant_two_language_fixture()
    report = analyze_geometry(index, residuals)
    assert report["language_contributions"]["ru"]["direction_loss"] < 1e-6

def test_unique_language_branch_is_retained_as_cold_not_deleted():
    index, residuals = one_unique_language_fixture()
    report = analyze_geometry(index, residuals)
    assert report["language_contributions"]["ru"]["direction_loss"] > 0
    assert report["temperature"]["cold_observations"] > 0
    assert report["temperature"]["discarded_observations"] == 0

def test_nonduplicated_policy_uses_one_language_per_canonical_id():
    selected = virtual_policy_indices(index, "cycle_languages", seed=42)
    assert canonical_ids(selected).is_unique
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_language_map_analysis.py -q`

Expected: import failure because `heretic.language_map_analysis` does not exist.

- [ ] **Step 3: Implement continuous map statistics**

For every layer calculate:

- robust translation-family center and distances;
- adaptive bandwidth from median/MAD distance;
- RBF heat without deleting observations;
- SAFE/UNSAFE mean directions;
- language and nested-category centroids;
- canonical translation cosine distributions;
- language-by-direction and language-by-category(direction) energy shares.

Use deterministic float64 reductions on CPU. Store aggregate statistics only;
never place raw vectors or prompts into JSON/HTML.

- [ ] **Step 4: Implement virtual corpus policies**

Implement `full`, `en_only`, `leave_out:<language>`, `cycle_languages`, and
`weighted:<json>` policies. Recompute means/directions from cached row indices,
then compare each policy to full using per-layer cosine, norm ratio, retained
heat, lost branches, and reconstruction error. The cycle policy uses one
language per canonical ID and never includes translated duplicates.

- [ ] **Step 5: Write text-free JSON and HTML reports**

Write `layer_statistics.json`, `factor_map.json`,
`language_contributions.json`, `subset_candidates.json`, and `report.html`.
The HTML embeds aggregate JSON and tables only. Add a mechanical forbidden-key
scan for `prompt`, `response`, `answer`, and `text` before PASS.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `pytest tests/test_language_map_analysis.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/heretic/language_map_analysis.py tests/test_language_map_analysis.py
git commit -m "feat: analyze cached multilingual geometry"
```

---

### Task 4: `hereticMOE geometry-map` CLI and CPU integration test

**Files:**
- Create: `src/heretic/language_map_cli.py`
- Modify: `src/heretic/cli.py`
- Create: `tests/test_language_map_cli.py`
- Modify: `README.md`

**Interfaces:**
- Adds `hereticMOE geometry-map analyze --cache-dir ...`.
- Adds `hereticMOE geometry-map run ...` as capture followed by analyze.

- [ ] **Step 1: Write failing CLI dispatch and dry-run tests**

```python
def test_cli_dispatches_language_map_without_supervisor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hereticMOE", "geometry-map", "--help"])
    assert cli.main() is None

def test_dry_run_validates_without_loading_model(tmp_path):
    result = language_map_cli.run(["run", "--dry-run", *fixture_args(tmp_path)])
    assert result["status"] == "PASS"
    assert result["rows"] == 8
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_cli.py -q`

Expected: CLI does not recognize `geometry-map`.

- [ ] **Step 3: Implement early subcommand dispatch**

In `heretic.cli.main`, inspect the first argument before importing supervisor or
the ML worker. Dispatch `geometry-map` to `language_map_cli.main(argv[1:])`.
The parser accepts repeated `--safe LANG=PATH` and `--unsafe LANG=PATH`,
`--expected-languages`, `--expected-per-cell`, `--model`, `--output-dir`,
`--batch-size`, `--device`, `--seed`, and `--dry-run`.

- [ ] **Step 4: Implement model loading and progress**

Set `CUDA_VISIBLE_DEVICES` before importing torch/model code when `--device` is
provided. Construct `Settings` with `dtypes=["bfloat16"]`, no quantization,
`device_map="auto"`, fixed seed, and CPU output offload. Print only
`rows_done/rows_total`, rows/s, ETA, RAM/VRAM, hashes, and artifact paths.

- [ ] **Step 5: Document exact commands**

Document a dry run and an operative run with five SAFE and five UNSAFE files.
State that `run` never invokes Optuna, never modifies a model, and never prints
prompt text.

- [ ] **Step 6: Run CLI and regression tests**

Run:

```powershell
pytest tests/test_language_map_data.py tests/test_language_map_cache.py tests/test_language_map_analysis.py tests/test_language_map_cli.py -q
pytest tests -q -k "cli or residual or language_map"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/heretic/cli.py src/heretic/language_map_cli.py tests/test_language_map_cli.py README.md
git commit -m "feat: add language geometry diagnostic command"
```

---

### Task 5: Gemma smoke and operative-run preparation

**Files:**
- Create outside git: `F:/AI/hf_originals/heretic_out/research/results/multilingual_geometry_v1/gemma4_e4b/`
- Update: `docs/superpowers/plans/2026-08-11-multilingual-geometry-map.md` checkboxes only after evidence exists.

**Interfaces:**
- Consumes the CLI from Task 4 and frozen dataset files.
- Produces the first verified cache/report directory; no model export.

- [x] **Step 1: Run text-free five-language dry-run validation**

Run `hereticMOE geometry-map run --dry-run` from the frozen corpus root with
EN/RU/ZH/ES/FR and 1,200 rows per cell.

Evidence (2026-08-11): PASS, 12,000 rows, exact coverage, no GPU allocation.

- [x] **Step 2: Run a 20-row-per-cell Gemma smoke**

Use frozen slices generated by the loader, BF16 Gemma-4-E4B, one visible
PowerShell window, and a dedicated smoke output directory. Verify cache shape,
finite values, hashes, and cached re-analysis without reloading the model.

Evidence (2026-08-11): EN/RU, 20 rows per cell, 80/80 rows, 43 layers,
hidden size 2560, cache SHA-256 `25d2c2c70aa173820abb204360cf3fd6807ccc09a1a14291e782860d01de0a03`.
The cache-only replay used 0 MiB on GPU 1 and produced five byte-identical reports.

- [x] **Step 3: Gate the 12,000-row run**

Re-check live GPU use and require all ten final five-language train files at
1,200 rows each. If FR/ZH are incomplete, stop at READY and do not infer a
language policy from EN/RU/ES.

- [ ] **Step 4: Run full Gemma capture once when the gate passes** *(running)*

Launch in a visible UTF-8 PowerShell window. Verify advancing row counters and
final cache/report hashes. Re-run `geometry-map analyze` from cache and confirm
that the model is not loaded.

- [ ] **Step 5: Decide whether to run Ministral and Qwen**

Proceed with the unchanged manifest only if Gemma cache verification and
synthetic-math tests pass. Any change to row selection, position, or analysis
settings requires a new versioned cache rather than overwriting Gemma output.

---

### Task 6: Durable global range queue

**Files:**
- Create: `src/heretic/range_work_queue.py`
- Create: `tests/test_range_work_queue.py`

**Interfaces:**
- Produces `RangeWorkQueue(path).initialize(row_count, rows_per_task, fingerprint)`.
- Produces `claim(worker_id) -> RangeWorkItem | None`.
- Produces `complete(item, part_file, sha256, shape)` and `fail(item, error_type)`.
- Produces `verify_parts(parts_dir)`, `release_worker(worker_id)`, and `stats()`.

- [ ] **Step 1: Write failing queue tests**

Test exact range coverage, dynamic claiming by a faster worker, release after a
worker failure, contract mismatch rejection, and requeue of a missing or
hash-mismatched part.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_range_work_queue.py -q`

Expected: import failure because `heretic.range_work_queue` does not exist.

- [ ] **Step 3: Implement the minimal SQLite queue**

Use `BEGIN IMMEDIATE`, WAL, `synchronous=FULL`, attempt-scoped claims, and atomic
state transitions. Store only numeric ranges, hashes, shapes, and worker IDs;
never store prompts.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_range_work_queue.py -q`

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/range_work_queue.py tests/test_range_work_queue.py
git commit -m "feat: queue geometry capture ranges"
```

---

### Task 7: Resident GPU workers and canonical merge

**Files:**
- Modify: `src/heretic/language_map_cache.py`
- Create: `src/heretic/language_map_parallel.py`
- Create: `tests/test_language_map_parallel.py`

**Interfaces:**
- Produces `capture_claimed_ranges(model, rows, queue, parts_dir, worker_id, batch_size)`.
- Produces `finalize_range_cache(rows, queue, parts_dir, output_dir, metadata)`.

- [ ] **Step 1: Write failing worker and merge tests**

Use two fake resident workers of different speed. Assert that the faster worker
claims more tasks, every row is measured once, output returns to canonical
order, an interrupted restart measures only missing ranges, and a tampered part
is rejected before finalization.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_parallel.py -q`

- [ ] **Step 3: Implement range capture and finalization**

Each claimed range may contain multiple model batches. Concatenate only that
range, validate finite `[rows,layers,hidden]` values, write one atomic part, and
record its hash. Finalization loads verified parts by global start row and
publishes the existing cache contract with the manifest last.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_language_map_parallel.py tests/test_language_map_cache.py -q`

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/language_map_cache.py src/heretic/language_map_parallel.py tests/test_language_map_parallel.py
git commit -m "feat: capture geometry on resident GPU workers"
```

---

### Task 8: One-window multi-GPU CLI

**Files:**
- Modify: `src/heretic/language_map_cli.py`
- Modify: `tests/test_language_map_cli.py`
- Modify: `README.md`

**Interfaces:**
- Adds `--devices auto|0,1,...`, `--task-rows`, and `--cpu-threads-per-worker`.
- Keeps `--device N` as a parse-time error when combined with `--devices`.
- Adds an internal worker entry point that is not exposed as a public workflow.

- [ ] **Step 1: Write failing CLI/controller tests**

Assert device discovery, one child per selected GPU, GPU-prefixed progress in
one controller stream, nonzero worker exit propagation, and rejection of an
already-busy explicit device unless the user overrides the gate.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_cli.py -q`

- [ ] **Step 3: Implement controller and internal workers**

Set `CUDA_VISIBLE_DEVICES` independently before each worker imports the model.
Keep workers resident, stream prefixed stdout/stderr to the controller, poll
queue counts, release failed claims, and finalize only when all ranges verify.

- [ ] **Step 4: Run targeted regression tests**

Run:

```powershell
pytest tests/test_range_work_queue.py tests/test_language_map_parallel.py tests/test_language_map_cache.py tests/test_language_map_cli.py -q
```

- [ ] **Step 5: Run a visible two-GPU smoke when both GPUs are free**

Use a new output directory and a complete aligned slice. Verify one public
controller, two resident workers, dynamic unequal task counts, exact coverage,
hashes, canonical merge, and cache-only re-analysis.

- [ ] **Step 6: Commit**

```powershell
git add src/heretic/language_map_cli.py tests/test_language_map_cli.py README.md
git commit -m "feat: run geometry capture across available GPUs"
```

---

### Task 9: Frozen three-dimensional basis and base-point package

**Files:**
- Create: `src/heretic/language_map_projection.py`
- Create: `tests/test_language_map_projection.py`

**Interfaces:**
- Produces `fit_frozen_projection(residuals, seed=42) -> ProjectionBasis`.
- Produces `project_residuals(residuals, basis) -> ndarray[float32]`.
- Produces `write_base_projection(cache_dir, output_dir, seed=42)`.

- [ ] **Step 1: Write failing deterministic projection tests**

Test exact output shapes, finite coordinates, byte-identical basis and points on
repeat, stable component signs, projection without refitting, text-free indexes,
and a retained-shift ratio computed against the full-space displacement.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_projection.py -q`

- [ ] **Step 3: Implement frozen per-layer PCA and atomic artifacts**

Use deterministic full SVD for small inputs and deterministic randomized PCA for
operative caches. Fix component signs by forcing the largest absolute loading
positive. Record per-layer centers, components, explained variance, algorithm,
seed, input hash, and basis hash in safetensors metadata/manifest. Write base
coordinates as contiguous float32 binary plus a text-free row index.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_language_map_projection.py -q`

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/language_map_projection.py tests/test_language_map_projection.py
git commit -m "feat: freeze 3d geometry projection"
```

---

### Task 10: Journal timeline and append-only trial coordinates

**Files:**
- Create: `src/heretic/language_map_trajectory.py`
- Create: `tests/test_language_map_trajectory.py`

**Interfaces:**
- Produces `load_text_free_trial_timeline(journal) -> list[TrialRecord]`.
- Produces `select_stratified_anchors(index, count, seed) -> list[int]`.
- Produces `initialize_trajectory_package(cache_dir, journal, output_dir, ...)`.
- Produces `append_trial_projection(package_dir, trial_record, residuals)`.

- [ ] **Step 1: Write failing journal/package tests**

Create a synthetic Optuna journal and assert canonical trial order, numeric-only
metadata, explicit `not_captured` state, deterministic balanced anchors,
append-only duplicate rejection, atomic writes, verdict isolation, and source
trial hashes.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_trajectory.py -q`

- [ ] **Step 3: Implement timeline and package writer**

Read Optuna through its public journal storage API. Copy only state, number,
phase, parameters, numeric values, constraints, and selected numeric user attrs.
Do not copy settings, prompt paths, response archives, or arbitrary strings.
Append projected trial blocks under a lock and publish their index only after
shape/hash/finite checks pass.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_language_map_trajectory.py -q`

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/language_map_trajectory.py tests/test_language_map_trajectory.py
git commit -m "feat: track geometry across search trials"
```

---

### Task 11: Self-contained filterable 3D HTML

**Files:**
- Create: `src/heretic/language_map_report.py`
- Create: `tests/test_language_map_report.py`
- Modify: `src/heretic/language_map_analysis.py`

**Interfaces:**
- Produces `write_interactive_geometry_report(package_dir, output_path)`.

- [ ] **Step 1: Write failing report contract tests**

Assert offline HTML with embedded typed arrays, no external URLs, controls for
layer/language/group/category/trial/phase/finalist/verdict/arrows, animation,
Original reset, solid Original-to-trial and dashed optimizer-path legends, and
a forbidden-key/text scan.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_report.py -q`

- [ ] **Step 3: Implement the offline 3D renderer**

Use a dependency-free canvas renderer with deterministic camera math, mouse
rotation, pan/zoom, depth sorting, filters, sliders, animation, centroid arrows,
and optional individual arrows. Embed base64 float32 arrays and safe metadata;
do not use a CDN or `file://` fetch.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_language_map_report.py -q`

- [ ] **Step 5: Commit**

```powershell
git add src/heretic/language_map_report.py src/heretic/language_map_analysis.py tests/test_language_map_report.py
git commit -m "feat: render interactive 3d search geometry"
```

---

### Task 12: Geometry trajectory CLI and live trial hook

**Files:**
- Modify: `src/heretic/language_map_cli.py`
- Modify: `src/heretic/config.py`
- Modify: `src/heretic/main.py`
- Modify: `tests/test_language_map_cli.py`
- Create: `tests/test_trial_geometry_capture.py`
- Modify: `README.md`

**Interfaces:**
- Adds `hereticMOE geometry-map project --cache-dir ... --journal ...`.
- Adds `hereticMOE geometry-map render --package-dir ...`.
- Adds an opt-in live trial capture hook using a frozen private anchor file.

- [ ] **Step 1: Write failing CLI and hook tests**

Assert journal-only initialization, no invented coordinates, optional capture
of first-token residuals from the normal evaluation generation, anchor capture
after scoring/before reset, exact source-trial identity, no duplicate answer
generation, and fail-closed artifact publication.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_language_map_cli.py tests/test_trial_geometry_capture.py -q`

- [ ] **Step 3: Implement CLI and opt-in live hook**

The `project` command builds the base map, private anchor input, text-free
timeline, and HTML. The live hook records first-token residuals from the normal
evaluation generation, then measures the small anchor control on the already
edited model, projects both with the frozen basis, appends the trial, and leaves
scores unchanged. Existing journals remain immutable and missing coordinates
remain `not_captured` until replayed.

- [ ] **Step 4: Run targeted and full verification**

Run:

```powershell
pytest tests/test_language_map_projection.py tests/test_language_map_trajectory.py tests/test_language_map_report.py tests/test_language_map_cli.py tests/test_trial_geometry_capture.py -q
pytest tests -q
```

- [ ] **Step 5: Build the Gemma base package from the verified cache**

Run `geometry-map project` against the 12,000-row Gemma cache. Mechanically
verify counts, hashes, offline HTML, all filter controls, and zero sensitive
keys/text in public artifacts. Do not start a model replay without a separate
visible-run decision.

- [ ] **Step 6: Commit**

```powershell
git add src/heretic/language_map_cli.py src/heretic/config.py src/heretic/main.py tests/test_language_map_cli.py tests/test_trial_geometry_capture.py README.md docs/superpowers
git commit -m "feat: capture search trajectory geometry"
```
