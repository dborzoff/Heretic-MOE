# Multilingual Heretic geometry-map diagnostic

## Goal

Measure every aligned prompt exactly once per original model, cache its internal
geometry, and use only deterministic mathematics over the cache to determine:

1. language-independent SAFE, UNSAFE, and category regions;
2. language-specific regions and language-by-category interactions;
3. whether future Heretic direction construction should use English only, one
   non-duplicated multilingual sample, or category-conditioned language weights;
4. the smallest prompt subset that preserves the operative geometry.

No semantic judge participates in constructing or selecting geometry. Human or
model audits are allowed only before measurement to verify translation fidelity
or after the experiment to interpret already-frozen regions.

## First diagnostic corpus

The first operative map uses the frozen aligned five-language training split:

- 1,200 SAFE canonical prompts in EN, RU, ZH, ES, and FR;
- 1,200 UNSAFE canonical prompts in EN, RU, ZH, ES, and FR;
- 12,000 measured rows total (`1,200 x 2 directions x 5 languages`);
- exact canonical-ID, direction, category, and order alignment is mandatory.

The 200 SAFE plus 200 UNSAFE canonical holdout remains outside map fitting. The
map selects the language assignment for this 400-row operative test panel. Its
available aligned variants remain frozen and may be measured once as an audit
reference, but they do not turn the operative panel into `400 x 5` rows and are
not used to tune thresholds or language weights.

The operative run starts only when every language passes translation-integrity
gates. Development and loader smoke tests may use a complete EN/RU subset, but
cannot produce the final language policy. The ISO language code is `zh`, not
`ch`.

## Models and order

The identical frozen protocol is executed in this order:

1. `F:/AI/hf_originals/LLM/google__gemma-4-E4B-it` as the fastest primary run;
2. `F:/AI/hf_originals/LLM/mistralai__Ministral-3-3B-Instruct-2512-BF16`;
3. `F:/AI/hf_originals/LLM/Qwen__Qwen3.5-9B`.

Gemma is a schema, runtime, and mathematical smoke gate. A language policy is
not promoted as a general default until its conclusion is checked on at least
one different architecture. Each model receives its own immutable cache and
report; caches are never combined as if their hidden spaces were aligned.

## One-time measurement

The new command is a diagnostic subcommand of `hereticMOE`, not an Optuna trial
or a response classifier. It loads one unmodified BF16 model and writes one row
per input prompt with:

- canonical and row IDs, language, direction, and category metadata;
- residual-stream vectors for every layer at the existing Heretic decision
  position (the first generated-token position);
- norms and deterministic low-dimensional sketches needed for integrity checks;
- model, tokenizer, configuration, input, and output hashes.

Raw residuals are stored in a tensor container; prompt text is not duplicated in
the tensor cache and is never printed to console or included in reports. The
first implementation measures the exact position already used by Heretic
direction construction. Prompt-stage and later answer-stage checkpoints may be
added as separate named tensor stages later, but must not silently change the
meaning of the initial cache.

There is no per-trial regeneration. All subset, language-mixture, leave-one-out,
and temperature calculations operate on the immutable cache.

## Resident multi-GPU capture

`hereticMOE geometry-map run` accepts `--devices auto` or an explicit ordered
list such as `--devices 0,1`. The public command remains one controller in one
visible console. It starts one long-lived worker process per selected GPU; each
worker loads exactly one model copy and keeps it resident until the shared
capture finishes.

The controller creates a durable SQLite queue of global, non-overlapping row
ranges. A worker claims the next range only after finishing its current range,
so a faster GPU naturally processes more rows and heterogeneous 2-, 6-, or
8-GPU hosts do not wait on equal static shards. Each range is written atomically
as a separately hashed safetensors part. Global row numbers, rather than worker
identity, determine final order.

On restart, the controller verifies every completed part against its recorded
range, shape, and SHA-256. Missing, truncated, non-finite, or hash-mismatched
parts return to the queue; valid ranges are never recomputed. A worker failure
releases only that worker's claimed range. The final cache manifest is published
only after exact one-time coverage and canonical-order merge pass.

Each worker uses a bounded CPU thread pool and coarse queue ranges containing
multiple model batches. The implementation records rows/s, peak VRAM, claimed
ranges, and completed rows by GPU. Batch-size tuning and optional length buckets
may change execution order, but the final row index and residual tensor remain
in canonical order and byte-stable for the same measured values.

## Continuous heat map

For direction `d`, language `l`, category `c`, canonical item `i`, and layer
`k`, let the cached residual vector be `h[d,l,c,i,k]`.

The analysis keeps continuous vectors and subspaces. It does not convert layers
to binary active/inactive flags before statistics are complete.

For each canonical translation family, calculate a robust multilingual center
and each language residual. Each observation contributes fuzzy membership to a
region with a radial kernel:

```text
heat = support_rate
     * exp(-(robust_distance ** 2) / (2 * bandwidth ** 2))
     * cross_prompt_stability
     * cross_language_stability
```

The bandwidth is derived from the observed robust dispersion, not a hand-picked
radius. Distant observations are retained as colder branches. They are never
deleted merely for being far from the center. One canonical family has total
weight one so five translations cannot outvote five independent prompts.

## Factor and intersection maps

For every layer the report estimates the main and interaction components:

```text
direction + language + category
+ direction:language
+ category(direction)
+ language:category(direction)
+ residual
```

`category` is nested within direction because the SAFE and UNSAFE category
taxonomies are not matched factor levels. Treating them as a crossed
`direction x category` design would incorrectly attribute topic differences to
refusal geometry.

The corresponding fuzzy regions include:

- universal SAFE (white) core;
- universal UNSAFE core;
- universal refusal candidate: high UNSAFE heat, low SAFE heat, and low
  language/category dependence;
- language core shared by SAFE and UNSAFE in one language;
- category core shared across languages;
- language-specific UNSAFE and category-specific language branches;
- ambiguous overlap regions that are retained rather than forced into a label.

Robust intersections use quantiles or geometric means with bootstrap intervals,
not a literal minimum that one translation can erase. The output names regions
as candidates: cached activation geometry is associative evidence, not proof of
causal influence.

## Language and prompt selection from the cache

The analyzer compares these virtual corpora without another model pass:

- all aligned languages;
- EN only;
- leave-one-language-out variants;
- one language per canonical ID with no translated duplicates;
- globally weighted language mixtures;
- category-conditioned language mixtures;
- greedily reduced prompt subsets.

For each virtual corpus it reports:

- angle and subspace overlap relative to the full map;
- retained hot-region mass;
- lost language/category branches;
- reconstruction error per layer and category;
- bootstrap uncertainty;
- marginal contribution of every language and prompt family.

The preferred corpus is the smallest one whose loss is below the empirically
measured noise floor and which preserves every statistically supported branch.
If language is globally redundant but relevant to one category, selection uses
`language x category` weights rather than a single global percentage.

The resulting policy may therefore be EN-only, a non-duplicated rotating mix,
or a weighted mix such as EN/RU/ZH. Percentages are outputs, never assumptions.

## Holdout and search integration

After map fitting chooses a policy, materialize a 400-canonical holdout panel
(200 SAFE plus 200 UNSAFE) using that policy. Compare it with the complete
aligned holdout map once. A reduced panel passes only if it preserves the full
holdout geometry within the frozen tolerances and does not erase any category
branch.

This diagnostic does not start Optuna and does not change the existing search
objective. Search integration is a later gated change. When integrated, trials
must share frozen evaluation panels; giving every trial an unrelated random
panel would make Optuna scores incomparable.

## Artifacts

Each model produces a self-contained output directory:

```text
manifest.json
residuals.safetensors
row_index.jsonl
layer_statistics.json
factor_map.json
language_contributions.json
subset_candidates.json
holdout_validation.json
report.html
```

Console output contains only progress, counts, memory, hashes, and paths. HTML
contains plots and aggregate statistics but no prompt or response text.

## Frozen 3D trajectory package

The verified original-model cache is also the immutable coordinate reference
for an interactive three-dimensional search map. For every model layer, fit a
three-component PCA basis on the original cache only. Record the center,
components, explained variance, and basis hash. All later trial and finalist
measurements are projected with this frozen basis; adding a candidate must
never move the original points or refit axes.

The base map contains every measured original point. Public groups are neutral:

- group A: blue;
- group B: red;
- externally verified successful candidate outcome: green;
- unverified, missing-verdict, or borderline candidate outcome: yellow/grey.

Green is never inferred from the search proxy, keywords, SRG, or a journal
rank. It requires a row-ID-keyed external verdict. The search proxy remains a
separate numeric overlay.

The report provides independent filters for layer, language, group, category,
trial number/range, search phase, finalist, verdict state, and arrow type. Each
language and each search pass can be hidden completely. The view supports
mouse rotation, pan/zoom, point-size and opacity controls, animation through
trials, and an Original-only reset. It defaults to centroid arrows plus the
largest individual movements so the plot remains readable; all individual
arrows are an explicit opt-in.

Every candidate edit is applied to the original model, so the causal-looking
measurement shown by a solid arrow is strictly `Original -> Trial N`. A dashed
`Trial N -> Trial N+1` line is only an optimizer-search path between independent
edits and must be labelled as such. It is not presented as cumulative model
training.

The calibration package uses all original rows. Per-trial capture uses a frozen
stratified anchor panel (default 32 canonical rows, balanced across public
groups, languages, and categories). TOP-6 recheck uses a larger frozen panel
(default 200); Balanced and Max may use the complete operative test panel.
Anchors are selected by row metadata and seed. Their private prompt file is
separate from the publishable geometry package.

Store only projected float32 coordinates and full-space displacement summaries
for ordinary trials. Retain original full residuals for the anchor panel so the
report can publish PCA retained-shift ratios and warn when the three-dimensional
view hides material movement. Full per-trial residual tensors are not retained.

The package is append-only and text-free:

```text
geometry_3d/
  manifest.json
  projection_basis.safetensors
  base_points.f32
  base_index.json
  anchor_index.json
  anchor_reference.safetensors
  journal_trials.jsonl
  trial_points.f32
  trial_index.jsonl
  finalists/
    balanced.f32
    max.f32
  verdicts.jsonl
  report.html
private/
  anchors.jsonl
```

`journal_trials.jsonl` contains only trial numbers, phase, parameters, numeric
objectives, constraints, state, and hashes. An existing journal can therefore
populate the complete search timeline immediately. It cannot supply model-space
coordinates by itself. Old trials without captured residuals are explicitly
marked `not_captured`; their real points require deterministic replay of the
saved edit followed by an anchor residual pass. Coordinates are never invented
from objective values or response text.

New searches first capture the decision-position residuals of the same prompts
already used by normal trial evaluation. The model generation path exposes the
first generated-token hidden states to an optional sink, avoiding a second full
generation or a second large evaluation panel. A small frozen anchor panel is
still measured after scoring and before reset so different trials, scorer
versions, and search panels have an exactly comparable control trajectory.
Both hooks are disabled unless a frozen trajectory package is supplied. Capture
failure fails the trial artifact but does not silently publish a partial
coordinate record.

The self-contained HTML embeds typed binary coordinates and metadata without a
network dependency. It contains no prompts, answers, excerpts, or archive text.

## Hard gates

- exact aligned canonical coverage and matching metadata;
- no empty prompts, duplicate IDs, or cross-split overlap;
- original BF16 checkpoint only for the operative residual map;
- finite tensors and identical row counts across all layers;
- deterministic row ordering and content hashes;
- no prompt/response text in console or aggregate reports;
- no policy conclusion from a single architecture;
- no claim of causal influence without a later controlled intervention.

## Five-language prototype success criterion

The Gemma run is successful when the 12,000-row cache verifies, the analyzer can
reconstruct full, EN-only, non-duplicated multilingual, and every
leave-one-language-out map without another model invocation, and produces
text-free per-layer language, direction, category, and interaction statistics.
Only then is the identical protocol run on Ministral and Qwen. This criterion
describes the earlier five-language prototype; the following section defines
the later cross-model language-selection diagnostic.

## PolyGuard 17-language model-comparison diagnostic

The first language-selection benchmark uses the frozen local
`ToxicityPrompts/PolyGuardPrompts` parquet revision. It materializes only rows
whose source annotations agree unanimously on both prompt harmfulness and
response refusal behavior:

- SAFE: 482 canonical IDs labelled `unharmful` with a `compliance` response;
- UNSAFE: 238 canonical IDs labelled `harmful` with a `refusal` response;
- 720 canonical IDs x 17 exactly aligned languages = 12,240 measured rows.

The 17-language order is frozen in the dataset manifest, not in production
Python defaults. Every language must contain the same ordered canonical IDs
within SAFE and within UNSAFE. SAFE and UNSAFE may have different row counts.
The legacy balanced `rows_per_cell` interface remains valid for existing
five-language search artifacts; the diagnostic adds an auto-detected
per-direction count contract.

Each public index row retains a primary `category_id` for compatibility and a
text-free `category_ids` list containing every applicable PolyGuard S1-S14 tag.
SAFE rows without a risk tag use `benign`. Reports expose generated language
and category filters. Original SAFE points are blue and original UNSAFE points
are red; no report uses the neutral `group A/group B` copy for this diagnostic.

Capture is performed with one generated token and all decision-position
residual layers. Workers use every selected CUDA device through the existing
dynamic range queue, and model weights stay resident for the complete model
run. The protocol is run independently and sequentially for:

1. `Qwen__Qwen3.5-9B`;
2. `google__gemma-4-E4B-it`;
3. `mistralai__Ministral-3-3B-Instruct-2512-BF16`.

All three analyses use the same immutable dataset manifest and row order. A
fourth architecture is not required for the initial language decision.

Language selection uses paired canonical IDs and full-space residual metrics,
not visual proximity alone. For every layer and language it records the
SAFE-to-UNSAFE direction, the language offset relative to the pooled canonical
mean, category-conditioned interaction, and pairwise direction similarity.
Distances are normalized per layer, aggregated across the three models, and
clustered by deterministic k-medoids for each candidate `k` in 4 through 6.
The per-layer pair distance is `0.60 * refusal_direction_distance + 0.25 *
language_offset_distance + 0.15 * category_interaction_distance`; every term
is scaled to `[0, 1]` before aggregation and layers are weighted by their
SAFE/UNSAFE separation. Cross-model aggregation is the arithmetic mean of the
three model distance matrices.
English is a mandatory anchor. The remaining medoids are the languages with
the smallest total within-cluster distance; ties are resolved by frozen
language order. The recommended `k` maximizes mean silhouette across the three
models and the aggregate matrix, with ties resolved toward the smaller `k`.
`recommended_languages.json` records medoids, cluster members,
weights, per-model stability, and the evidence for excluding geometrically
redundant languages such as a possible EN/ES overlap.

The interactive package fits one pooled three-component basis per model layer.
It automatically lists every language found in the manifest, supports
language/category/SAFE/UNSAFE filters, and includes the pairwise heatmap and
cluster assignment beside the 3D view. Independent per-language PCA bases are
forbidden because their arbitrary rotations make language distances visually
incomparable.

Success requires three verified 12,240-row caches, three text-free HTML maps,
one cross-model k-medoids recommendation, exact source and cache hashes, and no
prompt or response text in console, public JSON, HTML, or logs.
