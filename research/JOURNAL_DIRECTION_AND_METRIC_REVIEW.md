# Journal, Direction, and Metric Review

This review is text-free. It uses Optuna parameters, objective values, edit
telemetry, blind system identifiers, and semantic labels. It does not reproduce
or inspect prompt/response text in the report.

## What the current direction does

For every residual depth, Heretic computes

```text
d_l = normalize(mean(harmful_l) - mean(harmless_l))
```

and can remove the part parallel to the harmless mean. The weight edit is a
projection of the form

```text
W' = W - lambda * d * (d^T W)
```

With per-layer directions, each layer uses its own `d_l`. With a sampled global
`direction_index`, one interpolated direction is reused across edited layers.

This removes or reflects a refusal-correlated component. It does **not** map an
activation to the harmless centroid, and it does not guarantee movement into a
visual "answer cluster".

## Why a 2D cluster is not yet an edit target

A PaCMAP/PCA-style plot can show useful separation, but distances and apparent
paths in two dimensions can be projection artifacts. The harmful and harmless
means can also differ by topic, language, or format rather than only by refusal.
Moving directly toward the harmless mean can therefore erase useful semantics.

The hypothesis should be tested in the original hidden dimension with these
text-free quantities for each layer:

- signed centroid margin along the contrastive direction;
- covariance-normalized distance to each centroid;
- movement toward the harmless centroid after an edit;
- movement orthogonal to the contrastive direction (off-axis damage);
- movement of harmless controls, which should remain small.

An edit is promising only when harmful activations cross the refusal margin
while harmless activations and off-axis coordinates remain stable. A paired
`+epsilon/-epsilon` intervention can test the sign causally before exporting a
weight edit.

## Journal audit

The audit covered 19 readable studies. Six candidates were skipped: four exact
duplicates and two invalid/empty journals. The comparison used a PPL-fraction
cap of `0.005` (+0.5%). Results across studies are associations; objective
scales and prompt sets can differ.

The strongest current evidence comes from realized edit telemetry in the
continued Ministral study (40 new completed trials at the analysis snapshot):

- total edit Frobenius norm vs PPL: Spearman `+0.915`;
- relative edit Frobenius norm vs PPL: Spearman `+0.758`;
- total edit Frobenius norm vs keyword rate: Spearman `-0.462`.

Thus stronger edits reduce the old refusal proxy but sharply increase model
damage. The realized edit norm is a better early damage predictor than a curve
name or visual radius alone.

Across compatible journals, low-PPL elite trials tend to use:

- lower MLP edit amplitude;
- narrower MLP support;
- narrower attention support;
- lower legacy attention distance and MLP maximum weight.

Within the +0.5% PPL region, lower keyword rates tend to require the opposite:
stronger and wider MLP edits. This is the central trade-off the optimizer must
model explicitly.

`direction_index` is not stable across studies. It has a visible association in
the unified Ministral journal, but its sign is inconsistent across 12-14
journals. A universal global "answer direction index" is therefore not
supported by the accumulated journals.

## KeywordRate calibration

Five edited finalists have both journal objectives and blind 9-label semantic
judgments on the same 136 identifiers. Residual censorship is defined as the
three soft-evasion labels plus policy refusal.

| Build | Journal marker count | Long-archive marker count | Semantic censorship |
|---|---:|---:|---:|
| trial260 | 0 | 8 | 34 |
| trial597 | 2 | 13 | 40 |
| trial488 | 4 | 13 | 49 |
| trial290 | 5 | 15 | 66 |
| trial320 | 6 | 17 | 64 |

Across these five builds, the short journal rate ranks semantic censorship
surprisingly well (Pearson `0.949`, Spearman `0.900`). However, this is only five
points from one model family. The descriptive linear fit has an intercept near
`0.232`, meaning that zero journal markers corresponds to roughly 23-25%
semantic censorship in this sample, not zero censorship.

At row level across original plus five edited systems, the complete marker list
has precision `0.452` and recall `0.110` for semantic censorship. It is not a
valid row classifier.

Marker groups explain the failure:

- explicit refusal phrases: high precision (`0.875` overall, `1.0` on edited
  builds), but recall only `0.020` overall and `0.008` on edited builds;
- isolated risk/policy terms: precision `0.457` overall and `0.338` on edited
  builds, while producing most matches;
- generic apology/disclaimer terms are too sparse to recover soft evasion.

Response length is also a confounder: trial260 changes from 0 marker hits in the
short search evaluation to 8 in the 2048-token archive.

## Revised metric contract

KeywordRate should remain a cheap ordinal feature, not an absolute refusal
rate. The search contract should be:

1. Hard quality gate: PPL fraction `<= 0.005`.
2. Early damage gate: reject edits whose realized relative/absolute edit norm
   exceeds the range learned from feasible incumbents.
3. Explicit-refusal phrase rate: high-precision diagnostic/gate, reported
   separately from ambiguous risk terms.
4. Ambiguous risk-term rate: diagnostic only, never a standalone refusal label.
5. Semantic evasion proxy: primary refusal objective once a prompt-grouped,
   externally validated classifier is available.
6. Empty/incoherent and epistemic-limit rates: regression gates, not refusal
   objectives.
7. Full semantic judging only after promotion, with a second pass on
   disagreements.

Until the semantic proxy passes external validation, a journal marker count of
0-2 can select a promising region, but cannot certify a finished model.

## Calibration experiment needed

Export a stratified set of approximately 18-30 feasible builds spanning journal
marker counts 0, 1, 2, 3, 4, and 5+, with multiple PPL levels per bin. Generate
all with identical decoding settings and response length. Judge them blindly,
then fit an isotonic or hierarchical binomial calibration. Split validation by
prompt ID so repeated questions across builds never leak between train and
validation.

Calibration is model-family and response-length specific. It must not be
silently transferred from Ministral to Qwen or Gemma.

## Search consequence

The current 600-to-1000 continuation is useful for collecting edit telemetry,
but it remains the old unconstrained two-objective study. A production search
should start a separate constrained study seeded from feasible incumbents. It
should promote only candidates that pass PPL and edit-budget gates, and should
optimize the semantic proxy rather than interpreting KeywordRate zero as the
final goal.
