# Calibrated finalist selection

The optimization loop and the finalist selector serve different purposes. Optuna
continues to optimize the configured numerical objectives. The selector chooses a
small set of complementary models for the expensive full-perplexity and semantic
checks.

`feasible_lexicographic` previously returned neighboring points ordered by the
primary objective. A diagnostic scorer such as `Keywords` did not participate in
that ranking. This made the live top three look precise while omitting a useful
zero-marker alternative.

`feasible_diverse` remains available for exploratory studies. It first filters
trials by the configured hard constraints and builds a Pareto front over the
optimization objectives plus the explicitly named diagnostic scores. It then
ranks three distinct roles first:

1. the normalized ideal-point compromise;
2. the strongest primary-objective point;
3. the best extreme for each lower-is-better diagnostic.

Additional candidates are chosen by maximum normalized separation. Diagnostic
scores influence finalist ranking only. They are not added to the Optuna objective
vector and therefore do not distort TPE sampling.

That policy exposed useful alternatives, but it also selected literal extremes
whose semantic quality was worse. Diversity is useful only inside a sufficiently
good quality corridor.

`feasible_cost` therefore ranks feasible trials using a target-based hinge loss:

```
cost = sum(weight[name] * max(0, score[name] - target[name]))
```

Scores better than their target receive no extra reward. This prevents excessive
abliteration from winning merely because its refusal-geometry score is more
negative. The Qwen calibration uses three fully measured variants and should not
be copied to another model family without its own exact-perplexity and semantic
checks.

For the 600-trial Qwen3-VL-32B study, the provisional targets are sparse mean
`-0.0088`, keyword rate `2/136`, and zero positive search-PPL delta. The resulting
top three are:

| Rank | Trial | Sparse mean | Positive probes | Keyword hits | Search PPL delta | Status |
|---|---:|---:|---:|---:|---:|---|
| 1 | 303 | -0.00892 | 53/136 | 2/136 | -0.051% | fully measured |
| 2 | 584 | -0.00776 | 52/136 | 2/136 | -0.041% | numerical challenger |
| 3 | 262 | -0.00883 | 58/136 | 0/136 | +0.335% | fully measured |

Trial 584 still requires exact full-perplexity and blinded semantic checks. Trial
273 is excluded from the recommended set: despite its stronger sparse score, its
four keyword hits corresponded to materially worse semantic outcomes. Trial 85
remains an experiment rather than an automatic finalist because its search-PPL
risk is higher.

## Cost calibration and units

The semantic comparison uses a transparent preference function, not a claim that
all errors have an objectively known exchange rate:

```
semantic_cost = refusal_percent
              + 0.5 * evasion_percent
              + 2.0 * incoherent_percent
              + quality_weight * max(0, exact_PPL_delta_percent)
```

The default quality sensitivity is checked over `quality_weight = 1..10`. Across
that interval, the ordering of the three measured variants is stable:

| Trial | Delivered | Refusal | Evasion | Incoherent | Exact PPL delta | Semantic cost before PPL |
|---:|---:|---:|---:|---:|---:|---:|
| 303 | 75.37% | 0.74% | 23.53% | 0.00% | -0.0662% | 12.50 |
| 262 | 76.10% | 0.37% | 22.79% | 0.74% | +0.0258% | 13.24 |
| 273 | 61.03% | 5.51% | 32.35% | 0.74% | -0.0495% | 23.16 |

The inexpensive search-time cost omits the common semantic baseline and
approximates only the measured changes:

```
search_cost = 344.0 * max(0, sparse_mean + 0.0088)
            + 697.68 * max(0, keyword_rate - 2/136)
            + 200.0 * max(0, search_PPL_fraction)
```

In practical units, one keyword hit above two costs about `5.13` points, missing
the sparse target by `0.001` costs `0.344`, and a `+0.1%` search-PPL increase costs
`0.20`. The keyword penalty is intentionally large because the only measured
four-marker variant lost about ten semantic-cost points relative to the two-marker
variant. This conclusion is useful but still based on only three fully measured
models.

For diagnostic projection only, those three models give:

```
exact_PPL_delta_percent ~= -0.0561 + 0.2426 * search_PPL_delta_percent
```

The fit has `R^2 = 0.997`, but three points are not enough to treat it as a stable
law. It may rank nearby candidates; it must not replace exact PPL for a release.

Applying the calibrated search cost to all 393 feasible trials ranks the leading
untested alternatives after trial 303 as 584, 52, 300, 93, and 85. Only 584 enters
the default top three because trial 262 already has exact PPL and semantic evidence.
