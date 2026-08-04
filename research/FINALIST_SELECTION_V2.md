# Diversified finalist selection

The optimization loop and the finalist selector serve different purposes. Optuna
continues to optimize the configured numerical objectives. The selector chooses a
small set of complementary models for the expensive full-perplexity and semantic
checks.

`feasible_lexicographic` previously returned neighboring points ordered by the
primary objective. A diagnostic scorer such as `Keywords` did not participate in
that ranking. This made the live top three look precise while omitting a useful
zero-marker alternative.

`feasible_diverse` first filters trials by the configured hard constraints and
builds a Pareto front over the optimization objectives plus the explicitly named
diagnostic scores. It then ranks three distinct roles first:

1. the normalized ideal-point compromise;
2. the strongest primary-objective point;
3. the best extreme for each lower-is-better diagnostic.

Additional candidates are chosen by maximum normalized separation. Diagnostic
scores influence finalist ranking only. They are not added to the Optuna objective
vector and therefore do not distort TPE sampling.

For the 600-trial Qwen3-VL-32B study, using `Keywords` as the diagnostic produces:

| Role | Trial | Sparse mean | Positive probes | Keyword hits | Search PPL delta |
|---|---:|---:|---:|---:|---:|
| balanced compromise | 303 | -0.00892 | 53/136 | 2/136 | -0.051% |
| primary extreme | 273 | -0.00998 | 53/136 | 4/136 | +0.039% |
| zero-marker extreme | 85 | -0.00909 | 48/136 | 0/136 | +0.439% |

Trial 85 is a new numerical challenger, not a validated winner. It still requires
the same full-perplexity and blinded semantic checks as every exported finalist.
The already tested trial 262 remains evidence that the zero-marker role is useful,
but its result is not used as a hard-coded selection rule.
