import unittest

import optuna

from heretic.config import SelectionPolicy
from heretic.trial_selection import candidate_trials


class DiverseTrialSelectionTest(unittest.TestCase):
    def test_ranks_balanced_primary_and_diagnostic_roles_first(self):
        study = optuna.create_study(directions=["minimize", "minimize"])
        points = [
            (0.2, 0.2, 0.2),
            (0.0, 0.5, 0.4),
            (0.1, 0.4, 0.0),
            (0.9, 0.0, 0.9),
        ]
        for primary, quality, diagnostic in points:
            trial = study.ask()
            trial.set_user_attr("constraints", [-0.1])
            trial.set_user_attr(
                "scores",
                [{"name": "Keywords", "score": {"value": diagnostic}}],
            )
            study.tell(trial, [primary, quality])

        ranked = candidate_trials(
            study.trials,
            study.directions,
            policy=SelectionPolicy.FEASIBLE_DIVERSE,
            constraint_count=1,
            primary_objective_index=0,
            diagnostic_names=("Keywords",),
        )

        self.assertEqual([trial.number for trial in ranked[:3]], [0, 1, 2])

    def test_cost_policy_penalizes_only_excess_above_targets(self):
        study = optuna.create_study(directions=["minimize", "minimize"])
        points = [
            (-0.0089, -0.0005, 2 / 136),
            (-0.0090, 0.0004, 4 / 136),
            (-0.0088, 0.0034, 0.0),
            (-0.0078, -0.0004, 2 / 136),
        ]
        for sparse, perplexity, keywords in points:
            trial = study.ask()
            trial.set_user_attr("constraints", [-0.001])
            trial.set_user_attr(
                "scores",
                [
                    {
                        "name": "Sparse refusal geometry",
                        "score": {"value": sparse},
                    },
                    {"name": "Keywords", "score": {"value": keywords}},
                    {
                        "name": "Perplexity increase",
                        "score": {"value": perplexity},
                    },
                ],
            )
            study.tell(trial, [sparse, perplexity])

        ranked = candidate_trials(
            study.trials,
            study.directions,
            policy=SelectionPolicy.FEASIBLE_COST,
            constraint_count=1,
            score_targets={
                "Sparse refusal geometry": -0.0088,
                "Keywords": 2 / 136,
                "Perplexity increase": 0.0,
            },
            score_weights={
                "Sparse refusal geometry": 344.0,
                "Keywords": 697.68,
                "Perplexity increase": 200.0,
            },
        )

        self.assertEqual([trial.number for trial in ranked], [0, 3, 2, 1])


if __name__ == "__main__":
    unittest.main()
