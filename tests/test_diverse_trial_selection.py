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


if __name__ == "__main__":
    unittest.main()
