# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest

from optuna.study import StudyDirection
from optuna.trial import TrialState, create_trial

from heretic.config import SelectionPolicy
from heretic.trial_selection import candidate_trials, is_feasible


def make_trial(number: int, values: tuple[float, float], constraints):
    trial = create_trial(
        state=TrialState.COMPLETE,
        values=values,
        user_attrs={"constraints": constraints},
    )
    # create_trial uses an unset number outside storage; assign a deterministic
    # number for pure ordering tests.
    trial._number = number
    return trial


class TrialSelectionTests(unittest.TestCase):
    def test_filters_infeasible_zero_refusal_trial(self) -> None:
        trials = [
            make_trial(1, (0.0, 0.010), [0.005]),
            make_trial(2, (0.01, 0.000), [-0.005]),
            make_trial(3, (0.02, -0.001), [-0.006]),
        ]

        selected = candidate_trials(
            trials,
            [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            policy=SelectionPolicy.FEASIBLE_LEXICOGRAPHIC,
            constraint_count=1,
        )

        self.assertEqual([trial.number for trial in selected], [2, 3])
        self.assertTrue(is_feasible(selected[0], 1))

    def test_respects_maximize_direction(self) -> None:
        trials = [
            make_trial(1, (0.6, 0.1), []),
            make_trial(2, (0.8, 0.1), []),
        ]

        selected = candidate_trials(
            trials,
            [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE],
            policy=SelectionPolicy.PARETO,
            constraint_count=0,
        )

        self.assertEqual([trial.number for trial in selected], [2])

    def test_least_violation_is_first_when_no_trial_is_feasible(self) -> None:
        trials = [
            make_trial(1, (0.0, 0.2), [0.2]),
            make_trial(2, (0.1, 0.1), [0.05]),
        ]

        selected = candidate_trials(
            trials,
            [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            policy=SelectionPolicy.FEASIBLE_LEXICOGRAPHIC,
            constraint_count=1,
        )

        self.assertEqual(selected[0].number, 2)


if __name__ == "__main__":
    unittest.main()
