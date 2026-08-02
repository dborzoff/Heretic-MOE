# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import QMCSampler, TPESampler

from heretic.config import StartupDesign
from heretic.search import OptimizationRunner, select_spread_points

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", -1.0, 1.0)
    return x * x, y * y


class OptimizationRunnerTests(unittest.TestCase):
    def test_random_design_matches_legacy_tpe_sequence(self) -> None:
        legacy_sampler = TPESampler(
            n_startup_trials=6,
            n_ei_candidates=128,
            multivariate=True,
            seed=3,
        )
        legacy_study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=legacy_sampler,
        )
        legacy_study.optimize(objective, n_trials=12)

        runner = OptimizationRunner(
            startup_design=StartupDesign.RANDOM,
            n_startup_trials=6,
            seed=3,
        )
        new_study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )
        runner.optimize_to(new_study, objective, target_trial_count=12)

        self.assertEqual(
            [trial.params for trial in new_study.trials],
            [trial.params for trial in legacy_study.trials],
        )

    def test_random_design_preserves_tpe_startup(self) -> None:
        runner = OptimizationRunner(
            startup_design=StartupDesign.RANDOM,
            n_startup_trials=6,
            seed=3,
        )
        self.assertIsInstance(runner.initial_sampler, TPESampler)
        self.assertIsNone(runner.sobol_sampler)

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )
        runner.optimize_to(study, objective, target_trial_count=10)
        self.assertEqual(len(study.trials), 10)
        self.assertIs(study.sampler, runner.tpe_sampler)

    def test_sobol_design_switches_to_multivariate_tpe(self) -> None:
        runner = OptimizationRunner(
            startup_design=StartupDesign.SOBOL,
            n_startup_trials=6,
            seed=3,
        )
        self.assertIsInstance(runner.initial_sampler, QMCSampler)

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )
        runner.optimize_to(study, objective, target_trial_count=10)
        self.assertEqual(len(study.trials), 10)
        self.assertIs(study.sampler, runner.tpe_sampler)

    def test_extension_reuses_sampler_and_exact_target(self) -> None:
        runner = OptimizationRunner(
            startup_design=StartupDesign.SOBOL,
            n_startup_trials=4,
            seed=3,
        )
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )
        runner.optimize_to(study, objective, target_trial_count=3)
        self.assertIs(study.sampler, runner.sobol_sampler)
        runner.optimize_to(study, objective, target_trial_count=7)
        self.assertEqual(len(study.trials), 7)
        self.assertIs(study.sampler, runner.tpe_sampler)


class SpreadSelectionTests(unittest.TestCase):
    def test_keeps_extremes_then_selects_interior_separation(self) -> None:
        front = [
            ((0.0, 1.0), 10),
            ((0.2, 0.7), 11),
            ((0.5, 0.5), 12),
            ((0.7, 0.2), 13),
            ((1.0, 0.0), 14),
        ]

        selected = select_spread_points(front, 3)

        self.assertEqual([trial_id for _, trial_id in selected[:2]], [10, 14])
        self.assertEqual(selected[2][1], 12)

    def test_count_larger_than_front_keeps_all_points(self) -> None:
        front = [((0.0, 1.0), 10), ((1.0, 0.0), 11)]
        self.assertEqual(select_spread_points(front, 10), front)


if __name__ == "__main__":
    unittest.main()
