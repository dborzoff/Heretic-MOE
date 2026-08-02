# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import QMCSampler, TPESampler

from heretic.config import StartupDesign
from heretic.search import OptimizationRunner

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


if __name__ == "__main__":
    unittest.main()
