# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from time import sleep

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import QMCSampler, RandomSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

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

    def test_hybrid_alternates_one_shared_startup_then_uses_tpe(self) -> None:
        runner = OptimizationRunner(
            startup_design=StartupDesign.HYBRID,
            n_startup_trials=6,
            seed=3,
        )
        self.assertIsInstance(runner.initial_sampler, RandomSampler)
        self.assertIsInstance(runner.sobol_sampler, QMCSampler)

        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )
        runner.optimize_to(study, objective, target_trial_count=3)
        self.assertEqual(len(study.trials), 3)
        self.assertIs(study.sampler, runner.random_sampler)

        runner.optimize_to(study, objective, target_trial_count=6)
        self.assertEqual(len(study.trials), 6)
        self.assertIs(study.sampler, runner.sobol_sampler)

        runner.optimize_to(study, objective, target_trial_count=10)
        self.assertEqual(len(study.trials), 10)
        self.assertIs(study.sampler, runner.tpe_sampler)

    def test_parallel_worker_budgets_sum_without_restarting_startup(self) -> None:
        exploration = OptimizationRunner(
            startup_design=StartupDesign.HYBRID,
            n_startup_trials=4,
            seed=3,
        )
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=exploration.initial_sampler,
        )
        exploration.optimize_to(study, objective, target_trial_count=4)

        worker_a = OptimizationRunner(
            startup_design=StartupDesign.HYBRID,
            n_startup_trials=4,
            seed=3,
            parallel_workers=2,
        )
        worker_b = OptimizationRunner(
            startup_design=StartupDesign.HYBRID,
            n_startup_trials=4,
            seed=4,
            parallel_workers=2,
        )
        worker_a.optimize_budget(study, objective, trial_budget=3)
        worker_b.optimize_budget(study, objective, trial_budget=3)

        self.assertEqual(len(study.trials), 10)
        self.assertIs(study.sampler, worker_b.tpe_sampler)

    def test_parallel_workers_share_one_journal_without_duplicate_trials(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal_path = f"{temporary_directory}/parallel.log"
            storage = JournalStorage(
                JournalFileBackend(
                    journal_path,
                    lock_obj=JournalFileOpenLock(journal_path),
                )
            )
            exploration = OptimizationRunner(
                startup_design=StartupDesign.HYBRID,
                n_startup_trials=4,
                seed=3,
            )
            study = optuna.create_study(
                study_name="parallel",
                directions=["minimize", "minimize"],
                sampler=exploration.initial_sampler,
                storage=storage,
            )
            exploration.optimize_to(study, objective, target_trial_count=4)

            def run_worker(seed: int) -> None:
                worker_storage = JournalStorage(
                    JournalFileBackend(
                        journal_path,
                        lock_obj=JournalFileOpenLock(journal_path),
                    )
                )
                worker_study = optuna.load_study(
                    study_name="parallel",
                    storage=worker_storage,
                )
                worker = OptimizationRunner(
                    startup_design=StartupDesign.HYBRID,
                    n_startup_trials=4,
                    seed=seed,
                    parallel_workers=2,
                )

                def overlapping_objective(
                    trial: optuna.Trial,
                ) -> tuple[float, float]:
                    sleep(0.02)
                    return objective(trial)

                worker.optimize_budget(
                    worker_study,
                    overlapping_objective,
                    trial_budget=3,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run_worker, seed) for seed in (7, 11)]
                for future in futures:
                    future.result()

            reloaded = optuna.load_study(study_name="parallel", storage=storage)
            self.assertEqual(len(reloaded.trials), 10)
            self.assertEqual(
                [trial.number for trial in reloaded.trials],
                list(range(10)),
            )
            self.assertTrue(
                all(trial.state == TrialState.COMPLETE for trial in reloaded.trials)
            )


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
