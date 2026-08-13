# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import patch

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import QMCSampler, RandomSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from heretic.config import StartupDesign
from heretic.search import (
    ConstraintAwareTPESampler,
    OptimizationRunner,
    record_trial_constraints,
    select_spread_points,
)
from heretic.work_queue import TrialWorkQueue

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=ExperimentalWarning)


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", -1.0, 1.0)
    return x * x, y * y


class OptimizationRunnerTests(unittest.TestCase):
    @staticmethod
    def _single_task_queue(directory: str) -> TrialWorkQueue:
        queue = TrialWorkQueue(Path(directory) / "queue.sqlite3")
        queue.initialize(
            first_task_id=0,
            task_count=1,
            exploration_task_count=1,
            target_trial_count=1,
            tpe_concurrency=1,
            journal_base_trial_count=0,
            journal_base_complete_count=0,
            journal_base_size_bytes=0,
            journal_base_sha256=sha256().hexdigest(),
            queue_seed=20260812,
        )
        return queue

    def test_queue_requeues_pruned_trial_instead_of_marking_it_complete(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            queue = self._single_task_queue(temporary_directory)
            runner = OptimizationRunner(
                startup_design=StartupDesign.HYBRID,
                n_startup_trials=1,
                seed=3,
            )
            study = optuna.create_study(direction="minimize")

            def prune(_: optuna.Trial) -> float:
                raise optuna.TrialPruned("synthetic prune")

            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                runner.optimize_queue(
                    study,
                    prune,
                    queue_path=str(queue.path),
                    worker_id="gpu-0",
                )

            record = queue.task_records()[0]
            self.assertEqual(record.state, "pending")
            self.assertEqual(record.attempt, 1)
            self.assertEqual(study.trials[0].state, TrialState.PRUNED)

    def test_callback_failure_after_complete_does_not_retry_queue_permit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            queue = self._single_task_queue(temporary_directory)
            runner = OptimizationRunner(
                startup_design=StartupDesign.HYBRID,
                n_startup_trials=1,
                seed=3,
            )
            study = optuna.create_study(direction="minimize")

            def callback_error(_: optuna.Study, __: optuna.trial.FrozenTrial) -> None:
                raise RuntimeError("synthetic callback failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic callback failure"):
                runner.optimize_queue(
                    study,
                    lambda _: 1.0,
                    queue_path=str(queue.path),
                    worker_id="gpu-0",
                    callbacks=[callback_error],
                )

            record = queue.task_records()[0]
            self.assertEqual(record.state, "complete")
            self.assertEqual(record.trial_state, "COMPLETE")
            self.assertEqual(len(study.trials), 1)

    def test_queue_renews_and_stops_heartbeat_for_the_exact_claim(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            queue = self._single_task_queue(temporary_directory)
            runner = OptimizationRunner(
                startup_design=StartupDesign.HYBRID,
                n_startup_trials=1,
                seed=3,
            )
            study = optuna.create_study(direction="minimize")
            heartbeat_claims: list[tuple[int, int, str]] = []
            original_heartbeat = TrialWorkQueue.heartbeat

            def record_heartbeat(
                active_queue: TrialWorkQueue,
                item,
                *,
                worker_id: str,
            ) -> None:
                heartbeat_claims.append((item.task_id, item.attempt, worker_id))
                original_heartbeat(active_queue, item, worker_id=worker_id)

            with patch.object(TrialWorkQueue, "heartbeat", record_heartbeat):
                runner.optimize_queue(
                    study,
                    lambda _: sleep(0.04) or 1.0,
                    queue_path=str(queue.path),
                    worker_id="gpu-0",
                    heartbeat_interval_seconds=0.005,
                )
                completed_count = len(heartbeat_claims)
                sleep(0.02)

            self.assertGreaterEqual(completed_count, 1)
            self.assertEqual(len(heartbeat_claims), completed_count)
            self.assertEqual(
                set(heartbeat_claims),
                {(0, queue.task_records()[0].attempt, "gpu-0")},
            )

            runner.optimize_queue(
                study,
                lambda _: 2.0,
                queue_path=str(queue.path),
                worker_id="gpu-1",
            )
            self.assertEqual(len(study.trials), 1)

    def test_queue_exploration_is_stable_across_worker_seeds(self) -> None:
        def run(worker_seed: int) -> list[dict[str, object]]:
            with TemporaryDirectory() as temporary_directory:
                queue = TrialWorkQueue(Path(temporary_directory) / "queue.sqlite3")
                queue.initialize(
                    first_task_id=0,
                    task_count=2,
                    exploration_task_count=2,
                    target_trial_count=2,
                    tpe_concurrency=1,
                    journal_base_trial_count=0,
                    journal_base_complete_count=0,
                    journal_base_size_bytes=0,
                    journal_base_sha256=sha256().hexdigest(),
                    queue_seed=20260812,
                )
                runner = OptimizationRunner(
                    startup_design=StartupDesign.HYBRID,
                    n_startup_trials=2,
                    seed=worker_seed,
                )
                study = optuna.create_study(direction="minimize")

                def sampled(trial: optuna.Trial) -> float:
                    trial.suggest_categorical("scope", ["global", "per-layer"])
                    return trial.suggest_float("x", -1.0, 1.0)

                runner.optimize_queue(
                    study,
                    sampled,
                    queue_path=str(queue.path),
                    worker_id=f"gpu-{worker_seed}",
                )
                return [trial.params for trial in study.trials]

        self.assertEqual(run(100), run(101))

    def test_constraint_record_is_visible_to_every_sampler_before_completion(
        self,
    ) -> None:
        study = optuna.create_study(direction="minimize", sampler=RandomSampler(seed=3))
        trial = study.ask()

        record_trial_constraints(trial, (0.125,))

        frozen = study._storage.get_trial(trial._trial_id)
        self.assertEqual(frozen.user_attrs["constraints"], [0.125])
        self.assertEqual(frozen.system_attrs["constraints"], [0.125])
        study.tell(trial, 1.0)
        self.assertEqual(study.trials[0].system_attrs["constraints"], [0.125])

    def test_tpe_hydrates_legacy_exploration_constraints_from_user_attrs(self) -> None:
        study = optuna.create_study(direction="minimize", sampler=RandomSampler(seed=3))
        exploration = study.ask()
        exploration.set_user_attr("constraints", [0.125])
        exploration.suggest_float("x", 0.0, 1.0)
        study.tell(exploration, 1.0)
        study.sampler = ConstraintAwareTPESampler(
            n_startup_trials=0,
            multivariate=True,
            constant_liar=True,
            constraints_func=lambda trial: trial.user_attrs["constraints"],
            seed=4,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            next_trial = study.ask()
            next_trial.suggest_float("x", 0.0, 1.0)

        self.assertFalse(
            any(
                "does not have constraint values" in str(item.message)
                for item in caught
            )
        )
        self.assertEqual(study.trials[0].system_attrs["constraints"], [0.125])

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

    def test_grouped_tpe_supports_conditional_component_space(self) -> None:
        runner = OptimizationRunner(
            startup_design=StartupDesign.RANDOM,
            n_startup_trials=4,
            seed=17,
            constraint_count=1,
            tpe_group=True,
        )
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=runner.initial_sampler,
        )

        def conditional_objective(trial: optuna.Trial) -> tuple[float, float]:
            enabled = trial.suggest_categorical("component.enabled", [True, False])
            x = trial.suggest_float("component.x", -1.0, 1.0) if enabled else 0.0
            violation = abs(x) - 0.75
            trial.set_user_attr("constraints", [violation])
            return x * x, abs(x)

        runner.optimize_to(study, conditional_objective, target_trial_count=12)

        self.assertEqual(len(study.trials), 12)
        self.assertTrue(
            all("constraints" in trial.user_attrs for trial in study.trials)
        )
        self.assertTrue(
            any("component.x" not in trial.params for trial in study.trials)
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
