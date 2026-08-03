# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from collections.abc import Callable, Sequence

from optuna import Trial
from optuna.samplers import BaseSampler, QMCSampler, RandomSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import StartupDesign

Objective = Callable[[Trial], float | Sequence[float]]
StudyCallback = Callable[[Study, FrozenTrial], None]


def select_spread_points(
    front: Sequence[tuple[Sequence[float], int]],
    count: int,
) -> list[tuple[Sequence[float], int]]:
    """Select objective-space extremes, then greedily maximize separation."""

    if count <= 0 or not front:
        return []
    if count >= len(front):
        return list(front)

    objective_count = len(front[0][0])
    if objective_count == 0:
        return list(front[:count])
    if any(len(values) != objective_count for values, _ in front):
        raise ValueError("All Pareto points must have the same objective count")

    lows = [min(values[i] for values, _ in front) for i in range(objective_count)]
    highs = [max(values[i] for values, _ in front) for i in range(objective_count)]

    def normalized(values: Sequence[float]) -> tuple[float, ...]:
        return tuple(
            0.0 if highs[i] == lows[i] else (value - lows[i]) / (highs[i] - lows[i])
            for i, value in enumerate(values)
        )

    coordinates = [normalized(values) for values, _ in front]
    selected_indices: list[int] = []

    for objective_index in range(objective_count):
        candidate_index = min(
            range(len(front)),
            key=lambda index: (
                front[index][0][objective_index],
                tuple(front[index][0]),
                front[index][1],
            ),
        )
        if candidate_index not in selected_indices:
            selected_indices.append(candidate_index)
        if len(selected_indices) == count:
            break

    def squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right))

    while len(selected_indices) < count:
        remaining = [
            index for index in range(len(front)) if index not in selected_indices
        ]
        next_index = max(
            remaining,
            key=lambda index: (
                min(
                    squared_distance(coordinates[index], coordinates[selected])
                    for selected in selected_indices
                ),
                tuple(-value for value in front[index][0]),
                -front[index][1],
            ),
        )
        selected_indices.append(next_index)

    return [front[index] for index in selected_indices]


class OptimizationRunner:
    """Run an Optuna study with a selectable exploration design.

    The legacy path remains a single multivariate TPE sampler with its built-in
    random startup. The Sobol path uses a scrambled low-discrepancy design for
    the requested prefix. The hybrid path alternates explicit Random and Sobol
    trials in one study, which keeps an interrupted prefix balanced while still
    giving TPE one combined exploration history. Samplers are retained across
    calls so interactive extension does not reset their in-process random state.
    """

    def __init__(
        self,
        *,
        startup_design: StartupDesign,
        n_startup_trials: int,
        seed: int | None,
        parallel_workers: int = 1,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
        self.startup_design = startup_design
        self.n_startup_trials = n_startup_trials
        self.tpe_sampler = TPESampler(
            n_startup_trials=(
                n_startup_trials if startup_design == StartupDesign.RANDOM else 0
            ),
            n_ei_candidates=128,
            multivariate=True,
            constant_liar=parallel_workers > 1,
            seed=seed,
        )
        self.random_sampler = (
            RandomSampler(seed=seed)
            if startup_design == StartupDesign.HYBRID
            else None
        )
        self.sobol_sampler = (
            QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=seed,
            )
            if startup_design in (StartupDesign.SOBOL, StartupDesign.HYBRID)
            else None
        )

    @property
    def initial_sampler(self) -> BaseSampler:
        if self.random_sampler is not None:
            return self.random_sampler
        if self.sobol_sampler is not None:
            return self.sobol_sampler
        return self.tpe_sampler

    def optimize_to(
        self,
        study: Study,
        objective: Objective,
        *,
        target_trial_count: int,
        callbacks: Sequence[StudyCallback] = (),
    ) -> None:
        """Optimize until the study contains ``target_trial_count`` trials."""

        if target_trial_count < len(study.trials):
            raise ValueError("target_trial_count is below the existing trial count")

        startup_target = min(self.n_startup_trials, target_trial_count)
        if self.random_sampler is not None and len(study.trials) < startup_target:
            assert self.sobol_sampler is not None
            while len(study.trials) < startup_target:
                study.sampler = (
                    self.random_sampler
                    if len(study.trials) % 2 == 0
                    else self.sobol_sampler
                )
                study.optimize(
                    objective,
                    n_trials=1,
                    callbacks=list(callbacks),
                )
        elif self.sobol_sampler is not None and len(study.trials) < startup_target:
            study.sampler = self.sobol_sampler
            study.optimize(
                objective,
                n_trials=startup_target - len(study.trials),
                callbacks=list(callbacks),
            )

        if self.sobol_sampler is not None and len(study.trials) < min(
            self.n_startup_trials, target_trial_count
        ):
            raise RuntimeError("Startup sampler did not reach its requested target")

        remaining_trials = target_trial_count - len(study.trials)
        if remaining_trials > 0:
            study.sampler = self.tpe_sampler
            study.optimize(
                objective,
                n_trials=remaining_trials,
                callbacks=list(callbacks),
            )

    def optimize_budget(
        self,
        study: Study,
        objective: Objective,
        *,
        trial_budget: int,
        callbacks: Sequence[StudyCallback] = (),
    ) -> None:
        """Run this worker's exact TPE budget against a shared study."""

        if trial_budget <= 0:
            raise ValueError("trial_budget must be positive")
        if len(study.trials) < self.n_startup_trials:
            raise ValueError(
                "Parallel worker budgets require the exploration prefix to be complete"
            )
        study.sampler = self.tpe_sampler
        study.optimize(
            objective,
            n_trials=trial_budget,
            callbacks=list(callbacks),
        )
