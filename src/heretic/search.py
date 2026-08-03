# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from collections.abc import Callable, Sequence

from optuna import Trial
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler, QMCSampler, RandomSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import StartupDesign

Objective = Callable[[Trial], float | Sequence[float]]
StudyCallback = Callable[[Study, FrozenTrial], None]


class StratifiedQMCSampler(QMCSampler):
    """Sobol sampler with deterministic coverage for known categorical axes.

    Optuna's QMC sampler intentionally falls back to independent RandomSampler
    draws for categorical distributions.  Heretic has one always-present
    categorical axis, ``direction_scope``.  Alternating its choices by trial
    number gives exact 50/50 coverage while leaving every continuous parameter
    to scrambled Sobol.  Unknown future categorical parameters still delegate
    to Optuna and retain its warning.
    """

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution,
    ):
        if (
            param_name == "direction_scope"
            and isinstance(param_distribution, CategoricalDistribution)
        ):
            choices = param_distribution.choices
            return choices[trial.number % len(choices)]
        return super().sample_independent(
            study,
            trial,
            param_name,
            param_distribution,
        )


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
        constraint_count: int = 0,
        tpe_group: bool = False,
    ) -> None:
        if parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
        if constraint_count < 0:
            raise ValueError("constraint_count cannot be negative")
        self.startup_design = startup_design
        self.n_startup_trials = n_startup_trials
        self.constraint_count = constraint_count

        def constraints_func(trial: FrozenTrial) -> Sequence[float]:
            values = trial.user_attrs.get("constraints")
            if values is None:
                # A study resumed after adding constraints must not treat legacy
                # trials with unknown feasibility as valid evidence.
                return [float("inf")] * constraint_count
            if not isinstance(values, (list, tuple)) or len(values) != constraint_count:
                return [float("inf")] * constraint_count
            return [float(value) for value in values]

        self.tpe_sampler = TPESampler(
            n_startup_trials=(
                n_startup_trials if startup_design == StartupDesign.RANDOM else 0
            ),
            n_ei_candidates=128,
            multivariate=True,
            group=tpe_group,
            constant_liar=parallel_workers > 1,
            constraints_func=(constraints_func if constraint_count else None),
            seed=seed,
        )
        self.random_sampler = (
            RandomSampler(seed=seed)
            if startup_design == StartupDesign.HYBRID
            else None
        )
        self.sobol_sampler = (
            StratifiedQMCSampler(
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
