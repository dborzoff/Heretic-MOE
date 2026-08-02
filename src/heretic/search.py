# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

from collections.abc import Callable, Sequence

from optuna import Trial
from optuna.samplers import BaseSampler, QMCSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import StartupDesign

Objective = Callable[[Trial], float | Sequence[float]]
StudyCallback = Callable[[Study, FrozenTrial], None]


class OptimizationRunner:
    """Run an Optuna study with a selectable exploration design.

    The legacy path remains a single multivariate TPE sampler with its built-in
    random startup. The Sobol path uses a scrambled low-discrepancy design for
    the requested prefix, then lets multivariate TPE learn from those completed
    observations. Samplers are retained across calls so interactive extension
    does not reset their in-process random state.
    """

    def __init__(
        self,
        *,
        startup_design: StartupDesign,
        n_startup_trials: int,
        seed: int | None,
    ) -> None:
        self.startup_design = startup_design
        self.n_startup_trials = n_startup_trials
        self.tpe_sampler = TPESampler(
            n_startup_trials=(
                n_startup_trials if startup_design == StartupDesign.RANDOM else 0
            ),
            n_ei_candidates=128,
            multivariate=True,
            seed=seed,
        )
        self.sobol_sampler = (
            QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=seed,
            )
            if startup_design == StartupDesign.SOBOL
            else None
        )

    @property
    def initial_sampler(self) -> BaseSampler:
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

        if self.sobol_sampler is not None and len(study.trials) < min(
            self.n_startup_trials, target_trial_count
        ):
            study.sampler = self.sobol_sampler
            startup_target = min(self.n_startup_trials, target_trial_count)
            study.optimize(
                objective,
                n_trials=startup_target - len(study.trials),
                callbacks=list(callbacks),
            )

        remaining_trials = target_trial_count - len(study.trials)
        if remaining_trials > 0:
            study.sampler = self.tpe_sampler
            study.optimize(
                objective,
                n_trials=remaining_trials,
                callbacks=list(callbacks),
            )
