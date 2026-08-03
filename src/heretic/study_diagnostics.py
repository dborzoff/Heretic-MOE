# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


def _completed_trials(study: Study) -> list[FrozenTrial]:
    return [trial for trial in study.trials if trial.state == TrialState.COMPLETE]


def build_parameter_importance_report(
    study: Study,
    objective_names: Sequence[str],
    *,
    seed: int | None,
) -> dict[str, Any]:
    """Build a text-free fANOVA report for every optimization objective.

    This is deliberately diagnostic only: the result is not fed back into the
    sampler. Each objective is analysed separately because Optuna requires an
    explicit target for multi-objective studies.
    """

    completed = _completed_trials(study)
    report: dict[str, Any] = {
        "schema_version": 1,
        "study_name": study.study_name,
        "completed_trials": len(completed),
        "pareto_trials": len(study.best_trials) if completed else 0,
        "objectives": [],
    }

    if not completed:
        report["status"] = "insufficient_trials"
        return report

    objective_count = len(completed[0].values or ())
    if len(objective_names) != objective_count:
        raise ValueError(
            "objective_names must match the number of values stored in each trial"
        )

    for objective_index, objective_name in enumerate(objective_names):
        objective_report: dict[str, Any] = {
            "index": objective_index,
            "name": objective_name,
        }
        try:
            importances = get_param_importances(
                study,
                evaluator=FanovaImportanceEvaluator(seed=seed),
                target=lambda trial, index=objective_index: trial.values[index],
            )
            objective_report["status"] = "ok"
            objective_report["importances"] = importances
        except (RuntimeError, ValueError) as error:
            # A young or highly conditional study can be impossible to analyse.
            # Diagnostics must never abort or alter the optimization itself.
            objective_report["status"] = "unavailable"
            objective_report["error_type"] = type(error).__name__
            objective_report["error"] = str(error)
        report["objectives"].append(objective_report)

    report["status"] = (
        "ok"
        if all(item["status"] == "ok" for item in report["objectives"])
        else "partial"
    )
    return report


def write_parameter_importance_report(
    study: Study,
    output_path: str | Path,
    objective_names: Sequence[str],
    *,
    seed: int | None,
) -> dict[str, Any]:
    """Atomically write the current fANOVA report and return it."""

    report = build_parameter_importance_report(
        study,
        objective_names,
        seed=seed,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return report


class ParameterImportanceReporter:
    """Optuna callback that periodically snapshots parameter importance."""

    def __init__(
        self,
        *,
        interval: int,
        output_path: str | Path,
        objective_names: Sequence[str],
        seed: int | None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.output_path = Path(output_path)
        self.objective_names = tuple(objective_names)
        self.seed = seed
        self._last_completed_count = 0

    def __call__(self, study: Study, _trial: FrozenTrial) -> None:
        completed_count = len(_completed_trials(study))
        if completed_count < self.interval:
            return
        if completed_count % self.interval != 0:
            return
        if completed_count == self._last_completed_count:
            return

        write_parameter_importance_report(
            study,
            self.output_path,
            self.objective_names,
            seed=self.seed,
        )
        self._last_completed_count = completed_count


def make_parameter_importance_callbacks(
    *,
    interval: int,
    checkpoint_path: str | Path,
    objective_names: Sequence[str],
    seed: int | None,
) -> list[ParameterImportanceReporter]:
    """Return callbacks without special-casing disabled diagnostics at call sites."""

    if interval <= 0:
        return []
    return [
        ParameterImportanceReporter(
            interval=interval,
            output_path=f"{checkpoint_path}.importance.json",
            objective_names=objective_names,
            seed=seed,
        )
    ]
