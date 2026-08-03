# SPDX-License-Identifier: AGPL-3.0-or-later

"""Constraint-aware, deterministic trial selection helpers."""

from __future__ import annotations

from collections.abc import Sequence

from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from .config import SelectionPolicy


def trial_constraints(trial: FrozenTrial, count: int) -> tuple[float, ...] | None:
    """Return a validated constraint vector, or None for unknown feasibility."""

    raw = trial.user_attrs.get("constraints")
    if count == 0:
        return ()
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        return None
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None


def is_feasible(trial: FrozenTrial, constraint_count: int) -> bool:
    constraints = trial_constraints(trial, constraint_count)
    return constraints is not None and all(value <= 0 for value in constraints)


def minimized_values(
    trial: FrozenTrial, directions: Sequence[StudyDirection]
) -> tuple[float, ...]:
    """Map all objectives to minimization while retaining their canonical order."""

    if trial.values is None or len(trial.values) != len(directions):
        raise ValueError("Trial values do not match objective directions")
    return tuple(
        float(value) if direction == StudyDirection.MINIMIZE else -float(value)
        for value, direction in zip(trial.values, directions)
    )


def nondominated_trials(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> list[FrozenTrial]:
    """Return a deterministic Pareto front from the supplied completed trials."""

    points = [(trial, minimized_values(trial, directions)) for trial in trials]
    front: list[FrozenTrial] = []
    for trial, values in points:
        dominated = any(
            all(other_value <= value for other_value, value in zip(other, values))
            and any(other_value < value for other_value, value in zip(other, values))
            for other_trial, other in points
            if other_trial.number != trial.number
        )
        if not dominated:
            front.append(trial)
    return sorted(front, key=lambda trial: trial.number)


def candidate_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    *,
    policy: SelectionPolicy,
    constraint_count: int,
    primary_objective_index: int = 0,
) -> list[FrozenTrial]:
    """Select and order the trials shown to users or automated exporters."""

    completed = [
        trial
        for trial in trials
        if trial.state == TrialState.COMPLETE and trial.values is not None
    ]
    if not completed:
        return []
    if not 0 <= primary_objective_index < len(directions):
        raise ValueError("primary_objective_index is out of range")

    if policy == SelectionPolicy.PARETO:
        front = nondominated_trials(completed, directions)
        return sorted(
            front,
            key=lambda trial: (*minimized_values(trial, directions), trial.number),
        )

    feasible = [trial for trial in completed if is_feasible(trial, constraint_count)]
    if feasible:
        pool = nondominated_trials(feasible, directions)

        def feasible_key(trial: FrozenTrial) -> tuple[float, ...]:
            values = minimized_values(trial, directions)
            reordered = (values[primary_objective_index],) + tuple(
                value
                for index, value in enumerate(values)
                if index != primary_objective_index
            )
            constraints = trial_constraints(trial, constraint_count) or ()
            # More negative means more headroom below an upper constraint.
            slack_key = max(constraints, default=0.0)
            return (*reordered, slack_key, float(trial.number))

        return sorted(pool, key=feasible_key)

    # When nothing is feasible, keep the least-violating trials visible instead
    # of silently presenting them as valid winners.
    def violation_key(trial: FrozenTrial) -> tuple[float, ...]:
        constraints = trial_constraints(trial, constraint_count)
        violation = (
            float("inf")
            if constraints is None
            else sum(max(0.0, value) for value in constraints)
        )
        return (violation, *minimized_values(trial, directions), float(trial.number))

    return sorted(completed, key=violation_key)

