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


def diagnostic_value(trial: FrozenTrial, name: str) -> float | None:
    """Read a numeric scorer value recorded for display-only diagnostics."""

    records = trial.user_attrs.get("scores")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict) or record.get("name") != name:
            continue
        score = record.get("score")
        if not isinstance(score, dict):
            return None
        try:
            return float(score["value"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _nondominated_coordinates(
    points: Sequence[tuple[FrozenTrial, tuple[float, ...]]],
) -> list[tuple[FrozenTrial, tuple[float, ...]]]:
    front: list[tuple[FrozenTrial, tuple[float, ...]]] = []
    for trial, values in points:
        dominated = any(
            all(other_value <= value for other_value, value in zip(other, values))
            and any(other_value < value for other_value, value in zip(other, values))
            for other_trial, other in points
            if other_trial.number != trial.number
        )
        if not dominated:
            front.append((trial, values))
    return sorted(front, key=lambda item: item[0].number)


def _diverse_feasible_trials(
    feasible: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    *,
    primary_objective_index: int,
    diagnostic_names: Sequence[str],
) -> list[FrozenTrial]:
    """Rank complementary finalists instead of neighboring Pareto points.

    The leading roles are deliberately distinct: an ideal-point compromise,
    the strongest primary-objective point, and one extreme for each configured
    diagnostic. Remaining points maximize normalized separation from those roles.
    """

    valid_points: list[tuple[FrozenTrial, tuple[float, ...]]] = []
    incomplete: list[FrozenTrial] = []
    for trial in feasible:
        diagnostics = tuple(diagnostic_value(trial, name) for name in diagnostic_names)
        if any(value is None for value in diagnostics):
            incomplete.append(trial)
            continue
        valid_points.append(
            (
                trial,
                (
                    *minimized_values(trial, directions),
                    *(float(value) for value in diagnostics if value is not None),
                ),
            )
        )

    if not valid_points:
        return []

    front = _nondominated_coordinates(valid_points)
    lows = [
        min(values[index] for _, values in front) for index in range(len(front[0][1]))
    ]
    highs = [
        max(values[index] for _, values in front) for index in range(len(front[0][1]))
    ]

    def normalized(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            0.0
            if highs[index] == lows[index]
            else (value - lows[index]) / (highs[index] - lows[index])
            for index, value in enumerate(values)
        )

    coordinates = {trial.number: normalized(values) for trial, values in front}

    def ideal_key(item: tuple[FrozenTrial, tuple[float, ...]]) -> tuple[float, ...]:
        trial, values = item
        coordinate = coordinates[trial.number]
        return (
            sum(value * value for value in coordinate),
            *values,
            float(trial.number),
        )

    selected: list[FrozenTrial] = []

    def add(item: tuple[FrozenTrial, tuple[float, ...]]) -> None:
        trial = item[0]
        if all(existing.number != trial.number for existing in selected):
            selected.append(trial)

    # The first item is the balanced compromise, not a low-quality secondary
    # objective extreme. This is the default deployment candidate.
    add(min(front, key=ideal_key))

    # The second item is the strongest censorship-removal candidate.
    add(
        min(
            front,
            key=lambda item: (
                item[1][primary_objective_index],
                ideal_key(item),
            ),
        )
    )

    # Diagnostic extremes (for example zero keyword markers) are separate roles.
    objective_count = len(directions)
    for diagnostic_index in range(len(diagnostic_names)):
        coordinate_index = objective_count + diagnostic_index
        add(
            min(
                front,
                key=lambda item: (
                    item[1][coordinate_index],
                    item[1][primary_objective_index],
                    ideal_key(item),
                ),
            )
        )

    def squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right))

    remaining = [
        item
        for item in front
        if all(item[0].number != trial.number for trial in selected)
    ]
    while remaining:
        next_item = max(
            remaining,
            key=lambda item: (
                min(
                    squared_distance(
                        coordinates[item[0].number], coordinates[trial.number]
                    )
                    for trial in selected
                ),
                tuple(-value for value in ideal_key(item)),
            ),
        )
        add(next_item)
        remaining.remove(next_item)

    selected_numbers = {trial.number for trial in selected}
    dominated = sorted(
        (trial for trial in feasible if trial.number not in selected_numbers),
        key=lambda trial: (
            *minimized_values(trial, directions),
            float(trial.number),
        ),
    )
    incomplete_numbers = {trial.number for trial in incomplete}
    return (
        selected
        + [trial for trial in dominated if trial.number not in incomplete_numbers]
        + sorted(incomplete, key=lambda trial: trial.number)
    )


def _cost_ranked_trials(
    feasible: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    *,
    primary_objective_index: int,
    score_targets: dict[str, float],
    score_weights: dict[str, float],
) -> list[FrozenTrial]:
    """Rank feasible trials by weighted excess above lower-is-better targets."""

    score_names = tuple(name for name in score_targets if name in score_weights)
    if not score_names:
        raise ValueError(
            "feasible_cost requires at least one scorer in both "
            "selection_score_targets and selection_score_weights"
        )

    def cost_key(trial: FrozenTrial) -> tuple[float, ...]:
        cost = 0.0
        for name in score_names:
            value = diagnostic_value(trial, name)
            if value is None:
                cost = float("inf")
                break
            cost += score_weights[name] * max(0.0, value - score_targets[name])

        values = minimized_values(trial, directions)
        return (
            cost,
            values[primary_objective_index],
            *values,
            float(trial.number),
        )

    return sorted(feasible, key=cost_key)


def candidate_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    *,
    policy: SelectionPolicy,
    constraint_count: int,
    primary_objective_index: int = 0,
    diagnostic_names: Sequence[str] = (),
    score_targets: dict[str, float] | None = None,
    score_weights: dict[str, float] | None = None,
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
        if policy == SelectionPolicy.FEASIBLE_COST:
            return _cost_ranked_trials(
                feasible,
                directions,
                primary_objective_index=primary_objective_index,
                score_targets=score_targets or {},
                score_weights=score_weights or {},
            )
        if policy == SelectionPolicy.FEASIBLE_DIVERSE:
            return _diverse_feasible_trials(
                feasible,
                directions,
                primary_objective_index=primary_objective_index,
                diagnostic_names=diagnostic_names,
            )

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
