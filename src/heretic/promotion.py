# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from .config import SeedSelection, SelectionPolicy
from .search import select_spread_points
from .trial_selection import candidate_trials, minimized_values


RENAMED_COMPONENT_PREFIXES = {"mlp.down_proj.": "mlp.experts.down_proj."}


def load_seed_parameters(
    path: str,
    count: int,
    components: list[str],
    selection: SeedSelection = SeedSelection.FIRST_OBJECTIVE,
    additional_trial_numbers: list[int] | None = None,
) -> list[dict]:
    """Load external parameter values from a previous feasible Pareto front.

    Objective scores are never copied. Returned parameters are intended for
    ``Study.enqueue_trial`` and are measured again at the new fidelity.
    """

    journal = str(Path(path).resolve())
    try:
        storage = JournalStorage(
            JournalFileBackend(
                journal,
                lock_obj=JournalFileOpenLock(journal),
            )
        )
        summaries = storage.get_all_studies()
    except OSError:
        return []
    if len(summaries) != 1:
        raise ValueError(f"Expected one study in seed journal, found {len(summaries)}")

    study = optuna.load_study(
        study_name=summaries[0].study_name,
        storage=storage,
    )
    completed = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if selection == SeedSelection.ALL:
        selected = completed[:count] if count else completed
    else:
        constraint_count = len(study.user_attrs.get("constraint_names", []))
        front = candidate_trials(
            completed,
            study.directions,
            policy=SelectionPolicy.FEASIBLE_LEXICOGRAPHIC,
            constraint_count=constraint_count,
            primary_objective_index=0,
        )

    if selection == SeedSelection.SPREAD:
        coordinates = [
            (minimized_values(trial, study.directions), trial.number)
            for trial in front
        ]
        selected_numbers = {
            trial_number
            for _, trial_number in select_spread_points(coordinates, count)
        }
        selected = [trial for trial in front if trial.number in selected_numbers]
    elif selection == SeedSelection.FIRST_OBJECTIVE:
        selected = front[:count]

    if additional_trial_numbers:
        completed_by_number = {trial.number: trial for trial in completed}
        selected_numbers = {trial.number for trial in selected}
        for trial_number in additional_trial_numbers:
            if trial_number not in completed_by_number:
                raise ValueError(
                    f"Seed trial {trial_number} is not complete in the source study"
                )
            if trial_number not in selected_numbers:
                selected.append(completed_by_number[trial_number])
                selected_numbers.add(trial_number)

    allowed = {"direction_scope", "direction_index"}
    for component in components:
        allowed.add(f"{component}.enabled")
        for suffix in (
            "max_weight",
            "max_weight_position",
            "min_weight",
            "min_weight_distance",
        ):
            allowed.add(f"{component}.{suffix}")

    seeds: list[dict] = []
    for trial in selected:
        kept: dict = {}
        for name, value in trial.params.items():
            new_name = name
            if name not in allowed:
                for old_prefix, new_prefix in RENAMED_COMPONENT_PREFIXES.items():
                    if name.startswith(old_prefix):
                        renamed = new_prefix + name[len(old_prefix) :]
                        if renamed in allowed:
                            new_name = renamed
                        break
            if new_name in allowed:
                kept[new_name] = value
        if kept:
            seeds.append(kept)
    return seeds
