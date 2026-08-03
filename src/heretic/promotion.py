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
    else:
        selected = front[:count]

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
            for old_prefix, new_prefix in RENAMED_COMPONENT_PREFIXES.items():
                if name.startswith(old_prefix):
                    new_name = new_prefix + name[len(old_prefix) :]
                    break
            if new_name in allowed:
                kept[new_name] = value
        if kept:
            seeds.append(kept)
    return seeds
