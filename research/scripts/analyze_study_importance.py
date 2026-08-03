#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from heretic.study_diagnostics import write_parameter_importance_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a text-free fANOVA report from a Heretic Optuna journal."
    )
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=None,
        help="Analyse only the first N completed trials.",
    )
    return parser.parse_args()


def infer_objective_names(study: optuna.study.Study) -> list[str]:
    completed = [
        trial for trial in study.trials if trial.state == TrialState.COMPLETE
    ]
    if not completed:
        return [f"objective_{index}" for index in range(len(study.directions))]

    score_records = completed[0].user_attrs.get("scores")
    if isinstance(score_records, list):
        names = [
            record.get("name")
            for record in score_records
            if isinstance(record, dict) and isinstance(record.get("name"), str)
        ]
        if len(names) == len(study.directions):
            return names

    return [f"objective_{index}" for index in range(len(study.directions))]


def main() -> None:
    args = parse_args()
    journal = args.journal.resolve()
    output = args.output.resolve()
    if not journal.is_file():
        raise FileNotFoundError(journal)

    backend = JournalFileBackend(
        str(journal),
        lock_obj=JournalFileOpenLock(str(journal)),
    )
    storage = JournalStorage(backend)
    studies = storage.get_all_studies()
    if len(studies) != 1:
        raise ValueError(f"Expected exactly one study, found {len(studies)}")

    study = optuna.load_study(
        study_name=studies[0].study_name,
        storage=storage,
    )
    if args.trial_limit is not None:
        if args.trial_limit <= 0:
            raise ValueError("--trial-limit must be positive")
        completed = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
        ]
        if args.trial_limit > len(completed):
            raise ValueError(
                f"--trial-limit={args.trial_limit} exceeds {len(completed)} completed trials"
            )
        prefix_study = optuna.create_study(directions=study.directions)
        prefix_study.add_trials(completed[: args.trial_limit])
        study = prefix_study

    objective_names = infer_objective_names(study)
    report = write_parameter_importance_report(
        study,
        output,
        objective_names,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "completed_trials": report["completed_trials"],
                "pareto_trials": report["pareto_trials"],
                "objective_names": objective_names,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
