#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from heretic.search import select_spread_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine text-free Pareto fronts from multiple Heretic journals."
    )
    parser.add_argument(
        "--study",
        action="append",
        required=True,
        metavar="NAME=JOURNAL",
        help="Named Optuna journal. Repeat for every study.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ppl-cap-frac", type=float, default=0.005)
    parser.add_argument("--finalists", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_study_spec(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"Invalid --study value: {spec!r}; expected NAME=JOURNAL")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def score_records(trial: optuna.trial.FrozenTrial) -> list[dict[str, Any]]:
    records = trial.user_attrs.get("scores")
    return records if isinstance(records, list) else []


def trial_record(
    study_name: str,
    trial: optuna.trial.FrozenTrial,
    pareto_index: int | None,
) -> dict[str, Any]:
    scores: dict[str, dict[str, Any]] = {}
    for record in score_records(trial):
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            continue
        score = record.get("score")
        if not isinstance(score, dict):
            continue
        scores[record["name"]] = {
            "value": score.get("value"),
            "display": score.get("md_display") or score.get("rich_display"),
        }

    return {
        "study": study_name,
        "trial_number": trial.number,
        "pareto_index": pareto_index,
        "display_index": trial.user_attrs.get("index"),
        "values": list(trial.values or ()),
        "scores": scores,
        "direction_index": trial.user_attrs.get("direction_index"),
        "params": trial.params,
        "abliteration_parameters": trial.user_attrs.get("parameters"),
    }


def dominates(left: list[float], right: list[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def combined_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for record in records:
        values = record["values"]
        if not values:
            continue
        if any(
            other is not record and dominates(other["values"], values)
            for other in records
            if other["values"]
        ):
            continue
        front.append(record)
    return sorted(front, key=lambda item: (item["values"], item["study"], item["trial_number"]))


def main() -> None:
    args = parse_args()
    if args.ppl_cap_frac < 0:
        raise ValueError("--ppl-cap-frac must be non-negative")
    if args.finalists <= 0:
        raise ValueError("--finalists must be positive")

    all_records: list[dict[str, Any]] = []
    studies: list[dict[str, Any]] = []
    directions: list[str] | None = None

    for spec in args.study:
        name, journal = parse_study_spec(spec)
        storage = JournalStorage(
            JournalFileBackend(
                str(journal),
                lock_obj=JournalFileOpenLock(str(journal)),
            )
        )
        summaries = storage.get_all_studies()
        if len(summaries) != 1:
            raise ValueError(f"Expected one study in {journal}, found {len(summaries)}")
        study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
        current_directions = [direction.name for direction in study.directions]
        if directions is None:
            directions = current_directions
        elif current_directions != directions:
            raise ValueError("All studies must use identical objective directions")
        if any(direction != "MINIMIZE" for direction in current_directions):
            raise ValueError("This report currently supports minimize objectives only")

        completed = [
            trial for trial in study.trials if trial.state == TrialState.COMPLETE
        ]
        sorted_pareto = sorted(
            study.best_trials,
            key=lambda trial: (tuple(trial.values or ()), trial.number),
        )
        pareto_indices = {
            trial.number: index for index, trial in enumerate(sorted_pareto)
        }
        records = [
            trial_record(name, trial, pareto_indices.get(trial.number))
            for trial in completed
        ]
        all_records.extend(records)
        studies.append(
            {
                "name": name,
                "journal": str(journal),
                "journal_sha256": sha256(journal),
                "completed_trials": len(completed),
                "total_trials": len(study.trials),
                "finished": bool(study.user_attrs.get("finished", False)),
            }
        )

    front = combined_front(all_records)
    capped_front = [
        record
        for record in front
        if len(record["values"]) >= 2 and record["values"][1] <= args.ppl_cap_frac
    ]
    spread_input = [
        (record["values"], index) for index, record in enumerate(capped_front)
    ]
    selected_indices = {
        index
        for _, index in select_spread_points(
            spread_input,
            min(args.finalists, len(spread_input)),
        )
    }
    finalists = [
        record for index, record in enumerate(capped_front) if index in selected_indices
    ]
    finalists.sort(key=lambda item: (item["values"], item["study"], item["trial_number"]))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "objective_directions": directions or [],
        "ppl_cap_frac": args.ppl_cap_frac,
        "selection": (
            "combined nondominated front under the PPL cap; objective-space "
            "extremes plus greedy maximum separation"
        ),
        "studies": studies,
        "completed_trials_total": len(all_records),
        "combined_front_count": len(front),
        "capped_front_count": len(capped_front),
        "combined_front": front,
        "finalists": finalists,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "completed_trials": len(all_records),
                "combined_front": len(front),
                "capped_front": len(capped_front),
                "finalists": [
                    {
                        "study": record["study"],
                        "trial_number": record["trial_number"],
                        "values": record["values"],
                    }
                    for record in finalists
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
