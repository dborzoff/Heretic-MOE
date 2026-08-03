#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import distribution_to_json
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState, create_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge completed trials into a new resumable Optuna journal."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=JOURNAL",
        help="Named source journal. Repeat for every source study.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-trials", type=int, required=True)
    parser.add_argument("--study-name", default="heretic")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"Invalid --source value: {spec!r}; expected NAME=JOURNAL")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def load_study(path: Path) -> optuna.study.Study:
    storage = JournalStorage(
        JournalFileBackend(
            str(path),
            lock_obj=JournalFileOpenLock(str(path)),
        )
    )
    summaries = storage.get_all_studies()
    if len(summaries) != 1:
        raise ValueError(f"Expected one study in {path}, found {len(summaries)}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


STAGE_ONLY_SETTING_KEYS = {
    "checkpoint_action",
    "device_map",
    "n_startup_trials",
    "n_trials",
    "optimization_only",
    "parallel_workers",
    "seed",
    "startup_design",
    "study_checkpoint_dir",
    "worker_trial_budget",
}


def study_contract(settings: dict[str, Any]) -> dict[str, Any]:
    """Drop stage controls while retaining the model/scorer/search contract."""

    return {
        key: value
        for key, value in settings.items()
        if key not in STAGE_ONLY_SETTING_KEYS
    }


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution_map(study: optuna.study.Study) -> dict[str, str]:
    distributions: dict[str, str] = {}
    for trial in study.trials:
        for name, distribution in trial.distributions.items():
            serialized = distribution_to_json(distribution)
            previous = distributions.setdefault(name, serialized)
            if previous != serialized:
                raise ValueError(
                    f"Study changes the distribution for parameter {name!r}"
                )
    return distributions


def main() -> None:
    args = parse_args()
    sources = [parse_source(spec) for spec in args.source]
    if args.target_trials <= 0:
        raise ValueError("--target-trials must be positive")

    output = args.output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".merge.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite merged study artifacts: {output}, {manifest_path}"
        )

    source_records: list[dict[str, Any]] = []
    source_trials: list[tuple[str, optuna.trial.FrozenTrial]] = []
    source_settings: dict[str, Any] | None = None
    source_contract: dict[str, Any] | None = None
    shared_distributions: dict[str, str] = {}
    directions: list[str] | None = None
    constraint_names: list[str] | None = None

    for source_name, source_path in sources:
        study = load_study(source_path)
        current_directions = [direction.name.lower() for direction in study.directions]
        if directions is None:
            directions = current_directions
        elif current_directions != directions:
            raise ValueError("Source studies use different objective directions")

        noncomplete = [
            trial.number
            for trial in study.trials
            if trial.state != TrialState.COMPLETE
        ]
        if noncomplete:
            raise ValueError(
                f"Source {source_name} contains non-complete trials: {noncomplete}"
            )
        raw_settings = study.user_attrs.get("settings")
        if not isinstance(raw_settings, str):
            raise ValueError(f"Source {source_name} has no archived settings JSON")
        current_settings = json.loads(raw_settings)
        current_contract = study_contract(current_settings)
        if source_settings is None:
            source_settings = current_settings
            source_contract = current_contract
        elif current_contract != source_contract:
            raise ValueError(
                f"Source {source_name} uses a different model/scorer/search contract"
            )

        current_constraints = list(study.user_attrs.get("constraint_names", []))
        if constraint_names is None:
            constraint_names = current_constraints
        elif current_constraints != constraint_names:
            raise ValueError("Source studies use different scorer constraints")

        current_distributions = distribution_map(study)
        for name in set(shared_distributions) & set(current_distributions):
            if shared_distributions[name] != current_distributions[name]:
                raise ValueError(
                    f"Source studies disagree on distribution for parameter {name!r}"
                )
        shared_distributions.update(current_distributions)

        source_records.append(
            {
                "name": source_name,
                "journal": str(source_path),
                "sha256": sha256(source_path),
                "trials": len(study.trials),
                "finished": bool(study.user_attrs.get("finished", False)),
                "contract_sha256": canonical_hash(current_contract),
            }
        )
        source_trials.extend((source_name, trial) for trial in study.trials)

    if args.target_trials <= len(source_trials):
        raise ValueError(
            f"--target-trials={args.target_trials} must exceed the merged prefix "
            f"of {len(source_trials)} trials"
        )
    assert source_settings is not None
    assert source_contract is not None
    assert directions is not None

    signatures: dict[str, int] = {}
    duplicate_parameter_sets = 0
    clones = []
    for merged_index, (source_name, trial) in enumerate(source_trials, start=1):
        signature = canonical_params(trial.params)
        if signature in signatures:
            duplicate_parameter_sets += 1
        else:
            signatures[signature] = merged_index

        user_attrs = dict(trial.user_attrs)
        user_attrs.update(
            {
                "index": merged_index,
                "merged_source": source_name,
                "merged_source_trial_number": trial.number,
                "merged_source_display_index": trial.user_attrs.get("index"),
            }
        )
        clones.append(
            create_trial(
                state=TrialState.COMPLETE,
                values=trial.values,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=user_attrs,
                system_attrs=trial.system_attrs,
                intermediate_values=trial.intermediate_values,
            )
        )

    source_settings.update(
        {
            "n_trials": args.target_trials,
            "n_startup_trials": 0,
            "startup_design": "random",
            "optimization_only": True,
            "study_checkpoint_dir": str(output.parent),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(
        JournalFileBackend(
            str(output),
            lock_obj=JournalFileOpenLock(str(output)),
        )
    )
    merged = optuna.create_study(
        storage=storage,
        study_name=args.study_name,
        directions=directions,
        load_if_exists=False,
    )
    merged.add_trials(clones)
    merged.set_user_attr(
        "settings",
        json.dumps(source_settings, separators=(",", ":")),
    )
    merged.set_user_attr("finished", False)
    merged.set_user_attr("constraint_names", constraint_names or [])
    merged.set_user_attr(
        "merge_provenance",
        {
            "sources": source_records,
            "merged_prefix_trials": len(clones),
            "target_trials": args.target_trials,
        },
    )

    reloaded = load_study(output)
    if len(reloaded.trials) != len(clones):
        raise RuntimeError(
            f"Merged journal reload has {len(reloaded.trials)} trials, expected {len(clones)}"
        )
    if any(trial.state != TrialState.COMPLETE for trial in reloaded.trials):
        raise RuntimeError("Merged journal reload contains non-complete prefix trials")

    manifest = {
        "schema_version": 1,
        "output": str(output),
        "output_sha256": sha256(output),
        "sources": source_records,
        "objective_directions": directions,
        "constraint_names": constraint_names or [],
        "contract_sha256": canonical_hash(source_contract),
        "search_space_distributions": shared_distributions,
        "merged_prefix_trials": len(clones),
        "target_trials": args.target_trials,
        "continuation_trials": args.target_trials - len(clones),
        "duplicate_parameter_sets": duplicate_parameter_sets,
        "startup_after_merge": "none; multivariate TPE continuation only",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
