#!/usr/bin/env python3
"""Create a continuation journal whose PPL objective is absolute drift."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_attrs(
    attrs: dict[str, Any], signed_change: float, limit: float
) -> dict[str, Any]:
    result = copy.deepcopy(attrs)
    drift = abs(signed_change)
    result["constraints"] = [drift - limit]
    result["feasible"] = drift <= limit
    for record in result.get("scores", []):
        if not isinstance(record, dict):
            continue
        if not str(record.get("name", "")).startswith("Perplexity"):
            continue
        record["name"] = "Perplexity drift"
        score = record.get("score") or {}
        diagnostics = score.setdefault("diagnostics", {})
        diagnostics["relative_change"] = signed_change
        diagnostics["absolute_relative_change"] = drift
        score["value"] = drift
        ppl = diagnostics.get("perplexity")
        if isinstance(ppl, (int, float)):
            score["rich_display"] = f"{ppl:.4f} ({drift * 100:.2f}% drift)"
            score["md_display"] = f"{ppl:.4f} ({drift * 100:.2f}% drift)"
        baseline = record.get("baseline")
        if isinstance(baseline, dict):
            baseline["value"] = 0.0
        break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ppl-objective-index", type=int, default=1)
    parser.add_argument("--ppl-limit", type=float, default=0.005)
    parser.add_argument("--drop-objective-index", type=int, action="append", default=[])
    parser.add_argument("--drop-scorer", action="append", default=[])
    parser.add_argument("--max-trials", type=int)
    args = parser.parse_args()

    if args.target.exists():
        raise FileExistsError(args.target)
    args.target.parent.mkdir(parents=True, exist_ok=True)

    source_storage = JournalStorage(JournalFileBackend(str(args.source)))
    summaries = optuna.get_all_study_summaries(source_storage)
    if len(summaries) != 1:
        raise ValueError(f"Expected one source study, got {len(summaries)}")
    source = optuna.load_study(
        study_name=summaries[0].study_name, storage=source_storage
    )

    dropped = set(args.drop_objective_index)
    if args.ppl_objective_index in dropped:
        raise ValueError("The PPL objective cannot be dropped")
    if any(index < 0 or index >= len(source.directions) for index in dropped):
        raise ValueError(f"Invalid dropped objective indices: {sorted(dropped)}")
    target_directions = [
        direction
        for index, direction in enumerate(source.directions)
        if index not in dropped
    ]
    source_trials = source.trials[: args.max_trials]

    target_storage = JournalStorage(JournalFileBackend(str(args.target)))
    target = optuna.create_study(
        study_name=source.study_name,
        storage=target_storage,
        directions=target_directions,
    )
    dropped_scorers = set(args.drop_scorer)
    for key, value in source.user_attrs.items():
        if key == "settings" and dropped_scorers:
            settings = json.loads(value)
            settings["scorers"] = [
                scorer
                for scorer in settings.get("scorers", [])
                if scorer.get("plugin") not in dropped_scorers
            ]
            scorer_settings = settings.get("scorer", {})
            for plugin in dropped_scorers:
                scorer_settings.pop(plugin.rsplit(".", 1)[-1], None)
            value = json.dumps(settings)
        target.set_user_attr(key, value)

    for trial in source_trials:
        if trial.values is None:
            raise ValueError(f"Trial {trial.number} has no objective values")
        values = list(trial.values)
        signed_change = float(values[args.ppl_objective_index])
        values[args.ppl_objective_index] = abs(signed_change)
        values = [value for index, value in enumerate(values) if index not in dropped]
        user_attrs = transform_attrs(trial.user_attrs, signed_change, args.ppl_limit)
        system_attrs = copy.deepcopy(trial.system_attrs)
        system_attrs["constraints"] = [abs(signed_change) - args.ppl_limit]
        target.add_trial(
            optuna.trial.create_trial(
                state=trial.state,
                values=values,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs=user_attrs,
                system_attrs=system_attrs,
                intermediate_values=trial.intermediate_values,
            )
        )

    migrated = target.trials
    if len(migrated) != len(source_trials):
        raise RuntimeError("Migrated trial count mismatch")
    for old, new in zip(source_trials, migrated, strict=True):
        if old.params != new.params:
            raise RuntimeError(f"Parameter mismatch at trial {old.number}")
        if old.values is None or new.values is None:
            raise RuntimeError(f"Missing values at trial {old.number}")
        expected = list(old.values)
        expected[args.ppl_objective_index] = abs(expected[args.ppl_objective_index])
        expected = [
            value for index, value in enumerate(expected) if index not in dropped
        ]
        if expected != list(new.values):
            raise RuntimeError(f"Objective mismatch at trial {old.number}")

    migrated_settings = json.loads(target.user_attrs["settings"])
    target_scorers = [
        scorer.get("plugin") for scorer in migrated_settings.get("scorers", [])
    ]
    remaining_dropped = dropped_scorers.intersection(target_scorers)
    if remaining_dropped:
        raise RuntimeError(f"Dropped scorers remain in settings: {remaining_dropped}")

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "target": str(args.target.resolve()),
        "target_sha256": sha256(args.target),
        "study_name": source.study_name,
        "trial_count": len(migrated),
        "source_trial_count": len(source.trials),
        "max_trials": args.max_trials,
        "ppl_objective_index": args.ppl_objective_index,
        "ppl_limit": args.ppl_limit,
        "dropped_objective_indices": sorted(dropped),
        "dropped_scorers": sorted(dropped_scorers),
        "target_scorers": target_scorers,
        "target_objective_count": len(target_directions),
        "transformation": "abs(perplexity / baseline_perplexity - 1)",
        "corrected_feasible": sum(
            bool(trial.user_attrs.get("feasible")) for trial in migrated
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
