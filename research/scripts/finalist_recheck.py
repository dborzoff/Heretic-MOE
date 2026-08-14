# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare, run, and finalize an isolated high-fidelity finalist recheck."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

import optuna
import tomllib
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState

from heretic.config import SelectionPolicy
from heretic.multilingual_finalists import (
    freeze_top_six_manifest,
    select_multilingual_winners,
    select_top_six,
)
from heretic.trial_selection import candidate_trials

_MULTILINGUAL_RATE_CONSTRAINTS = {
    "max_safe_ppl_drift": "Safe PPL drift",
    "max_truncated_response_rate": "Truncated response rate",
    "max_safe_d_to_r_rate": "SAFE D->R rate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def params_sha256(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_study(path: Path) -> optuna.study.Study:
    storage = JournalStorage(
        JournalFileBackend(str(path), lock_obj=JournalFileOpenLock(str(path)))
    )
    summaries = optuna.study.get_all_study_summaries(storage)
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one study in {path}, found {len(summaries)}")
    return optuna.load_study(study_name=summaries[0].study_name, storage=storage)


def journal_name(model: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "--" for c in model)
    return f"{safe}.jsonl"


def replace_top_level(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    end = next((index for index, line in enumerate(lines) if line.lstrip().startswith("[")), len(lines))
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(end):
        if pattern.match(lines[index]):
            lines[index] = f"{key} = {value}"
            return "\n".join(lines) + "\n"
    lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def replace_table_value(text: str, table: str, key: str, value: str) -> str:
    lines = text.splitlines()
    header = f"[{table}]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header) + 1
    except StopIteration as error:
        raise RuntimeError(f"Missing TOML table {header}") from error
    end = next(
        (index for index in range(start, len(lines)) if lines[index].lstrip().startswith("[")),
        len(lines),
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start, end):
        if pattern.match(lines[index]):
            lines[index] = f"{key} = {value}"
            return "\n".join(lines) + "\n"
    lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def score_value(trial: FrozenTrial, *names: str) -> tuple[float, dict[str, Any]]:
    for record in trial.user_attrs.get("scores", []):
        if record.get("name") in names:
            score = record["score"]
            return float(score["value"]), score
    raise RuntimeError(f"Trial {trial.number} has no score named {names}")


def baseline_score_value(trial: FrozenTrial, *names: str) -> float | None:
    """Read a paired original-model baseline from a completed trial."""

    for record in trial.user_attrs.get("scores", []):
        if record.get("name") not in names:
            continue
        baseline = record.get("baseline")
        if not isinstance(baseline, dict) or "value" not in baseline:
            return None
        return float(baseline["value"])
    return None


def source_baseline_srg(source: optuna.study.Study) -> float | None:
    values = {
        value
        for trial in source.trials
        if trial.state == TrialState.COMPLETE
        and (
            value := baseline_score_value(trial, "Sparse refusal geometry")
        )
        is not None
    }
    if not values:
        return None
    if max(values) - min(values) > 1e-9:
        raise RuntimeError("Source trials contain inconsistent SRG baselines")
    return next(iter(values))


def finalist_ranking_settings(
    source_settings: dict[str, Any],
    base_config: dict[str, Any],
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Resolve finalist ranking from the current contract, not stale journals.

    Old search journals legitimately predate calibrated Cost metadata.  The
    recheck is launched with a currently validated base config, so its ranking
    targets and weights are authoritative.  Source settings remain a fallback
    only for optional diagnostic names used by non-Cost policies.
    """

    diagnostics = base_config.get(
        "selection_diagnostics",
        source_settings.get("selection_diagnostics", []),
    )
    targets = base_config.get("selection_score_targets", {})
    weights = base_config.get("selection_score_weights", {})
    return list(diagnostics), dict(targets), dict(weights)


def load_finalization_overrides(source_journal: Path) -> tuple[dict[str, Any], Path | None]:
    """Load an explicit run-local recovery policy, if one was provided."""

    run_root = source_journal.resolve().parents[2]
    path = run_root / "finalization_overrides.json"
    if not path.is_file():
        return {}, None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported finalization override schema: {path}")
    allowed = {
        "schema_version",
        "balanced_srg_gate",
        "baseline_srg",
        "balanced_removal_fraction",
        "source_constraints",
        "finalist_constraints",
        "max_safe_ppl_drift",
        "max_truncated_response_rate",
        "max_safe_d_to_r_rate",
        "provenance",
    }
    extras = sorted(set(record) - allowed)
    if extras:
        raise RuntimeError(f"Unknown finalization override keys: {extras}")
    legacy_source = {
        key: record[key]
        for key in _MULTILINGUAL_RATE_CONSTRAINTS
        if key in record
    }
    if legacy_source:
        if "source_constraints" in record:
            raise RuntimeError(
                "Legacy rate keys cannot be combined with source_constraints"
            )
        record = dict(record)
        record["source_constraints"] = legacy_source
        for key in legacy_source:
            record.pop(key)
    recovery_sections = {
        name: record[name]
        for name in ("source_constraints", "finalist_constraints")
        if name in record
    }
    for name, section in recovery_sections.items():
        if not isinstance(section, dict):
            raise TypeError(f"{name} must be an object")
        extras = sorted(set(section) - set(_MULTILINGUAL_RATE_CONSTRAINTS))
        if extras:
            raise RuntimeError(f"Unknown {name} keys: {extras}")
    if recovery_sections:
        provenance = record.get("provenance")
        if not isinstance(provenance, dict) or not str(
            provenance.get("reason", "")
        ).strip():
            raise RuntimeError("Rate recovery overrides require provenance.reason")
    return record, path


def _constraint_override_values(
    overrides: dict[str, Any], section_name: str
) -> dict[str, float]:
    section = overrides.get(section_name, {})
    if not isinstance(section, dict):
        raise TypeError(f"{section_name} must be an object")
    values: dict[str, float] = {}
    for key in _MULTILINGUAL_RATE_CONSTRAINTS:
        if key not in section:
            continue
        value = float(section[key])
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{key} must be in [0, 1]")
        values[key] = value
    return values


def apply_multilingual_constraint_overrides(
    settings_data: dict[str, Any],
    constraint_names: list[str],
    overrides: dict[str, Any],
) -> list[str]:
    """Apply explicit run-local rate gates to finalist settings and labels."""

    values = _constraint_override_values(overrides, "finalist_constraints")
    if not values:
        return list(constraint_names)
    contract = settings_data.get("multilingual_search")
    if not isinstance(contract, dict):
        raise TypeError("multilingual settings are missing for constraint recovery")
    updated = list(constraint_names)
    for key, value in values.items():
        label = _MULTILINGUAL_RATE_CONSTRAINTS[key]
        matches = [
            index
            for index, name in enumerate(updated)
            if name.startswith(f"{label} <= ")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label!r} constraint, found {len(matches)}")
        contract[key] = value
        updated[matches[0]] = f"{label} <= {value}"
    return updated


def _constraints_with_overrides(
    constraints: Any,
    constraint_names: list[str],
    overrides: dict[str, Any],
) -> list[float] | None:
    if not isinstance(constraints, (list, tuple)):
        return None
    adjusted = [float(value) for value in constraints]
    values = _constraint_override_values(overrides, "source_constraints")
    for key, new_limit in values.items():
        label = _MULTILINGUAL_RATE_CONSTRAINTS[key]
        matches = [
            index
            for index, name in enumerate(constraint_names)
            if name.startswith(f"{label} <= ")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {label!r} constraint, found {len(matches)}")
        index = matches[0]
        try:
            old_limit = float(constraint_names[index].rsplit("<=", 1)[1].strip())
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                f"Cannot parse the frozen limit from {constraint_names[index]!r}"
            ) from error
        observed_rate = adjusted[index] + old_limit
        adjusted[index] = observed_rate - new_limit
    return adjusted


def trial_metrics(trial: FrozenTrial) -> dict[str, Any]:
    srg, srg_record = score_value(trial, "Sparse refusal geometry")
    keywords, keyword_record = score_value(trial, "Keywords")
    ppl, _ = score_value(trial, "Perplexity drift", "PPL drift", "Perplexity increase")
    diagnostics = srg_record.get("diagnostics") or {}
    keyword_display = str(keyword_record.get("rich_display", ""))
    marker_match = re.search(r"(\d+)\s*/\s*(\d+)", keyword_display)
    marker_count = int(marker_match.group(1)) if marker_match else None
    marker_total = int(marker_match.group(2)) if marker_match else None
    return {
        "trial_number": trial.number,
        "trial_index": trial.user_attrs.get("index", trial.number + 1),
        "source_trial_index": trial.user_attrs.get("recheck_source_trial_index"),
        "params_sha256": params_sha256(trial.params),
        "srg": srg,
        "r_side": diagnostics.get("positive_rate"),
        "keyword_rate": keywords,
        "keyword_count": marker_count,
        "keyword_total": marker_total,
        "ppl_drift": abs(ppl),
    }


def multilingual_trial_metrics(trial: FrozenTrial) -> dict[str, Any]:
    _, record = score_value(trial, "Removal")
    public = record.get("diagnostics")
    if not isinstance(public, dict):
        raise TypeError(f"Trial {trial.number} has no multilingual diagnostics")
    metrics = public.get("metrics")
    diagnostics = public.get("diagnostics")
    if not isinstance(metrics, dict) or not isinstance(diagnostics, dict):
        raise TypeError(f"Trial {trial.number} has incomplete multilingual diagnostics")
    final = diagnostics.get("final_holdout")
    trial_groups = diagnostics.get("srg_groups")
    if not isinstance(final, dict) or not isinstance(trial_groups, dict):
        raise TypeError(f"Trial {trial.number} has no independent final holdout")
    final_groups = final.get("groups")
    if not isinstance(final_groups, dict):
        raise TypeError(f"Trial {trial.number} final holdout has no group summary")
    worst_language = min(
        float(trial_groups["worst_language"]),
        float(final_groups["worst_language"]),
    )
    worst_category = min(
        float(trial_groups["worst_category"]),
        float(final_groups["worst_category"]),
    )
    constraints = trial.user_attrs.get("constraints")
    feasible = trial.user_attrs.get("feasible")
    if not isinstance(feasible, bool):
        feasible = not isinstance(constraints, (list, tuple)) or all(
            float(value) <= 0.0 for value in constraints
        )
    source_number = int(trial.user_attrs["recheck_source_trial_number"])
    return {
        "trial_number": trial.number,
        "source_trial_number": source_number,
        "source_trial_index": int(trial.user_attrs["recheck_source_trial_index"]),
        "params_sha256": params_sha256(trial.params),
        "feasible": feasible,
        "removal": float(metrics["removal"]),
        "preservation_loss": float(metrics["preservation_loss"]),
        "safe_ppl_drift": float(metrics["safe_ppl_drift"]),
        "safe_geometry_damage": float(metrics["safe_geometry_drift"]),
        "worst_language": worst_language,
        "worst_category": worst_category,
        "final_holdout_removal": float(final["removal"]),
    }


def final_holdout_prepare_command(
    heretic: Path,
    manifest: dict[str, Any],
    runtime_root: Path,
    devices: list[str],
) -> list[str]:
    return [
        str(heretic),
        "prepare-final-holdout",
        "--config",
        str(manifest["config"]),
        "--runtime-root",
        str(runtime_root),
        "--top-six-manifest",
        str(manifest["top_six_manifest"]),
        "--devices",
        ",".join(str(device) for device in devices),
    ]


def validate_final_holdout_reference(reference: Path, top_six_path: Path) -> None:
    if not reference.is_file():
        raise FileNotFoundError(reference)
    top_six = json.loads(top_six_path.read_text(encoding="utf-8"))
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    expected = top_six.get("shortlist_contract_sha256")
    if (
        top_six.get("status") != "FROZEN"
        or manifest.get("status") != "PASS"
        or not isinstance(expected, str)
        or manifest.get("top_six_contract_sha256") != expected
    ):
        raise RuntimeError("Final holdout reference does not match the frozen TOP-6")


def _multilingual_enabled(settings: dict[str, Any]) -> bool:
    contract = settings.get("multilingual_search")
    return isinstance(contract, dict) and contract.get("enabled") is True


def _multilingual_final_holdout_sha256(settings: dict[str, Any]) -> str:
    contract = settings["multilingual_search"]
    runtime_manifest = (
        Path(str(contract["runtime_root"])) / "dataset" / "manifest.json"
    )
    source_manifest = (
        Path(str(contract["dataset_root"])) / "manifest.json"
    )
    manifest_path = runtime_manifest if runtime_manifest.is_file() else source_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("multilingual dataset manifest has no files mapping")
    records = []
    for language in ("en", "ru", "zh", "es", "fr"):
        name = f"srg_calibration_{language}.jsonl"
        record = files.get(name)
        if not isinstance(record, dict):
            raise TypeError(f"multilingual dataset manifest is missing {name}")
        records.append(
            {
                "name": name,
                "rows": int(record["rows"]),
                "sha256": str(record["sha256"]),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _multilingual_source_rows(
    source: optuna.study.Study,
    constraint_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overrides = constraint_overrides or {}
    constraint_names = list(source.user_attrs.get("constraint_names", []))
    for trial in source.trials:
        complete = trial.state == TrialState.COMPLETE and trial.values is not None
        if not complete:
            continue
        try:
            cost, _ = score_value(trial, "Cost↑", "Cost")
        except RuntimeError:
            removal = float(trial.values[0])
            loss = float(trial.values[1])
            cost = (1.0 / (1.0 + pow(2.718281828459045, -4.0 * removal))) / (
                1.0 + loss
            )
        constraints = _constraints_with_overrides(
            trial.user_attrs.get("constraints"), constraint_names, overrides
        )
        feasible = trial.user_attrs.get("feasible") if not overrides else None
        if not isinstance(feasible, bool):
            feasible = constraints is not None and all(
                float(value) <= 0.0 for value in constraints
            )
        rows.append(
            {
                "trial_number": trial.number,
                "source_trial_number": trial.number,
                "source_trial_index": int(
                    trial.user_attrs.get("index", trial.number + 1)
                ),
                "complete": True,
                "feasible": feasible,
                "removal": float(trial.values[0]),
                "preservation_loss": float(trial.values[1]),
                "cost_up": float(cost),
                "params": dict(trial.params),
                "params_sha256": params_sha256(trial.params),
            }
        )
    return rows


def prepare_multilingual(
    args: argparse.Namespace,
    source: optuna.study.Study,
    settings_data: dict[str, Any],
) -> None:
    if args.top_n != 6:
        raise RuntimeError("multilingual v3 finalization requires exactly TOP-6")
    overrides, override_path = load_finalization_overrides(args.source_journal)
    removal_fraction = float(
        overrides.get("balanced_removal_fraction", args.balanced_removal_fraction)
    )
    if not 0.0 <= removal_fraction <= 1.0:
        raise RuntimeError("balanced_removal_fraction must be in [0, 1]")
    source_constraint_overrides = {
        "source_constraints": _constraint_override_values(
            overrides, "source_constraints"
        )
    }
    finalist_constraint_overrides = {
        "finalist_constraints": _constraint_override_values(
            overrides, "finalist_constraints"
        )
    }
    constraint_names = apply_multilingual_constraint_overrides(
        settings_data,
        list(source.user_attrs.get("constraint_names", [])),
        finalist_constraint_overrides,
    )
    candidates = _multilingual_source_rows(source, source_constraint_overrides)
    if args.trial_indices:
        if len(args.trial_indices) != 6 or len(set(args.trial_indices)) != 6:
            raise RuntimeError("--trial-indices must contain six distinct entries")
        by_index = {int(row["source_trial_index"]): row for row in candidates}
        missing = [index for index in args.trial_indices if index not in by_index]
        if missing:
            raise RuntimeError(f"Completed source trials not found: {missing}")
        selected = [
            {**by_index[index], "shortlist_rank": rank, "shortlist_roles": ["explicit"]}
            for rank, index in enumerate(args.trial_indices, start=1)
        ]
        selection_mode = "explicit_verified_shortlist"
    else:
        selected = select_top_six(
            candidates,
            top_n=6,
            preservation_removal_fraction=removal_fraction,
        )
        selection_mode = "multilingual_diverse_top6"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    source_sha = sha256(args.source_journal.resolve())
    holdout_sha = _multilingual_final_holdout_sha256(settings_data)
    top6_path = output / "top6_manifest.json"
    top6 = freeze_top_six_manifest(
        top6_path,
        selected,
        source_journal_sha256=source_sha,
        final_holdout_sha256=holdout_sha,
    )

    multilingual = dict(settings_data["multilingual_search"])
    multilingual["evaluation_phase"] = "finalist"
    settings_data.update(
        {
            "n_trials": 6,
            "n_startup_trials": 0,
            "parallel_workers": len(args.devices),
            "worker_trial_budget": None,
            "optimization_only": True,
            "checkpoint_action": "continue",
            "leaderboard_size": 6,
            "geometry_trial_number_offset": 1_000_000,
            "study_checkpoint_dir": str(checkpoints).replace("\\", "/"),
            "multilingual_search": multilingual,
        }
    )
    config_text = args.base_config.read_text(encoding="utf-8")
    for key, value in (
        ("n_trials", "6"),
        ("n_startup_trials", "0"),
        ("parallel_workers", str(len(args.devices))),
        ("optimization_only", "true"),
        ("checkpoint_action", '"continue"'),
        ("leaderboard_size", "6"),
        ("geometry_trial_number_offset", "1000000"),
        ("study_checkpoint_dir", json.dumps(str(checkpoints).replace("\\", "/"))),
    ):
        config_text = replace_top_level(config_text, key, value)
    config_text = replace_table_value(
        config_text, "multilingual_search", "evaluation_phase", '"finalist"'
    )
    config_text = replace_table_value(
        config_text,
        "multilingual_search",
        "runtime_root",
        json.dumps(str(multilingual["runtime_root"]).replace("\\", "/")),
    )
    for key, value in finalist_constraint_overrides["finalist_constraints"].items():
        config_text = replace_table_value(
            config_text,
            "multilingual_search",
            key,
            json.dumps(value),
        )
    config = output / "config.toml"
    config.write_text(config_text, encoding="utf-8", newline="\n")

    journal = checkpoints / journal_name(str(settings_data["model"]))
    if journal.exists():
        raise FileExistsError(f"Refusing to overwrite existing recheck: {journal}")
    storage = JournalStorage(
        JournalFileBackend(str(journal), lock_obj=JournalFileOpenLock(str(journal)))
    )
    recheck = optuna.create_study(
        study_name="heretic", storage=storage, directions=source.directions
    )
    recheck.set_user_attr("settings", json.dumps(settings_data, separators=(",", ":")))
    recheck.set_user_attr("constraint_names", constraint_names)
    recheck.set_user_attr("finished", False)
    recheck.set_user_attr("top_six_contract_sha256", top6["shortlist_contract_sha256"])
    for row in selected:
        recheck.enqueue_trial(
            row["params"],
            user_attrs={
                "recheck_rank": int(row["shortlist_rank"]),
                "recheck_source_trial_number": int(row["source_trial_number"]),
                "recheck_source_trial_index": int(row["source_trial_index"]),
                "recheck_source_params_sha256": str(row["params_sha256"]),
                "shortlist_roles": list(row["shortlist_roles"]),
            },
            skip_if_exists=False,
        )

    manifest = {
        "version": 1,
        "status": "prepared",
        "contract": "multilingual_v3_full_recheck",
        "source_journal": str(args.source_journal.resolve()),
        "source_journal_sha256": source_sha,
        "base_config": str(args.base_config.resolve()),
        "base_config_sha256": sha256(args.base_config.resolve()),
        "config": str(config),
        "config_sha256": sha256(config),
        "journal": str(journal),
        "top_n": 6,
        "selection_mode": selection_mode,
        "selection_policy": args.selection_policy,
        "devices": list(args.devices),
        "gates": {
            "balanced_removal_fraction": removal_fraction,
            "source_constraint_overrides": source_constraint_overrides[
                "source_constraints"
            ],
            "finalist_constraint_overrides": finalist_constraint_overrides[
                "finalist_constraints"
            ],
        },
        "finalization_overrides": None
        if override_path is None
        else {"path": str(override_path), "sha256": sha256(override_path)},
        "top_six_manifest": str(top6_path),
        "top_six_manifest_sha256": sha256(top6_path),
        "final_holdout_sha256": holdout_sha,
        "runtime_root": str(Path(str(multilingual["runtime_root"])).resolve()),
        "selection": [
            {
                "rank": int(row["shortlist_rank"]),
                "source_trial_number": int(row["source_trial_number"]),
                "source_trial_index": int(row["source_trial_index"]),
                "params": row["params"],
                "params_sha256": row["params_sha256"],
                "shortlist_roles": row["shortlist_roles"],
            }
            for row in selected
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "top_n": 6}))


def prepare(args: argparse.Namespace) -> None:
    source = load_study(args.source_journal.resolve())
    with args.base_config.open("rb") as stream:
        base_config_data = tomllib.load(stream)
    settings_data = json.loads(source.user_attrs["settings"])
    if _multilingual_enabled(settings_data):
        prepare_multilingual(args, source, settings_data)
        return
    overrides, override_path = load_finalization_overrides(args.source_journal)
    balanced_srg_gate = overrides.get("balanced_srg_gate", args.balanced_srg_gate)
    baseline_srg = overrides.get("baseline_srg", args.baseline_srg)
    removal_fraction = overrides.get(
        "balanced_removal_fraction", args.balanced_removal_fraction
    )
    if baseline_srg is None:
        baseline_srg = source_baseline_srg(source)
    if balanced_srg_gate is None and baseline_srg is None:
        raise RuntimeError(
            "Relative Balanced selection requires an SRG baseline. Current Heretic "
            "journals record it automatically; for an older journal provide "
            "--baseline-srg or a run-local finalization_overrides.json."
        )
    if not 0 <= float(removal_fraction) <= 1:
        raise RuntimeError("balanced_removal_fraction must be in [0, 1]")
    diagnostic_names, score_targets, score_weights = finalist_ranking_settings(
        settings_data,
        base_config_data,
    )
    constraint_names = list(source.user_attrs.get("constraint_names", []))
    selection_policy = SelectionPolicy(args.selection_policy)
    ranked = candidate_trials(
        source.trials,
        source.directions,
        policy=selection_policy,
        constraint_count=len(constraint_names),
        primary_objective_index=0,
        diagnostic_names=diagnostic_names,
        score_targets=score_targets,
        score_weights=score_weights,
    )
    if args.trial_indices:
        if len(args.trial_indices) != args.top_n:
            raise RuntimeError("--trial-indices must contain exactly --top-n entries")
        if len(set(args.trial_indices)) != len(args.trial_indices):
            raise RuntimeError("--trial-indices contains duplicates")
        completed = [
            trial
            for trial in source.trials
            if trial.state == TrialState.COMPLETE and trial.values is not None
        ]
        by_index = {
            int(trial.user_attrs.get("index", trial.number + 1)): trial
            for trial in completed
        }
        missing = [index for index in args.trial_indices if index not in by_index]
        if missing:
            raise RuntimeError(f"Completed source trials not found: {missing}")
        selected = [by_index[index] for index in args.trial_indices]
        selection_mode = "explicit_verified_shortlist"
    else:
        if len(ranked) < args.top_n:
            raise RuntimeError(
                f"Requested TOP {args.top_n}, but only {len(ranked)} candidates exist"
            )
        selected = ranked[: args.top_n]
        selection_mode = "source_ranking_top_n"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    settings_data.update(
        {
            "n_trials": args.top_n,
            "n_startup_trials": 0,
            "parallel_workers": len(args.devices),
            "worker_trial_budget": None,
            "optimization_only": True,
            "checkpoint_action": "continue",
            "leaderboard_size": args.top_n,
            "study_checkpoint_dir": str(checkpoints).replace("\\", "/"),
            "selection_policy": selection_policy.value,
            "selection_diagnostics": diagnostic_names,
            "selection_score_targets": score_targets,
            "selection_score_weights": score_weights,
        }
    )
    scorer_settings = settings_data.setdefault("scorer", {})
    ppl_settings = scorer_settings.setdefault("Perplexity", {})
    ppl_settings["chunks"] = args.ppl_chunks
    ppl_settings["window"] = args.ppl_window

    config_text = args.base_config.read_text(encoding="utf-8")
    for key, value in (
        ("n_trials", str(args.top_n)),
        ("n_startup_trials", "0"),
        ("parallel_workers", str(len(args.devices))),
        ("optimization_only", "true"),
        ("checkpoint_action", '"continue"'),
        ("leaderboard_size", str(args.top_n)),
        ("study_checkpoint_dir", json.dumps(str(checkpoints).replace("\\", "/"))),
    ):
        config_text = replace_top_level(config_text, key, value)
    config_text = replace_table_value(
        config_text, "scorer.Perplexity", "chunks", str(args.ppl_chunks)
    )
    config_text = replace_table_value(
        config_text, "scorer.Perplexity", "window", str(args.ppl_window)
    )
    config = output / "config.toml"
    config.write_text(config_text, encoding="utf-8", newline="\n")

    recheck_journal = checkpoints / journal_name(str(settings_data["model"]))
    if recheck_journal.exists():
        raise FileExistsError(f"Refusing to overwrite existing recheck: {recheck_journal}")
    storage = JournalStorage(
        JournalFileBackend(
            str(recheck_journal), lock_obj=JournalFileOpenLock(str(recheck_journal))
        )
    )
    recheck = optuna.create_study(
        study_name="heretic",
        storage=storage,
        directions=source.directions,
    )
    recheck.set_user_attr("settings", json.dumps(settings_data, separators=(",", ":")))
    recheck.set_user_attr("constraint_names", constraint_names)
    recheck.set_user_attr("finished", False)
    for rank, trial in enumerate(selected, start=1):
        display_index = int(trial.user_attrs.get("index", trial.number + 1))
        recheck.enqueue_trial(
            trial.params,
            user_attrs={
                "recheck_rank": rank,
                "recheck_source_trial_number": trial.number,
                "recheck_source_trial_index": display_index,
                "recheck_source_params_sha256": params_sha256(trial.params),
                "recheck_ppl_chunks": args.ppl_chunks,
                "recheck_ppl_window": args.ppl_window,
            },
            skip_if_exists=False,
        )

    manifest = {
        "version": 1,
        "status": "prepared",
        "source_journal": str(args.source_journal.resolve()),
        "source_journal_sha256": sha256(args.source_journal.resolve()),
        "base_config": str(args.base_config.resolve()),
        "base_config_sha256": sha256(args.base_config.resolve()),
        "config": str(config),
        "config_sha256": sha256(config),
        "journal": str(recheck_journal),
        "top_n": args.top_n,
        "selection_mode": selection_mode,
        "selection_policy": selection_policy.value,
        "devices": args.devices,
        "ppl": {"chunks": args.ppl_chunks, "window": args.ppl_window},
        "gates": {
            "max_ppl_drift": args.max_ppl_drift,
            "max_keyword_rate": args.max_keywords / args.keyword_total,
            "max_keywords": args.max_keywords,
            "keyword_total": args.keyword_total,
            "keyword_near_gate_extra": args.keyword_near_gate_extra,
            "balanced_srg_gate": balanced_srg_gate,
            "balanced_removal_fraction": float(removal_fraction),
            "baseline_srg": None if baseline_srg is None else float(baseline_srg),
        },
        "finalization_overrides": None
        if override_path is None
        else {
            "path": str(override_path),
            "sha256": sha256(override_path),
        },
        "selection": [
            {
                "rank": rank,
                "source_trial_number": trial.number,
                "source_trial_index": trial.user_attrs.get("index", trial.number + 1),
                "params": trial.params,
                "params_sha256": params_sha256(trial.params),
            }
            for rank, trial in enumerate(selected, start=1)
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "top_n": args.top_n}))


def eligible_finalists(
    measured: list[dict[str, Any]], gates: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply strict gates first, then one explicit bounded keyword near-gate."""

    ppl_eligible = [
        row for row in measured if row["ppl_drift"] <= gates["max_ppl_drift"]
    ]
    strict = [
        row
        for row in ppl_eligible
        if row["keyword_rate"] <= gates["max_keyword_rate"]
    ]
    keyword_total = int(gates["keyword_total"])
    max_keywords = int(gates["max_keywords"])
    if strict:
        return strict, {
            "name": "strict",
            "max_keywords": max_keywords,
            "keyword_total": keyword_total,
            "keyword_excess": 0,
        }

    extra = int(gates.get("keyword_near_gate_extra", 0))
    fallback_max = max_keywords + extra
    fallback = [
        row
        for row in ppl_eligible
        if row["keyword_rate"] <= fallback_max / keyword_total
    ]
    return fallback, {
        "name": "keyword_near_gate",
        "max_keywords": fallback_max,
        "keyword_total": keyword_total,
        "keyword_excess": extra,
    }


def finalize(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study = load_study(Path(manifest["journal"]))
    if manifest.get("contract") == "multilingual_v3_full_recheck":
        expected = {int(row["source_trial_index"]) for row in manifest["selection"]}
        measured: dict[int, dict[str, Any]] = {}
        for trial in study.trials:
            source_index = trial.user_attrs.get("recheck_source_trial_index")
            if source_index is None or trial.state != TrialState.COMPLETE:
                continue
            source_index = int(source_index)
            if source_index in measured:
                raise RuntimeError(
                    f"Duplicate completed recheck for source T{source_index}"
                )
            measured[source_index] = multilingual_trial_metrics(trial)
        missing = sorted(expected - measured.keys())
        if missing:
            raise RuntimeError(f"Incomplete recheck; missing source trials: {missing}")
        report = select_multilingual_winners(
            list(measured.values()),
            balanced_removal_fraction=float(
                manifest["gates"]["balanced_removal_fraction"]
            ),
        )
        result = output / "winners.json"
        result.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"status": "PASS", "result": str(result), "winners": report["winners"]}
            )
        )
        return report
    expected = {int(row["source_trial_index"]) for row in manifest["selection"]}
    measured: dict[int, dict[str, Any]] = {}
    for trial in study.trials:
        source_index = trial.user_attrs.get("recheck_source_trial_index")
        if source_index is None or trial.state != TrialState.COMPLETE:
            continue
        source_index = int(source_index)
        if source_index in measured:
            raise RuntimeError(f"Duplicate completed recheck for source T{source_index}")
        measured[source_index] = trial_metrics(trial)
    missing = sorted(expected - measured.keys())
    if missing:
        raise RuntimeError(f"Incomplete recheck; missing source trials: {missing}")

    gates = manifest["gates"]
    eligible, eligibility_tier = eligible_finalists(list(measured.values()), gates)
    if not eligible:
        raise RuntimeError(
            "No rechecked finalist passes the PPL gate and bounded keyword near-gate"
        )

    baseline_value = gates.get("baseline_srg")
    baseline_srg = None if baseline_value is None else float(baseline_value)
    balanced_gate = gates.get("balanced_srg_gate")
    if balanced_gate is None:
        if baseline_srg is None:
            raise RuntimeError("Relative Balanced selection has no SRG baseline")
        best_srg = min(float(row["srg"]) for row in eligible)
        if best_srg >= baseline_srg:
            raise RuntimeError(
                "No eligible finalist improves SRG over the original-model baseline"
            )
        removal_fraction = float(gates["balanced_removal_fraction"])
        balanced_gate = baseline_srg - removal_fraction * (
            baseline_srg - best_srg
        )
        gates["balanced_gate_mode"] = "relative_baseline_to_best"
        gates["best_eligible_srg"] = best_srg
        gates["resolved_balanced_srg_gate"] = balanced_gate
    else:
        balanced_gate = float(balanced_gate)
        gates["balanced_gate_mode"] = "absolute"
        gates["resolved_balanced_srg_gate"] = balanced_gate

    improved = (
        eligible
        if baseline_srg is None
        else [row for row in eligible if row["srg"] < baseline_srg]
    )
    if not improved:
        raise RuntimeError("No rechecked finalist improves SRG over the original baseline")

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_values = (
            float(left["srg"]),
            float(left["ppl_drift"]),
            float(left["keyword_rate"]),
        )
        right_values = (
            float(right["srg"]),
            float(right["ppl_drift"]),
            float(right["keyword_rate"]),
        )
        return all(a <= b for a, b in zip(left_values, right_values)) and any(
            a < b for a, b in zip(left_values, right_values)
        )

    pareto = [
        row
        for row in improved
        if not any(other is not row and dominates(other, row) for other in improved)
    ]
    max_removal = min(
        pareto,
        key=lambda row: (
            row["srg"],
            float("inf") if row["r_side"] is None else row["r_side"],
            row["keyword_rate"],
            row["ppl_drift"],
        ),
    )
    balanced_pool = [
        row
        for row in pareto
        if row["srg"] <= balanced_gate
    ]
    if not balanced_pool:
        raise RuntimeError("No rechecked finalist passes the Balanced refusal gate")
    balanced = min(
        balanced_pool,
        key=lambda row: (row["ppl_drift"], row["keyword_rate"], row["srg"]),
    )
    winners_distinct = (
        int(balanced["source_trial_index"]) != int(max_removal["source_trial_index"])
    )
    distinct_alternatives = [
        row
        for row in improved
        if int(row["source_trial_index"]) != int(max_removal["source_trial_index"])
    ]
    alternative = (
        min(
            distinct_alternatives,
            key=lambda row: (row["ppl_drift"], row["keyword_rate"], row["srg"]),
        )
        if distinct_alternatives
        else None
    )
    report = {
        "status": "PASS",
        "contract": "pareto_extremes_or_single_winner",
        "measured": sorted(measured.values(), key=lambda row: row["source_trial_index"]),
        "winners": {"Balanced": balanced, "Max": max_removal},
        "winners_distinct": winners_distinct,
        "pareto_source_trial_indices": sorted(
            int(row["source_trial_index"]) for row in pareto
        ),
        "expansion_recommended": not winners_distinct,
        "distinct_backup_not_promoted": alternative,
        "eligibility_tier": eligibility_tier,
        "gates": gates,
    }
    result = output / "winners.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": str(result), "winners": report["winners"]}))
    return report


def worker_environment(base: dict[str, str], device: str) -> dict[str, str]:
    environment = dict(base)
    environment["HERETIC_MOE_INTERNAL"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    environment["PYTHONUNBUFFERED"] = "1"
    cache_suffix = f"gpu-{device}"
    for variable in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        if base_cache := environment.get(variable):
            environment[variable] = str(Path(base_cache) / cache_suffix)
    return environment


def runtime_pool_rows(runtime_root: Path, pool: str) -> int:
    """Read a frozen pool size from the runtime dataset contract."""

    manifest = json.loads(
        (runtime_root / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or pool not in counts:
        raise RuntimeError(f"Frozen runtime manifest has no {pool!r} row count")
    rows = int(counts[pool])
    if rows <= 0:
        raise RuntimeError(f"Frozen runtime pool {pool!r} must be nonempty")
    return rows


def worker_text_options() -> dict[str, object]:
    """Decode Heretic worker output exactly as the UTF-8 console emits it."""

    return {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }


def run(args: argparse.Namespace) -> None:
    from heretic.pipeline_ui import PipelineUI

    output = args.output_dir.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    devices = args.devices or [str(device) for device in manifest["devices"]]
    study = load_study(Path(manifest["journal"]))
    running = [trial.number for trial in study.trials if trial.state == TrialState.RUNNING]
    if running:
        raise RuntimeError(f"Recheck has RUNNING trials; inspect before recovery: {running}")
    waiting = sum(trial.state == TrialState.WAITING for trial in study.trials)
    if waiting == 0:
        finalize(output)
        return
    if manifest.get("contract") == "multilingual_v3_full_recheck":
        runtime_root = Path(manifest["runtime_root"])
        final_reference = runtime_root / "final_holdout_reference" / "manifest.json"
        if not final_reference.is_file():
            command = final_holdout_prepare_command(
                args.heretic,
                manifest,
                runtime_root,
                [str(device) for device in devices],
            )
            print(
                json.dumps(
                    {
                        "event": "final_holdout_reference_start",
                        "devices": [str(device) for device in devices],
                        "rows": runtime_pool_rows(runtime_root, "final_holdout"),
                    }
                ),
                flush=True,
            )
            completed = subprocess.run(command, cwd=output, check=False)
            if completed.returncode:
                raise RuntimeError(
                    f"Final holdout preparation failed with exit code {completed.returncode}"
                )
            if not final_reference.is_file():
                raise RuntimeError("Final holdout preparation produced no manifest")
        validate_final_holdout_reference(
            final_reference,
            Path(manifest["top_six_manifest"]),
        )
    workers = min(len(devices), waiting)
    budgets = [waiting // workers + (index < waiting % workers) for index in range(workers)]
    lock = threading.Lock()
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    readers: list[threading.Thread] = []
    progress = PipelineUI()
    worker_ids = tuple(f"gpu-{device}" for device in devices[:workers])
    progress.stage(
        "TOP-6 finalist recheck",
        total=waiting,
        description=f"Full trial pool + independent holdout on {workers} GPU(s)",
    )
    budget_by_worker = dict(zip(worker_ids, budgets, strict=True))
    for worker_id in worker_ids:
        progress.add_worker(worker_id, total=budget_by_worker[worker_id])
    completed_by_worker = {worker_id: 0 for worker_id in worker_ids}

    def stream(device: str, process: subprocess.Popen[str], log_handle: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            with lock:
                worker_id = f"gpu-{device}"
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    event = None
                if isinstance(event, dict) and event.get("event") == "trial_complete":
                    completed_by_worker[worker_id] += 1
                    progress.update_worker(
                        worker_id,
                        completed=min(
                            completed_by_worker[worker_id], budget_by_worker[worker_id]
                        ),
                        total=budget_by_worker[worker_id],
                    )
                    progress.update_overall(
                        completed=min(sum(completed_by_worker.values()), waiting),
                        total=waiting,
                    )
                lowered = line.lower()
                if any(
                    marker in lowered
                    for marker in ("traceback", "error:", " exception", "failed with")
                ):
                    print(f"[GPU {device}] {line}", end="")

    for worker, (device, budget) in enumerate(zip(devices[:workers], budgets, strict=True)):
        log_handle = (output / f"gpu{device}.log").open("a", encoding="utf-8")
        env = worker_environment(os.environ, str(device))
        command = [
            str(args.heretic),
            "--parallel-workers",
            str(workers),
            "--worker-trial-budget",
            str(budget),
            "--n-trials",
            # All finalists are already enqueued as WAITING trials. Heretic's
            # bounded-worker stop callback counts those reserved rows toward
            # the global target, so using top_n here stops each worker after
            # its first completion. Keep the ceiling above the existing rows;
            # the per-worker budget still bounds the exact amount of work.
            str(int(manifest["top_n"]) + waiting),
            "--n-startup-trials",
            "0",
            "--checkpoint-action",
            "continue",
            "--leaderboard-size",
            str(manifest["top_n"]),
            "--trial-responses-file",
            str(output / f"responses-gpu{device}.jsonl"),
            "--trial-response-number-offset",
            str(worker),
            "--trial-response-number-stride",
            str(workers),
            # Heretic treats a trailing non-option as a positional model path.
            # Keep a boolean option last so the stride value is not rewritten.
            "--optimization-only",
        ]
        print(json.dumps({"event": "recheck_worker_start", "device": device, "budget": budget, "command": command}))
        process = subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **worker_text_options(),
        )
        processes.append((device, process, log_handle))
        reader = threading.Thread(target=stream, args=(device, process, log_handle), daemon=True)
        reader.start()
        readers.append(reader)

    failures = []
    for device, process, _ in processes:
        code = process.wait()
        if code:
            failures.append((device, code))
    for reader in readers:
        reader.join()
    for _, _, log_handle in processes:
        log_handle.close()
    if failures:
        progress.finish_stage({"status": "FAIL", "failures": len(failures)})
        progress.close()
        raise RuntimeError(f"Recheck worker failure(s): {failures}")
    for worker_id, budget in zip(worker_ids, budgets, strict=True):
        completed_by_worker[worker_id] = budget
        progress.update_worker(
            worker_id,
            completed=budget,
            total=budget,
        )
    progress.update_overall(completed=waiting, total=waiting)
    report = finalize(output)
    winners = report["winners"]
    progress.finish_stage(
        {
            "status": "PASS",
            "finalists": waiting,
            "workers": workers,
            "Balanced": f"T{winners['Balanced']['source_trial_number']}",
            "Max": f"T{winners['Max']['source_trial_number']}",
            "next": "export selected models",
        }
    )
    progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-journal", type=Path, required=True)
    prepare_parser.add_argument("--base-config", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--top-n", type=int, default=6)
    prepare_parser.add_argument(
        "--selection-policy",
        choices=[policy.value for policy in SelectionPolicy],
        default=SelectionPolicy.FEASIBLE_DIVERSE.value,
    )
    prepare_parser.add_argument("--trial-indices", type=int, nargs="+")
    prepare_parser.add_argument("--ppl-chunks", type=int, default=64)
    prepare_parser.add_argument("--ppl-window", type=int, default=1024)
    prepare_parser.add_argument("--devices", nargs="+", default=["0"])
    prepare_parser.add_argument("--max-ppl-drift", type=float, default=0.005)
    prepare_parser.add_argument("--max-keywords", type=int, default=2)
    prepare_parser.add_argument("--keyword-total", type=int, default=136)
    prepare_parser.add_argument("--keyword-near-gate-extra", type=int, default=1)
    prepare_parser.add_argument("--balanced-srg-gate", type=float)
    prepare_parser.add_argument("--baseline-srg", type=float)
    prepare_parser.add_argument(
        "--balanced-removal-fraction", type=float, default=0.8
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--heretic", type=Path, required=True)
    run_parser.add_argument("--devices", nargs="+")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        finalize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
