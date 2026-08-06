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
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import FrozenTrial, TrialState

from heretic.config import SelectionPolicy
from heretic.trial_selection import candidate_trials


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
        "provenance",
    }
    extras = sorted(set(record) - allowed)
    if extras:
        raise RuntimeError(f"Unknown finalization override keys: {extras}")
    return record, path


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


def prepare(args: argparse.Namespace) -> None:
    source = load_study(args.source_journal.resolve())
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
    settings_data = json.loads(source.user_attrs["settings"])
    constraint_names = list(source.user_attrs.get("constraint_names", []))
    selection_policy = SelectionPolicy(args.selection_policy)
    ranked = candidate_trials(
        source.trials,
        source.directions,
        policy=selection_policy,
        constraint_count=len(constraint_names),
        primary_objective_index=0,
        diagnostic_names=settings_data.get("selection_diagnostics", []),
        score_targets=settings_data.get("selection_score_targets", {}),
        score_weights=settings_data.get("selection_score_weights", {}),
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


def finalize(output: Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study = load_study(Path(manifest["journal"]))
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
    eligible = [
        row
        for row in measured.values()
        if row["ppl_drift"] <= gates["max_ppl_drift"]
        and row["keyword_rate"] <= gates["max_keyword_rate"]
    ]
    if not eligible:
        raise RuntimeError("No rechecked finalist passes the PPL and keyword gates")

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

    max_removal = min(
        improved,
        key=lambda row: (
            row["srg"],
            float("inf") if row["r_side"] is None else row["r_side"],
            row["keyword_rate"],
            row["ppl_drift"],
        ),
    )
    balanced_pool = [
        row
        for row in improved
        if row is not max_removal and row["srg"] <= balanced_gate
    ]
    if not balanced_pool:
        raise RuntimeError("No rechecked finalist passes the Balanced refusal gate")
    balanced = min(
        balanced_pool,
        key=lambda row: (row["ppl_drift"], row["keyword_rate"], row["srg"]),
    )
    report = {
        "status": "PASS",
        "contract": "two_distinct_extremes_from_one_high_fidelity_top_n",
        "measured": sorted(measured.values(), key=lambda row: row["source_trial_index"]),
        "winners": {"Balanced": balanced, "Max": max_removal},
        "gates": gates,
    }
    result = output / "winners.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": str(result), "winners": report["winners"]}))
    return report


def run(args: argparse.Namespace) -> None:
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
    workers = min(len(devices), waiting)
    budgets = [waiting // workers + (index < waiting % workers) for index in range(workers)]
    lock = threading.Lock()
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    readers: list[threading.Thread] = []

    def stream(device: str, process: subprocess.Popen[str], log_handle: Any) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            with lock:
                print(f"[GPU {device}] {line}", end="")

    for worker, (device, budget) in enumerate(zip(devices[:workers], budgets, strict=True)):
        log_handle = (output / f"gpu{device}.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env["HERETIC_MOE_INTERNAL"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(device)
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            str(args.heretic),
            "--parallel-workers",
            str(workers),
            "--worker-trial-budget",
            str(budget),
            "--n-trials",
            str(manifest["top_n"]),
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
            text=True,
            bufsize=1,
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
        raise RuntimeError(f"Recheck worker failure(s): {failures}")
    finalize(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-journal", type=Path, required=True)
    prepare_parser.add_argument("--base-config", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--top-n", type=int, default=5)
    prepare_parser.add_argument(
        "--selection-policy",
        choices=[policy.value for policy in SelectionPolicy],
        default=SelectionPolicy.FEASIBLE_DIVERSE.value,
    )
    prepare_parser.add_argument("--trial-indices", type=int, nargs="+")
    prepare_parser.add_argument("--ppl-chunks", type=int, default=64)
    prepare_parser.add_argument("--ppl-window", type=int, default=1024)
    prepare_parser.add_argument("--devices", nargs="+", default=["0", "1"])
    prepare_parser.add_argument("--max-ppl-drift", type=float, default=0.005)
    prepare_parser.add_argument("--max-keywords", type=int, default=2)
    prepare_parser.add_argument("--keyword-total", type=int, default=136)
    prepare_parser.add_argument("--balanced-srg-gate", type=float)
    prepare_parser.add_argument("--baseline-srg", type=float)
    prepare_parser.add_argument(
        "--balanced-removal-fraction", type=float, default=0.0
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
