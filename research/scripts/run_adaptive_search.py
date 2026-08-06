#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the bounded Random/Sobol -> shared TPE Heretic workflow.

The controller owns only configuration, processes, journals, and text-free
provenance. Prompt and response payloads remain inside the scorer processes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


@dataclass(frozen=True)
class Stage:
    name: str
    directory: Path
    config: Path
    journal: Path
    device: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two bounded exploration branches, merge their completed trials, "
            "then continue one shared multivariate-TPE study."
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--model",
        help="Override the model in the base config without editing the TOML file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Portable adaptive-search data bundle containing direction_safe.jsonl, "
            "direction_unsafe.jsonl, search_unsafe.jsonl, and prototypes.jsonl."
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--heretic",
        type=Path,
        help="Heretic executable (default: discover heretic in PATH).",
    )
    parser.add_argument(
        "--exploration-trials",
        type=int,
        default=120,
        help=(
            "Total exploration prefix across both GPUs. It is split evenly: "
            "half Random and half scrambled Sobol."
        ),
    )
    parser.add_argument("--target-trials", type=int, default=600)
    parser.add_argument("--random-device", default="0")
    parser.add_argument("--sobol-device", default="1")
    parser.add_argument(
        "--sequential-exploration",
        action="store_true",
        help="Debug mode: run Random and Sobol one after another.",
    )
    parser.add_argument(
        "--visible-worker-windows",
        action="store_true",
        help=(
            "On Windows, run every GPU stage in a separately titled console "
            "and tee its output to a stage-local log."
        ),
    )
    parser.add_argument(
        "--continue-shared-only",
        action="store_true",
        help="Skip branch creation/merge and extend an existing shared journal.",
    )
    parser.add_argument(
        "--allow-scorer-config-update",
        action="store_true",
        help=(
            "Explicitly allow replacing only the scorer section in existing "
            "managed stage configs, for example when increasing PPL windows."
        ),
    )
    parser.add_argument(
        "--finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the search, remeasure the TOP-N finalists at higher PPL "
            "fidelity, select distinct Balanced and Max winners, and export both."
        ),
    )
    parser.add_argument(
        "--search-only",
        dest="finalize",
        action="store_false",
        help="Stop after the 600-trial journal without recheck or model export.",
    )
    parser.add_argument("--finalist-top-n", type=int, default=5)
    parser.add_argument(
        "--finalist-selection-policy",
        choices=("pareto", "feasible_lexicographic", "feasible_diverse", "feasible_cost"),
        default="feasible_diverse",
        help=(
            "Ranking used only to build the high-fidelity finalist shortlist. "
            "The default deliberately covers distinct Pareto regions."
        ),
    )
    parser.add_argument("--recheck-ppl-chunks", type=int, default=64)
    parser.add_argument("--recheck-ppl-window", type=int, default=1024)
    parser.add_argument("--max-ppl-drift", type=float, default=0.005)
    parser.add_argument("--max-keywords", type=int, default=2)
    parser.add_argument("--keyword-total", type=int, default=136)
    parser.add_argument(
        "--balanced-srg-gate",
        type=float,
        help=(
            "Optional absolute SRG threshold for Balanced. By default the gate "
            "is derived from the original-model baseline and the best rechecked "
            "finalist."
        ),
    )
    parser.add_argument(
        "--balanced-removal-fraction",
        type=float,
        default=0.8,
        help=(
            "Required fraction of the SRG improvement from the original baseline "
            "to the best rechecked finalist when no absolute gate is supplied."
        ),
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        help="Output root for Balanced and Max (default: RUN_ROOT/exports).",
    )
    parser.add_argument(
        "--export-strategy",
        choices=("merge", "adapter"),
        default="merge",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repository: Path) -> tuple[str | None, bool | None]:
    """Capture the source revision once without requiring Git at runtime."""

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ).stdout.strip()
        )
        return revision or None, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def sanitized_model_name(model: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "--" for c in model)


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"Base config has no non-empty model: {path}")
    return config


def apply_data_root(base: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Point the known adaptive scorers at one portable local data bundle."""

    root = data_root.resolve()
    paths = {
        "direction_safe": root / "direction_safe.jsonl",
        "direction_unsafe": root / "direction_unsafe.jsonl",
        "search_unsafe": root / "search_unsafe.jsonl",
        "prototypes": root / "prototypes.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Data bundle is incomplete: " + ", ".join(missing))

    config = copy.deepcopy(base)
    config["good_prompts"]["dataset"] = paths["direction_safe"].as_posix()
    config["bad_prompts"]["dataset"] = paths["direction_unsafe"].as_posix()
    scorer = config["scorer"]
    sparse = scorer["SparseRefusalGeometry"]
    sparse["prototypes"] = paths["prototypes"].as_posix()
    sparse["prototypes_sha256"] = sha256(paths["prototypes"])
    sparse["prompts"]["dataset"] = paths["search_unsafe"].as_posix()
    scorer["KeywordRate"]["prompts"]["dataset"] = paths[
        "search_unsafe"
    ].as_posix()
    return config


def stage_config(
    base: dict[str, Any],
    *,
    checkpoint_dir: Path,
    n_trials: int,
    n_startup_trials: int,
    startup_design: str,
    response_archive: Path,
    response_number_offset: int,
    response_number_stride: int,
    parallel_workers: int,
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "device_map": "cuda:0",
            "n_trials": n_trials,
            "n_startup_trials": n_startup_trials,
            "startup_design": startup_design,
            "optimization_only": True,
            "checkpoint_action": "continue",
            "study_checkpoint_dir": checkpoint_dir.as_posix(),
            "parallel_workers": parallel_workers,
            "trial_responses_file": response_archive.as_posix(),
            "trial_response_number_offset": response_number_offset,
            "trial_response_number_stride": response_number_stride,
        }
    )
    return config


def write_managed_config(
    path: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
    allowed_updates: frozenset[str] = frozenset(),
) -> None:
    payload = tomli_w.dumps(config).encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing == payload:
            return
        with path.open("rb") as stream:
            existing_config = tomllib.load(stream)
        changed_keys = {
            key
            for key in set(existing_config) | set(config)
            if existing_config.get(key) != config.get(key)
        }
        if changed_keys and changed_keys.issubset(allowed_updates):
            if not dry_run:
                path.write_bytes(payload)
            return
        raise FileExistsError(
            f"Refusing to replace a different stage config: {path}. "
            f"Changed keys: {sorted(changed_keys)}. Use a new run root or "
            "update it intentionally."
        )
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_stage(
    root: Path,
    name: str,
    base: dict[str, Any],
    *,
    n_trials: int,
    n_startup_trials: int,
    startup_design: str,
    device: str | None,
    response_archive: Path,
    response_number_offset: int,
    response_number_stride: int,
    parallel_workers: int,
    dry_run: bool,
    allowed_config_updates: frozenset[str] = frozenset(),
) -> Stage:
    directory = root / name
    checkpoint_dir = directory / "checkpoints"
    config_path = directory / "config.toml"
    config = stage_config(
        base,
        checkpoint_dir=checkpoint_dir,
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        startup_design=startup_design,
        response_archive=response_archive,
        response_number_offset=response_number_offset,
        response_number_stride=response_number_stride,
        parallel_workers=parallel_workers,
    )
    write_managed_config(
        config_path,
        config,
        dry_run=dry_run,
        allowed_updates=allowed_config_updates,
    )
    journal = checkpoint_dir / f"{sanitized_model_name(str(base['model']))}.jsonl"
    return Stage(name, directory, config_path, journal, device)


def process_environment(device: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Child processes normally inherit a redirected controller stdout on CI and
    # through orchestration tools. Force line-by-line progress instead of
    # releasing several minutes of output only when a stage exits.
    environment["PYTHONUNBUFFERED"] = "1"
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["HERETIC_WORKER_LABEL"] = f"GPU {device}"
    return environment


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def start_stage(
    stage: Stage,
    executable: Path,
    *,
    dry_run: bool,
    command_args: tuple[str, ...] = (),
    display_name: str | None = None,
    device: str | None = None,
    visible_worker_window: bool = False,
) -> subprocess.Popen:
    command = [str(executable), *command_args]
    effective_device = stage.device if device is None else device
    effective_name = display_name or stage.name
    print(
        json.dumps(
            {
                "event": "stage_start",
                "stage": effective_name,
                "cwd": str(stage.directory),
                "device": effective_device,
                "command": command,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if dry_run:
        return subprocess.Popen(
            [sys.executable, "-c", "pass"],
            cwd=Path.cwd(),
        )
    if visible_worker_window:
        if os.name != "nt":
            raise ValueError("--visible-worker-windows is supported only on Windows")
        log_path = stage.directory / f"{effective_name}.console.log"
        title = f"CODEX | Heretic | {effective_name} | GPU {effective_device}"
        native_command_line = subprocess.list2cmdline(command) + " 2>&1"
        powershell_command = (
            "$ErrorActionPreference='Continue'; "
            f"$Host.UI.RawUI.WindowTitle={powershell_quote(title)}; "
            "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
            f"& cmd.exe /d /s /c {powershell_quote(native_command_line)} | "
            f"Tee-Object -FilePath {powershell_quote(str(log_path))}; "
            "$code=$LASTEXITCODE; exit $code"
        )
        return subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                powershell_command,
            ],
            cwd=stage.directory,
            env=process_environment(effective_device),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
    return subprocess.Popen(
        command,
        cwd=stage.directory,
        env=process_environment(effective_device),
    )


def wait_stage(stage: Stage, process: subprocess.Popen) -> None:
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage {stage.name} failed with exit code {return_code}")
    if not stage.journal.is_file():
        raise FileNotFoundError(f"Stage {stage.name} produced no journal: {stage.journal}")
    print(
        json.dumps(
            {
                "event": "stage_complete",
                "stage": stage.name,
                "journal": str(stage.journal),
                "journal_sha256": sha256(stage.journal),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_stage(
    stage: Stage,
    executable: Path,
    *,
    dry_run: bool,
    visible_worker_window: bool = False,
) -> None:
    process = start_stage(
        stage,
        executable,
        dry_run=dry_run,
        visible_worker_window=visible_worker_window,
    )
    if dry_run:
        process.wait()
        return
    wait_stage(stage, process)


def wait_parallel(
    stages: list[tuple[Stage, subprocess.Popen]],
) -> None:
    failures: list[str] = []
    for stage, process in stages:
        try:
            wait_stage(stage, process)
        except Exception as error:
            failures.append(f"{stage.name}: {error}")
    if failures:
        raise RuntimeError("Parallel stage failure(s): " + "; ".join(failures))


def journal_trial_count(journal: Path) -> int:
    total, _ = journal_trial_counts(journal)
    return total


def journal_trial_counts(journal: Path) -> tuple[int, int]:
    """Return total and queued-WAITING trial counts for a shared journal."""

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
    trials = study.get_trials(deepcopy=False)
    waiting = sum(
        trial.state == optuna.trial.TrialState.WAITING for trial in trials
    )
    return len(trials), waiting


def require_constraint_metadata(stage: Stage) -> None:
    storage = JournalStorage(
        JournalFileBackend(
            str(stage.journal),
            lock_obj=JournalFileOpenLock(str(stage.journal)),
        )
    )
    summaries = storage.get_all_studies()
    if len(summaries) != 1:
        raise ValueError(
            f"Expected one study in {stage.journal}, found {len(summaries)}"
        )
    study = optuna.load_study(
        study_name=summaries[0].study_name,
        storage=storage,
    )
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    missing = [
        trial.number for trial in complete if "constraints" not in trial.system_attrs
    ]
    if missing:
        preview = ", ".join(str(number) for number in missing[:8])
        raise RuntimeError(
            "Constraint metadata incomplete: "
            f"{len(complete) - len(missing)}/{len(complete)}; "
            f"missing trials: {preview}"
        )
    print(f"Constraints: {len(complete)}/{len(complete)} OK", flush=True)


def require_trial_count(stage: Stage, expected: int) -> None:
    actual = journal_trial_count(stage.journal)
    if actual != expected:
        raise RuntimeError(
            f"Stage {stage.name} has {actual} trials, expected exactly {expected}"
        )


def split_worker_budget(remaining_trials: int, worker_count: int = 2) -> list[int]:
    if remaining_trials < 0:
        raise ValueError("remaining_trials cannot be negative")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    quotient, remainder = divmod(remaining_trials, worker_count)
    return [quotient + (1 if index < remainder else 0) for index in range(worker_count)]


def assigned_devices(args: argparse.Namespace) -> list[str]:
    """Return distinct worker devices in stable user-specified order."""

    return list(dict.fromkeys((str(args.random_device), str(args.sobol_device))))


def merge_branches(
    random_stage: Stage,
    sobol_stage: Stage,
    shared_stage: Stage,
    *,
    target_trials: int,
    dry_run: bool,
) -> None:
    merge_script = Path(__file__).with_name("merge_optuna_studies.py")
    command = [
        sys.executable,
        str(merge_script),
        "--source",
        f"random={random_stage.journal}",
        "--source",
        f"sobol={sobol_stage.journal}",
        "--output",
        str(shared_stage.journal),
        "--target-trials",
        str(target_trials),
        "--order",
        "round-robin",
    ]
    print(json.dumps({"event": "merge", "command": command}), flush=True)
    if dry_run:
        return
    if shared_stage.journal.exists():
        manifest = shared_stage.journal.with_suffix(
            shared_stage.journal.suffix + ".merge.json"
        )
        if not manifest.is_file():
            raise FileExistsError(
                f"Shared journal exists without merge manifest: {shared_stage.journal}"
            )
        return
    shared_stage.journal.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=Path(__file__).parents[2])


def write_run_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    base_config: Path,
    stages: list[Stage],
    status: str,
) -> None:
    created_unix = time.time()
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            created_unix = float(previous.get("created_unix", created_unix))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    response_archive = path.parent / "trial-responses.sqlite3"
    devices = assigned_devices(args)
    record = {
        "schema_version": 2,
        "status": status,
        "created_unix": created_unix,
        "updated_unix": time.time(),
        "base_config": str(base_config),
        "base_config_sha256": sha256(base_config),
        "source_base_config": str(args.base_config.resolve()),
        "source_base_config_sha256": sha256(args.base_config.resolve()),
        "launch_provenance": {
            "controller_path": str(args.controller_path),
            "controller_sha256": args.controller_sha256,
            "heretic_executable": str(args.heretic_path),
            "heretic_executable_sha256": args.heretic_sha256,
            "git_revision": args.git_revision,
            "git_dirty": args.git_dirty,
        },
        "model_override": args.model,
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "exploration_trials": args.exploration_trials,
        "trials_per_exploration_branch": args.exploration_trials // 2,
        "target_trials": args.target_trials,
        "parallel_exploration": not args.sequential_exploration and len(devices) > 1,
        "worker_devices": devices,
        "shared_worker_count": len(devices),
        "visible_worker_windows": args.visible_worker_windows,
        "continue_shared_only": args.continue_shared_only,
        "finalize": args.finalize,
        "finalist_top_n": args.finalist_top_n,
        "finalist_selection_policy": args.finalist_selection_policy,
        "recheck_ppl": {
            "chunks": args.recheck_ppl_chunks,
            "window": args.recheck_ppl_window,
        },
        "export_root": str((args.export_root or path.parent / "exports").resolve()),
        "export_strategy": args.export_strategy,
        "response_archive": str(response_archive),
        "response_archive_size": (
            response_archive.stat().st_size if response_archive.is_file() else 0
        ),
        "stages": [
            {
                "name": stage.name,
                "directory": str(stage.directory),
                "config": str(stage.config),
                "config_sha256": sha256(stage.config) if stage.config.is_file() else None,
                "journal": str(stage.journal),
                "journal_sha256": sha256(stage.journal)
                if stage.journal.is_file()
                else None,
                "trial_count": journal_trial_count(stage.journal)
                if stage.journal.is_file()
                else 0,
                "device": stage.device,
            }
            for stage in stages
        ],
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    write_text_atomic(path, payload)


def write_text_atomic(path: Path, payload: str) -> None:
    """Replace a controller text artifact without exposing a partial file."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def command_line_value(arguments: list[str], name: str) -> str | None:
    """Read one option from argv without invoking the complete CLI parser."""

    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def mark_existing_run_failed(arguments: list[str], error: BaseException) -> None:
    """Leave a machine-readable terminal state after an unhandled failure."""

    run_root = command_line_value(arguments, "--run-root")
    if run_root is None:
        return
    manifest_path = Path(run_root).resolve() / "adaptive_run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    record["status"] = "failed"
    record["updated_unix"] = time.time()
    record["failure"] = {"type": type(error).__name__}
    try:
        write_text_atomic(
            manifest_path,
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        )
    except OSError:
        # Failure reporting must never mask the original controller exception.
        return


def run_checked(command: list[str], *, cwd: Path, event: str) -> None:
    print(
        json.dumps(
            {"event": event, "cwd": str(cwd), "command": command},
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=cwd, check=True)


def export_is_complete(directory: Path) -> bool:
    manifest_path = directory / "heretic_moe_export.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "PASS":
        return False
    for record in manifest.get("files", []):
        path = directory / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            return False
        if sha256(path) != record["sha256"]:
            return False
    return True


def write_export_manifest(
    directory: Path,
    *,
    variant: str,
    winner: dict[str, Any],
    device: str,
    export_strategy: str,
) -> Path:
    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        if path.name == "heretic_moe_export.json":
            continue
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if not files or not any(record["path"].endswith(".safetensors") for record in files):
        raise RuntimeError(f"Export {variant} contains no safetensors weights: {directory}")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "variant": variant,
        "device": device,
        "export_strategy": export_strategy,
        "winner": winner,
        "files": files,
    }
    path = directory / "heretic_moe_export.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def finalize_and_export(
    args: argparse.Namespace,
    *,
    root: Path,
    base_config: Path,
    shared_stage: Stage,
    executable: Path,
) -> None:
    finalist_script = Path(__file__).with_name("finalist_recheck.py")
    finalist_dir = root / "finalist_recheck"
    export_root = (args.export_root or root / "exports").resolve()
    devices = assigned_devices(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "event": "finalization_plan",
                    "source_journal": str(shared_stage.journal),
                    "top_n": args.finalist_top_n,
                    "selection_policy": args.finalist_selection_policy,
                    "recheck": {
                        "ppl_chunks": args.recheck_ppl_chunks,
                        "ppl_window": args.recheck_ppl_window,
                        "devices": devices,
                    },
                    "winners": ["Balanced", "Max"],
                    "export_root": str(export_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    finalist_manifest = finalist_dir / "manifest.json"
    if not finalist_manifest.is_file():
        prepare_command = [
            sys.executable,
            str(finalist_script),
            "prepare",
            "--source-journal",
            str(shared_stage.journal),
            "--base-config",
            str(base_config),
            "--output-dir",
            str(finalist_dir),
            "--top-n",
            str(args.finalist_top_n),
            "--selection-policy",
            args.finalist_selection_policy,
            "--ppl-chunks",
            str(args.recheck_ppl_chunks),
            "--ppl-window",
            str(args.recheck_ppl_window),
            "--devices",
            *devices,
            "--max-ppl-drift",
            str(args.max_ppl_drift),
            "--max-keywords",
            str(args.max_keywords),
            "--keyword-total",
            str(args.keyword_total),
            "--balanced-removal-fraction",
            str(args.balanced_removal_fraction),
        ]
        if args.balanced_srg_gate is not None:
            prepare_command.extend(
                ["--balanced-srg-gate", str(args.balanced_srg_gate)]
            )
        run_checked(prepare_command, cwd=Path(__file__).parents[2], event="finalist_prepare")
    else:
        print(
            json.dumps(
                {"event": "finalist_prepare_resume", "manifest": str(finalist_manifest)},
                sort_keys=True,
            ),
            flush=True,
        )

    recheck_command = [
        sys.executable,
        str(finalist_script),
        "run",
        "--output-dir",
        str(finalist_dir),
        "--heretic",
        str(executable),
        "--devices",
        *devices,
    ]
    run_checked(recheck_command, cwd=Path(__file__).parents[2], event="finalist_recheck")

    winners_path = finalist_dir / "winners.json"
    winners_report = json.loads(winners_path.read_text(encoding="utf-8"))
    winners = winners_report.get("winners", {})
    if set(winners) != {"Balanced", "Max"}:
        raise RuntimeError(f"Expected Balanced and Max winners, found {sorted(winners)}")
    if winners["Balanced"]["trial_number"] == winners["Max"]["trial_number"]:
        raise RuntimeError("Balanced and Max must be distinct rechecked trials")

    export_root.mkdir(parents=True, exist_ok=True)
    export_jobs: list[
        tuple[str, str, Path, dict[str, Any], subprocess.Popen[bytes]]
    ] = []
    export_assignments = (("Balanced", devices[0]), ("Max", devices[-1]))
    for variant, device in export_assignments:
        output = export_root / variant.lower()
        if export_is_complete(output):
            print(
                json.dumps(
                    {"event": "export_resume", "variant": variant, "path": str(output)},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite incomplete {variant} export: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        winner = winners[variant]
        command = [
            str(executable),
            "--checkpoint-action",
            "continue",
            "--restore-trial-number",
            str(winner["trial_number"]),
            "--model-action",
            "save",
            "--save-directory",
            str(output),
            "--export-strategy",
            args.export_strategy,
            "--no-optimization-only",
        ]
        print(
            json.dumps(
                {
                    "event": "export_start",
                    "variant": variant,
                    "device": device,
                    "source_trial_index": winner["source_trial_index"],
                    "recheck_trial_number": winner["trial_number"],
                    "path": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=finalist_dir,
            env=process_environment(str(device)),
        )
        export_jobs.append((variant, str(device), output, winner, process))

        # Two distinct GPUs export concurrently. With one GPU, finish each
        # variant before loading the next full model to avoid deterministic OOM.
        if len(devices) == 1:
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Export failure: {variant} on GPU {device}: exit {return_code}")
            export_manifest = write_export_manifest(
                output,
                variant=variant,
                winner=winner,
                device=str(device),
                export_strategy=args.export_strategy,
            )
            print(
                json.dumps(
                    {
                        "event": "export_complete",
                        "variant": variant,
                        "path": str(output),
                        "manifest": str(export_manifest),
                        "manifest_sha256": sha256(export_manifest),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            export_jobs.pop()

    export_failures: list[str] = []
    for variant, device, output, winner, process in export_jobs:
        return_code = process.wait()
        if return_code != 0:
            export_failures.append(f"{variant} on GPU {device}: exit {return_code}")
            continue
        export_manifest = write_export_manifest(
            output,
            variant=variant,
            winner=winner,
            device=device,
            export_strategy=args.export_strategy,
        )
        print(
            json.dumps(
                {
                    "event": "export_complete",
                    "variant": variant,
                    "path": str(output),
                    "manifest": str(export_manifest),
                    "manifest_sha256": sha256(export_manifest),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if export_failures:
        raise RuntimeError("Export failure(s): " + "; ".join(export_failures))

    workflow_report = {
        "schema_version": 1,
        "status": "PASS",
        "search_journal": str(shared_stage.journal),
        "search_journal_sha256": sha256(shared_stage.journal),
        "finalist_report": str(winners_path),
        "finalist_report_sha256": sha256(winners_path),
        "exports": {
            variant: str(export_root / variant.lower())
            for variant in ("Balanced", "Max")
        },
    }
    workflow_path = root / "heretic_moe_workflow.json"
    workflow_path.write_text(
        json.dumps(workflow_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "workflow_complete",
                "report": str(workflow_path),
                "report_sha256": sha256(workflow_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.exploration_trials <= 0 or args.target_trials <= 0:
        raise ValueError("Trial budgets must be positive")
    if args.exploration_trials % 2:
        raise ValueError("--exploration-trials must be even for a 50/50 split")
    if not args.continue_shared_only and args.target_trials <= args.exploration_trials:
        raise ValueError(
            "--target-trials must exceed the combined exploration prefix"
        )
    if args.finalist_top_n < 2:
        raise ValueError("--finalist-top-n must be at least 2")
    if args.recheck_ppl_chunks <= 0 or args.recheck_ppl_window <= 0:
        raise ValueError("Recheck PPL dimensions must be positive")
    if args.max_ppl_drift < 0:
        raise ValueError("--max-ppl-drift cannot be negative")
    if args.max_keywords < 0 or args.keyword_total <= 0:
        raise ValueError("Keyword gate values are invalid")
    if not 0 < args.balanced_removal_fraction <= 1:
        raise ValueError("--balanced-removal-fraction must be in (0, 1]")

    source_base_config = args.base_config.resolve()
    executable_value = args.heretic or shutil.which("heretic")
    if not executable_value:
        raise FileNotFoundError(
            "No Heretic executable found in PATH; provide --heretic explicitly"
        )
    executable = Path(executable_value).resolve()
    root = args.run_root.resolve()
    if not source_base_config.is_file():
        raise FileNotFoundError(source_base_config)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    controller_path = Path(__file__).resolve()
    repository = controller_path.parents[2]
    git_revision, git_dirty = git_provenance(repository)
    args.controller_path = controller_path
    args.controller_sha256 = sha256(controller_path)
    args.heretic_path = executable
    args.heretic_sha256 = sha256(executable)
    args.git_revision = git_revision
    args.git_dirty = git_dirty
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    base = read_config(source_base_config)
    base_config = source_base_config
    if args.data_root:
        base = apply_data_root(base, args.data_root)
    if args.model:
        base["model"] = args.model
    if args.model or args.data_root:
        if not args.dry_run:
            base_config = root / "effective_base_config.toml"
            write_managed_config(base_config, base, dry_run=False)
    branch_trials = args.exploration_trials // 2
    devices = assigned_devices(args)
    shared_worker_count = len(devices)
    response_archive = root / "trial-responses.sqlite3"
    scorer_updates = (
        frozenset({"scorer"})
        if args.allow_scorer_config_update
        else frozenset()
    )
    random_stage = build_stage(
        root,
        "random_branch",
        base,
        n_trials=branch_trials,
        n_startup_trials=branch_trials,
        startup_design="random",
        device=args.random_device,
        response_archive=response_archive,
        response_number_offset=0,
        response_number_stride=2,
        parallel_workers=1,
        dry_run=args.dry_run,
        allowed_config_updates=scorer_updates,
    )
    sobol_stage = build_stage(
        root,
        "sobol_branch",
        base,
        n_trials=branch_trials,
        n_startup_trials=branch_trials,
        startup_design="sobol",
        device=args.sobol_device,
        response_archive=response_archive,
        response_number_offset=1,
        response_number_stride=2,
        parallel_workers=1,
        dry_run=args.dry_run,
        allowed_config_updates=scorer_updates,
    )
    shared_stage = build_stage(
        root,
        "shared_tpe",
        base,
        n_trials=args.target_trials,
        n_startup_trials=0,
        startup_design="random",
        device=args.random_device,
        response_archive=response_archive,
        response_number_offset=0,
        response_number_stride=1,
        parallel_workers=shared_worker_count,
        dry_run=args.dry_run,
        allowed_config_updates=frozenset({"n_trials"}) | scorer_updates,
    )
    manifest_path = root / "adaptive_run_manifest.json"
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=[random_stage, sobol_stage, shared_stage],
            status=(
                "tpe_preparing"
                if args.continue_shared_only
                else "exploration_running"
            ),
        )

    if not args.continue_shared_only:
        if not args.sequential_exploration and len(devices) > 1:
            random_process = start_stage(
                random_stage,
                executable,
                dry_run=args.dry_run,
                visible_worker_window=args.visible_worker_windows,
            )
            sobol_process = start_stage(
                sobol_stage,
                executable,
                dry_run=args.dry_run,
                visible_worker_window=args.visible_worker_windows,
            )
            if args.dry_run:
                random_process.wait()
                sobol_process.wait()
            else:
                wait_parallel(
                    [
                        (random_stage, random_process),
                        (sobol_stage, sobol_process),
                    ]
                )
        else:
            run_stage(
                random_stage,
                executable,
                dry_run=args.dry_run,
                visible_worker_window=args.visible_worker_windows,
            )
            run_stage(
                sobol_stage,
                executable,
                dry_run=args.dry_run,
                visible_worker_window=args.visible_worker_windows,
            )
        if not args.dry_run:
            require_trial_count(random_stage, branch_trials)
            require_trial_count(sobol_stage, branch_trials)
        merge_branches(
            random_stage,
            sobol_stage,
            shared_stage,
            target_trials=args.target_trials,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            write_run_manifest(
                manifest_path,
                args=args,
                base_config=base_config,
                stages=[random_stage, sobol_stage, shared_stage],
                status="exploration_merged",
            )
    elif not args.dry_run and not shared_stage.journal.is_file():
        raise FileNotFoundError(
            f"No shared journal to continue: {shared_stage.journal}"
        )

    if args.dry_run:
        completed_trials = args.exploration_trials
    else:
        completed_trials, waiting_trials = journal_trial_counts(shared_stage.journal)
    if completed_trials > args.target_trials:
        raise ValueError(
            f"Shared study already has {completed_trials} trials, above target "
            f"{args.target_trials}"
        )
    # WAITING trials already occupy trial numbers but still need one optimization
    # call each. Add them back so queued remeasurements do not silently reduce the
    # requested number of actual evaluations.
    remaining_trials = args.target_trials - completed_trials + (
        0 if args.dry_run else waiting_trials
    )
    if not args.dry_run:
        require_constraint_metadata(shared_stage)
    worker_budgets = split_worker_budget(remaining_trials, shared_worker_count)
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=[random_stage, sobol_stage, shared_stage],
            status="tpe_running" if remaining_trials else "complete",
        )
    worker_seed_base = int(base.get("seed") or 0) + 10_000
    worker_processes: list[tuple[Stage, subprocess.Popen]] = []
    for worker_index, (device, budget) in enumerate(
        zip(devices, worker_budgets, strict=True)
    ):
        if budget == 0:
            continue
        display_name = f"shared_tpe_gpu{device}"
        worker_stage = Stage(
            display_name,
            shared_stage.directory,
            shared_stage.config,
            shared_stage.journal,
            device,
        )
        process = start_stage(
            worker_stage,
            executable,
            dry_run=args.dry_run,
            display_name=display_name,
            device=device,
            command_args=(
                "--n-trials",
                str(args.target_trials),
                "--n-startup-trials",
                "0",
                "--parallel-workers",
                str(shared_worker_count),
                "--worker-trial-budget",
                str(budget),
                f"--seed={worker_seed_base + worker_index}",
            ),
            visible_worker_window=args.visible_worker_windows,
        )
        worker_processes.append((worker_stage, process))
    if args.dry_run:
        for _, process in worker_processes:
            process.wait()
    else:
        wait_parallel(worker_processes)
        final_trial_count = journal_trial_count(shared_stage.journal)
        if final_trial_count != args.target_trials:
            raise RuntimeError(
                f"Shared study ended at {final_trial_count}, expected "
                f"{args.target_trials}"
            )
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=[random_stage, sobol_stage, shared_stage],
            status="complete",
        )
    if args.finalize:
        finalize_and_export(
            args,
            root=root,
            base_config=base_config,
            shared_stage=shared_stage,
            executable=executable,
        )
        if not args.dry_run:
            write_run_manifest(
                manifest_path,
                args=args,
                base_config=base_config,
                stages=[random_stage, sobol_stage, shared_stage],
                status="release_complete",
            )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        mark_existing_run_failed(sys.argv[1:], error)
        raise
