#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the bounded Random/Sobol -> shared TPE HereticMOE workflow.

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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import tomli_w
import tomllib
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from heretic.work_queue import TrialWorkQueue

_OUTPUT_PUMPS: dict[int, threading.Thread] = {}


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
            "Run a bounded alternating Random/Sobol prefix and continue the same "
            "shared journal with multivariate TPE."
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
        help="HereticMOE executable (default: discover hereticMOE in PATH).",
    )
    parser.add_argument(
        "--exploration-trials",
        type=int,
        default=120,
        help=(
            "Total exploration prefix across the shared worker queue (default: "
            "120). Tasks alternate evenly between Random and scrambled Sobol."
        ),
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=600,
        help=(
            "Exact work-permit target (default: 600). Pass a larger value to "
            "extend a compatible shared journal."
        ),
    )
    parser.add_argument("--random-device", default="0")
    parser.add_argument("--sobol-device", default="1")
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated physical GPU indices used by the shared TPE queue. "
            "When omitted, random-device and sobol-device are used."
        ),
    )
    parser.add_argument(
        "--dynamic-worker-queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Let each resident GPU worker claim the next available trial instead "
            "of assigning fixed per-device budgets."
        ),
    )
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
    post_search = parser.add_mutually_exclusive_group()
    post_search.add_argument(
        "--finalize",
        dest="post_search_mode",
        action="store_const",
        const="export",
        default="export",
        help=(
            "After the search, remeasure the TOP-N finalists at higher PPL "
            "fidelity, select Pareto-valid Balanced and Max roles, and export both. "
            "Both roles may resolve to one winner when every alternative is dominated."
        ),
    )
    post_search.add_argument(
        "--no-finalize",
        "--search-only",
        dest="post_search_mode",
        action="store_const",
        const="none",
        help="Stop at --target-trials without recheck or model export.",
    )
    post_search.add_argument(
        "--recheck-only",
        dest="post_search_mode",
        action="store_const",
        const="recheck",
        help=(
            "After the search, remeasure and select the TOP-N finalists but "
            "do not assemble or export model weights."
        ),
    )
    parser.add_argument("--finalist-top-n", type=int, default=6)
    parser.add_argument(
        "--finalist-selection-policy",
        choices=("pareto", "feasible_lexicographic", "feasible_diverse", "feasible_cost"),
        default="feasible_cost",
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
        "--keyword-near-gate-extra",
        type=int,
        default=1,
        help=(
            "If no finalist passes the strict keyword gate, permit at most this "
            "many additional matches and record the exception in winners.json."
        ),
    )
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
        default=0.0,
        help=(
            "Minimum fraction of the best SRG improvement required from the "
            "Balanced finalist. The default 0 accepts any genuine improvement "
            "over the original baseline."
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
    args = parser.parse_args()
    args.finalize = args.post_search_mode == "export"
    args.recheck_only = args.post_search_mode == "recheck"
    return args


def post_search_completion_status(mode: str) -> str:
    """Return the durable manifest status for one completed post-search mode."""

    if mode == "export":
        return "release_complete"
    if mode == "recheck":
        return "recheck_complete"
    if mode == "none":
        return "complete"
    raise ValueError(f"Unknown post-search mode: {mode}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prefix(path: Path, size_bytes: int) -> str | None:
    """Hash exactly one immutable file prefix, or return None if truncated."""

    digest = hashlib.sha256()
    remaining = size_bytes
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                return None
            digest.update(chunk)
            remaining -= len(chunk)
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
    environment["HERETIC_MOE_INTERNAL"] = "1"
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Child processes normally inherit a redirected controller stdout on CI and
    # through orchestration tools. Force line-by-line progress instead of
    # releasing several minutes of output only when a stage exits.
    environment["PYTHONUNBUFFERED"] = "1"
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
        environment["HERETIC_WORKER_LABEL"] = f"GPU {device}"
        cache_suffix = f"gpu-{device}"
        for variable in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            if base_cache := environment.get(variable):
                environment[variable] = str(Path(base_cache) / cache_suffix)
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
    environment = process_environment(effective_device)
    if os.environ.get("HERETIC_SUPERVISED") != "1":
        return subprocess.Popen(command, cwd=stage.directory, env=environment)

    process = subprocess.Popen(
        command,
        cwd=stage.directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def forward_output() -> None:
        assert process.stdout is not None
        prefix = f"GPU {effective_device} | "
        for line in process.stdout:
            print(prefix + line.rstrip("\r\n"), flush=True)

    pump = threading.Thread(
        target=forward_output,
        name=f"output-{effective_name}",
        daemon=True,
    )
    pump.start()
    _OUTPUT_PUMPS[id(process)] = pump
    return process


def wait_stage(stage: Stage, process: subprocess.Popen) -> None:
    return_code = process.wait()
    finish_output_pump(process)
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


def finish_output_pump(process: subprocess.Popen) -> None:
    """Drain and join a supervised worker's output forwarding thread."""

    pump = _OUTPUT_PUMPS.pop(id(process), None)
    if pump is not None:
        pump.join()


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


def fail_running_trials_for_worker(journal: Path, worker_id: str) -> list[int]:
    """Turn orphaned trials from a dead queue worker into terminal failures."""

    if not journal.is_file():
        return []
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
    failed: list[int] = []
    for trial in study.get_trials(deepcopy=False):
        if (
            trial.state == optuna.trial.TrialState.RUNNING
            and trial.user_attrs.get("queue_worker_id") == worker_id
        ):
            trial_id = storage.get_trial_id_from_study_id_trial_number(
                study._study_id,
                trial.number,
            )
            if storage.set_trial_state_values(
                trial_id,
                optuna.trial.TrialState.FAIL,
            ):
                failed.append(trial.number)
    return failed


def _monitor_dynamic_workers(
    workers: list[tuple[Stage, subprocess.Popen, str, tuple[str, ...], int]],
    *,
    queue: TrialWorkQueue,
    executable: Path,
    expected_tasks: int,
    visible_worker_window: bool,
    max_restarts_per_gpu: int = 2,
) -> list[dict[str, Any]]:
    """Monitor queue workers concurrently and recover abandoned claims."""

    active = workers
    recoveries: list[dict[str, Any]] = []
    while active:
        next_active: list[
            tuple[Stage, subprocess.Popen, str, tuple[str, ...], int]
        ] = []
        changed = False
        for stage, process, worker_id, command_args, restart_count in active:
            return_code = process.poll()
            if return_code is None:
                next_active.append(
                    (stage, process, worker_id, command_args, restart_count)
                )
                continue

            changed = True
            finish_output_pump(process)
            if return_code == 0:
                print(
                    json.dumps(
                        {
                            "event": "worker_complete",
                            "worker_id": worker_id,
                            "device": stage.device,
                            "restarts": restart_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue

            orphaned_trials = fail_running_trials_for_worker(
                stage.journal,
                worker_id,
            )
            released_tasks = queue.release_worker(worker_id)
            recovery = {
                "event": "worker_failure",
                "worker_id": worker_id,
                "device": stage.device,
                "exit_code": return_code,
                "orphaned_trials_failed": orphaned_trials,
                "released_tasks": released_tasks,
                "restart_count": restart_count,
            }
            recoveries.append(recovery)
            print(json.dumps(recovery, sort_keys=True), flush=True)

            queue_stats = queue.stats()
            unfinished = queue_stats.pending or queue_stats.claimed
            if unfinished and restart_count < max_restarts_per_gpu:
                replacement = start_stage(
                    stage,
                    executable,
                    dry_run=False,
                    display_name=stage.name,
                    device=stage.device,
                    command_args=command_args,
                    visible_worker_window=visible_worker_window,
                )
                next_active.append(
                    (
                        stage,
                        replacement,
                        worker_id,
                        command_args,
                        restart_count + 1,
                    )
                )
                print(
                    json.dumps(
                        {
                            "event": "worker_restarted",
                            "worker_id": worker_id,
                            "device": stage.device,
                            "restart_count": restart_count + 1,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        active[:] = next_active
        stats = queue.stats()
        if not active and (
            stats.pending
            or stats.claimed
            or stats.failed
            or stats.complete != expected_tasks
        ):
            raise RuntimeError(
                "All dynamic workers exited before the queue completed: "
                f"{stats}"
            )
        if active and not changed:
            time.sleep(0.25)
    return recoveries


def wait_dynamic_workers(
    workers: list[tuple[Stage, subprocess.Popen, str, tuple[str, ...], int]],
    *,
    queue: TrialWorkQueue,
    executable: Path,
    expected_tasks: int,
    visible_worker_window: bool,
    max_restarts_per_gpu: int = 2,
) -> list[dict[str, Any]]:
    """Monitor workers and guarantee child cleanup if the controller exits."""

    try:
        return _monitor_dynamic_workers(
            workers,
            queue=queue,
            executable=executable,
            expected_tasks=expected_tasks,
            visible_worker_window=visible_worker_window,
            max_restarts_per_gpu=max_restarts_per_gpu,
        )
    except BaseException:
        for _, process, _, _, _ in workers:
            if process.poll() is None:
                process.terminate()
        for stage, process, worker_id, _, _ in workers:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            finish_output_pump(process)
            orphaned_trials = fail_running_trials_for_worker(
                stage.journal,
                worker_id,
            )
            released_tasks = queue.release_worker(worker_id)
            print(
                json.dumps(
                    {
                        "event": "worker_shutdown_cleanup",
                        "worker_id": worker_id,
                        "device": stage.device,
                        "orphaned_trials_failed": orphaned_trials,
                        "released_tasks": released_tasks,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        raise


def journal_trial_count(journal: Path) -> int:
    total, _ = journal_trial_counts(journal)
    return total


def journal_trial_counts(journal: Path) -> tuple[int, int]:
    """Return total and queued-WAITING trial counts for a shared journal."""

    if journal.is_file() and journal.stat().st_size == 0:
        return 0, 0

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


def controller_trial_counts(
    journal: Path,
    *,
    dry_run: bool,
    continue_shared_only: bool,
    dynamic_worker_queue: bool,
    exploration_trials: int,
) -> tuple[int, int]:
    """Resolve existing work without making a resume dry-run invent workers."""

    if journal.is_file() and (not dry_run or continue_shared_only):
        return journal_trial_counts(journal)
    if dry_run:
        completed = (
            0
            if dynamic_worker_queue and not continue_shared_only
            else exploration_trials
        )
        return completed, 0
    return 0, 0


def load_journal_trials(journal: Path) -> list[optuna.trial.FrozenTrial]:
    """Load one journal's text-free Optuna trial metadata."""

    if journal.is_file() and journal.stat().st_size == 0:
        return []

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
    return list(study.get_trials(deepcopy=False))


def verify_queue_against_journal(
    queue: TrialWorkQueue,
    journal: Path,
    *,
    target_trial_count: int,
    tpe_concurrency: int,
) -> tuple[bool, str]:
    """Verify that every durable permit agrees with explicit journal metadata.

    Trial counts alone are insufficient: a previous buggy/non-queue worker can
    advance the same journal while leaving every queue row pending. Queue task
    and attempt attributes form the reconciliation key. Stale queues are never
    rewritten in place; the caller preserves them and allocates a new version.
    """

    try:
        contract = queue.contract()
        records = queue.task_records()
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"unreadable_contract:{type(error).__name__}:{error}"
    if contract.schema_version != 3:
        return False, f"schema_version:{contract.schema_version}"
    if contract.target_trial_count != target_trial_count:
        return False, (
            f"target:{contract.target_trial_count}!={target_trial_count}"
        )
    if contract.last_task_id_exclusive != target_trial_count:
        return False, (
            "last_task_id_exclusive:"
            f"{contract.last_task_id_exclusive}!={target_trial_count}"
        )
    if contract.tpe_concurrency != tpe_concurrency:
        return False, (
            f"tpe_concurrency:{contract.tpe_concurrency}!={tpe_concurrency}"
        )
    if contract.first_task_id < 0 or contract.task_count < 0:
        return False, "negative_task_range"
    if not (
        contract.first_task_id
        <= contract.journal_base_trial_count
        <= contract.target_trial_count
    ):
        return False, (
            "journal_base_trial_count:"
            f"{contract.journal_base_trial_count}"
        )
    if contract.journal_base_size_bytes < 0:
        return False, f"journal_base_size_bytes:{contract.journal_base_size_bytes}"
    if not 0 <= contract.exploration_task_count <= contract.task_count:
        return False, (
            "exploration_task_count:"
            f"{contract.exploration_task_count}/{contract.task_count}"
        )
    if contract.tpe_concurrency <= 0:
        return False, f"invalid_tpe_concurrency:{contract.tpe_concurrency}"
    if (
        contract.first_task_id + contract.task_count
        != contract.last_task_id_exclusive
    ):
        return False, "task_count_range_mismatch"
    if len(records) != contract.task_count:
        return False, f"task_count:{len(records)}!={contract.task_count}"
    expected_ids = list(
        range(contract.first_task_id, contract.last_task_id_exclusive)
    )
    if [record.task_id for record in records] != expected_ids:
        return False, "task_id_range_mismatch"
    for offset, record in enumerate(records):
        expected_kind = (
            "random"
            if offset < contract.exploration_task_count and offset % 2 == 0
            else (
                "sobol"
                if offset < contract.exploration_task_count
                else "tpe"
            )
        )
        if record.task_kind != expected_kind:
            return False, (
                f"task_kind:{record.task_id}:{record.task_kind}!={expected_kind}"
            )

    if not journal.is_file():
        empty_sha256 = hashlib.sha256().hexdigest()
        pristine = (
            contract.first_task_id == 0
            and contract.journal_base_trial_count == 0
            and contract.journal_base_size_bytes == 0
            and contract.journal_base_sha256 == empty_sha256
            and all(
                record.state == "pending"
                and record.attempt == 0
                and record.worker_id is None
                and record.trial_number is None
                and record.trial_state is None
                for record in records
            )
        )
        return (
            (True, "verified_pristine_without_journal")
            if pristine
            else (False, "queue_without_verifiable_journal")
        )

    if journal.stat().st_size < contract.journal_base_size_bytes:
        return False, (
            f"journal_bytes_truncated:{journal.stat().st_size}<"
            f"{contract.journal_base_size_bytes}"
        )
    base_sha256 = sha256_prefix(journal, contract.journal_base_size_bytes)
    if base_sha256 != contract.journal_base_sha256:
        return False, (
            f"journal_base_sha256:{base_sha256}!="
            f"{contract.journal_base_sha256}"
        )
    trials = load_journal_trials(journal)
    if len(trials) < contract.journal_base_trial_count:
        return False, (
            f"journal_truncated:{len(trials)}<"
            f"{contract.journal_base_trial_count}"
        )
    if [trial.number for trial in trials[: contract.journal_base_trial_count]] != list(
        range(contract.journal_base_trial_count)
    ):
        return False, "journal_base_trial_numbers_mismatch"
    by_attempt: dict[tuple[int, int], optuna.trial.FrozenTrial] = {}
    unmanaged: list[int] = []
    attempts_by_task: dict[int, list[optuna.trial.FrozenTrial]] = {}
    terminal = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    }
    for trial in trials:
        raw_task_id = trial.user_attrs.get("queue_task_id")
        raw_attempt = trial.user_attrs.get("queue_attempt")
        if raw_task_id is None:
            if trial.number >= contract.journal_base_trial_count:
                unmanaged.append(trial.number)
            continue
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError):
            return False, f"invalid_trial_queue_key:T{trial.number}"
        if not contract.first_task_id <= task_id < contract.last_task_id_exclusive:
            if trial.number < contract.journal_base_trial_count:
                continue
            return False, f"journal_task_out_of_range:T{trial.number}:{task_id}"
        try:
            attempt = int(raw_attempt)
        except (TypeError, ValueError):
            return False, f"invalid_trial_queue_key:T{trial.number}"
        if attempt <= 0:
            return False, f"invalid_attempt:T{trial.number}:{attempt}"
        key = (task_id, attempt)
        if key in by_attempt:
            return False, f"duplicate_attempt:{task_id}:{attempt}"
        by_attempt[key] = trial
        attempts_by_task.setdefault(task_id, []).append(trial)
    if unmanaged:
        preview = ",".join(str(number) for number in unmanaged[:8])
        return False, f"unmanaged_journal_trials:{preview}"

    for record in records:
        attempts = attempts_by_task.get(record.task_id, [])
        for trial in attempts:
            attempt = int(trial.user_attrs["queue_attempt"])
            if attempt > record.attempt:
                return False, f"future_attempt:{record.task_id}"
            if trial.user_attrs.get("queue_task_kind") != record.task_kind:
                return False, f"trial_task_kind_mismatch:{record.task_id}:{attempt}"
            if attempt < record.attempt and trial.state not in terminal:
                return False, (
                    f"nonterminal_prior_attempt:{record.task_id}:{attempt}"
                )
        current = by_attempt.get((record.task_id, record.attempt))
        if (
            current is not None
            and record.state in {"claimed", "complete", "failed"}
            and current.user_attrs.get("queue_worker_id") != record.worker_id
        ):
            return False, f"trial_worker_mismatch:{record.task_id}"
        if record.state == "complete":
            if current is None or current.state not in terminal:
                return False, f"complete_without_terminal:{record.task_id}"
            if current.number != record.trial_number:
                return False, f"trial_number_mismatch:{record.task_id}"
            if current.state.name != record.trial_state:
                return False, f"trial_state_mismatch:{record.task_id}"
        elif record.state == "claimed":
            if current is None or current.state != optuna.trial.TrialState.RUNNING:
                return False, f"claim_without_running:{record.task_id}"
        elif record.state == "pending":
            if current is not None and current.state in {
                optuna.trial.TrialState.RUNNING,
                optuna.trial.TrialState.WAITING,
                optuna.trial.TrialState.COMPLETE,
            }:
                return False, f"pending_has_live_or_complete:{record.task_id}"
        elif record.state == "failed":
            if current is None or current.state not in terminal:
                return False, f"failed_without_terminal:{record.task_id}"
        else:
            return False, f"unknown_task_state:{record.task_id}:{record.state}"
    return True, "verified"


def queue_path_for_version(root: Path, target: int, version: int) -> Path:
    suffix = "" if version == 1 else f".v{version}"
    return root / f"trial-work-queue-{target}{suffix}.sqlite3"


def select_verified_queue(
    root: Path,
    journal: Path,
    *,
    target_trial_count: int,
    tpe_concurrency: int,
) -> tuple[TrialWorkQueue | None, Path | None, list[dict[str, Any]]]:
    """Select a matching immutable queue; report but never mutate stale ones."""

    rejected: list[dict[str, Any]] = []
    version = 1
    while True:
        path = queue_path_for_version(root, target_trial_count, version)
        if not path.is_file():
            return None, path, rejected
        queue = TrialWorkQueue(path)
        valid, reason = verify_queue_against_journal(
            queue,
            journal,
            target_trial_count=target_trial_count,
            tpe_concurrency=tpe_concurrency,
        )
        if valid:
            return queue, path, rejected
        record = {
            "event": "stale_queue_preserved",
            "path": str(path),
            "reason": reason,
        }
        rejected.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        version += 1


def missing_constraint_metadata(trials: list[Any]) -> list[int]:
    """Return TPE/legacy trials missing Optuna constraint system metadata."""

    exploration_kinds = {"random", "sobol"}
    return [
        int(trial.number)
        for trial in trials
        if trial.user_attrs.get("queue_task_kind") not in exploration_kinds
        and "constraints" not in trial.system_attrs
    ]


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
    required = [
        trial
        for trial in complete
        if trial.user_attrs.get("queue_task_kind") not in {"random", "sobol"}
    ]
    missing = missing_constraint_metadata(complete)
    if missing:
        preview = ", ".join(str(number) for number in missing[:8])
        raise RuntimeError(
            "Constraint metadata incomplete: "
            f"{len(required) - len(missing)}/{len(required)} required trials; "
            f"missing trials: {preview}"
        )
    print(
        f"Constraints: {len(required)}/{len(required)} required trials OK "
        f"({len(complete) - len(required)} exploration trials exempt)",
        flush=True,
    )


def should_require_constraint_metadata(
    *,
    dry_run: bool,
    journal_has_trials: bool,
    remaining_trials: int,
) -> bool:
    """Constraint backfill is a search-resume guard, not a recheck prerequisite."""

    return not dry_run and journal_has_trials and remaining_trials > 0


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

    if args.devices:
        requested = [part.strip() for part in str(args.devices).split(",")]
        if any(not part for part in requested):
            raise ValueError("--devices contains an empty GPU index")
        devices = list(dict.fromkeys(requested))
    else:
        devices = list(
            dict.fromkeys((str(args.random_device), str(args.sobol_device)))
        )
    if not devices:
        raise ValueError("At least one worker device is required")
    return devices


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
        "dynamic_worker_queue": args.dynamic_worker_queue,
        "worker_queue": getattr(args, "worker_queue_path", None),
        "worker_queue_rejections": getattr(args, "worker_queue_rejections", []),
        "worker_queue_tpe_concurrency": len(devices),
        "worker_recoveries": getattr(args, "worker_recoveries", []),
        "visible_worker_windows": args.visible_worker_windows,
        "continue_shared_only": args.continue_shared_only,
        "post_search_mode": args.post_search_mode,
        "finalize": args.finalize,
        "recheck_only": args.recheck_only,
        "finalist_top_n": args.finalist_top_n,
        "finalist_selection_policy": args.finalist_selection_policy,
        "finalization_version": getattr(args, "finalization_version", None),
        "finalist_dir": getattr(args, "resolved_finalist_dir", None),
        "finalization_contract": getattr(args, "finalization_contract", None),
        "finalization_contract_sha256": getattr(
            args, "finalization_contract_sha256", None
        ),
        "recheck_ppl": {
            "chunks": args.recheck_ppl_chunks,
            "window": args.recheck_ppl_window,
        },
        "export_root": getattr(
            args,
            "resolved_export_root",
            str((args.export_root or path.parent / "exports").resolve()),
        ),
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


def export_is_complete(
    directory: Path,
    *,
    variant: str,
    winner: dict[str, Any],
    export_strategy: str,
) -> bool:
    manifest_path = directory / "heretic_moe_export.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("export_kind", "physical") != "physical"
        or manifest.get("variant") != variant
        or manifest.get("winner") != winner
        or manifest.get("export_strategy") != export_strategy
    ):
        return False
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False
    if not any(
        isinstance(record, dict)
        and str(record.get("path", "")).endswith(".safetensors")
        for record in files
    ):
        return False
    root = directory.resolve()
    try:
        for record in files:
            if not isinstance(record, dict):
                return False
            path = (directory / str(record["path"])).resolve()
            if not path.is_relative_to(root):
                return False
            if not path.is_file() or path.stat().st_size != int(record["bytes"]):
                return False
            if sha256(path) != record["sha256"]:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def winners_share_physical_export(winners: dict[str, Any]) -> bool:
    """Return true only when both release roles identify one exact intervention."""

    try:
        balanced = winners["Balanced"]
        maximum = winners["Max"]
        return (
            int(balanced["source_trial_index"])
            == int(maximum["source_trial_index"])
            and str(balanced["params_sha256"])
            == str(maximum["params_sha256"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def export_alias_is_complete(
    directory: Path,
    *,
    variant: str,
    winner: dict[str, Any],
    export_strategy: str,
    physical_directory: Path,
    physical_variant: str,
    physical_winner: dict[str, Any],
) -> bool:
    """Validate a weight-free release role bound to one physical export."""

    manifest_path = directory / "heretic_moe_export.json"
    physical_manifest = physical_directory / "heretic_moe_export.json"
    if not manifest_path.is_file() or not physical_manifest.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_physical = (directory / str(manifest["physical_export"])).resolve()
        recorded_manifest = (
            directory / str(manifest["physical_manifest"])
        ).resolve()
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("export_kind") != "role_alias"
        or manifest.get("variant") != variant
        or manifest.get("winner") != winner
        or manifest.get("export_strategy") != export_strategy
        or manifest.get("physical_variant") != physical_variant
        or manifest.get("physical_winner") != physical_winner
        or recorded_physical != physical_directory.resolve()
        or recorded_manifest != physical_manifest.resolve()
        or manifest.get("physical_manifest_sha256") != sha256(physical_manifest)
    ):
        return False
    return export_is_complete(
        physical_directory,
        variant=physical_variant,
        winner=physical_winner,
        export_strategy=export_strategy,
    )


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
        "export_kind": "physical",
        "variant": variant,
        "device": device,
        "export_strategy": export_strategy,
        "winner": winner,
        "files": files,
    }
    path = directory / "heretic_moe_export.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_export_alias_manifest(
    directory: Path,
    *,
    variant: str,
    winner: dict[str, Any],
    export_strategy: str,
    physical_directory: Path,
    physical_variant: str,
    physical_winner: dict[str, Any],
) -> Path:
    """Write a small immutable role alias without duplicating model weights."""

    physical_manifest = physical_directory / "heretic_moe_export.json"
    if not export_is_complete(
        physical_directory,
        variant=physical_variant,
        winner=physical_winner,
        export_strategy=export_strategy,
    ):
        raise RuntimeError(
            f"Cannot alias incomplete physical export {physical_variant}: "
            f"{physical_directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    relative_physical = Path(
        os.path.relpath(physical_directory.resolve(), directory.resolve())
    ).as_posix()
    relative_manifest = Path(
        os.path.relpath(physical_manifest.resolve(), directory.resolve())
    ).as_posix()
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "export_kind": "role_alias",
        "variant": variant,
        "export_strategy": export_strategy,
        "winner": winner,
        "physical_variant": physical_variant,
        "physical_winner": physical_winner,
        "physical_export": relative_physical,
        "physical_manifest": relative_manifest,
        "physical_manifest_sha256": sha256(physical_manifest),
    }
    path = directory / "heretic_moe_export.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def expected_export_roles(
    export_root: Path,
    winners: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Describe physical model payloads and weight-free role aliases."""

    balanced_path = export_root / "balanced"
    max_path = export_root / "max"
    if winners_share_physical_export(winners):
        return {
            "Balanced": {
                "kind": "physical_export",
                "path": str(balanced_path),
            },
            "Max": {
                "kind": "role_alias",
                "path": str(max_path),
                "physical_variant": "Balanced",
                "physical_path": str(balanced_path),
            },
        }
    return {
        "Balanced": {"kind": "physical_export", "path": str(balanced_path)},
        "Max": {"kind": "physical_export", "path": str(max_path)},
    }


def expected_collapsed_export_policy(
    export_root: Path,
    winners: dict[str, Any],
    recheck_contract: Path,
) -> dict[str, Any] | None:
    """Bind a storage-only role collapse to immutable recheck evidence."""

    if not winners_share_physical_export(winners):
        return None
    balanced_manifest = export_root / "balanced" / "heretic_moe_export.json"
    max_manifest = export_root / "max" / "heretic_moe_export.json"
    if not (
        balanced_manifest.is_file()
        and max_manifest.is_file()
        and recheck_contract.is_file()
    ):
        return None
    return {
        "schema_version": 1,
        "status": "PASS",
        "policy": "collapsed_roles_single_physical_export",
        "immutable_recheck_contract": {
            "path": str(recheck_contract),
            "sha256": sha256(recheck_contract),
        },
        "winners": winners,
        "physical_export": {
            "variant": "Balanced",
            "path": str(export_root / "balanced"),
            "manifest": str(balanced_manifest),
            "manifest_sha256": sha256(balanced_manifest),
        },
        "role_alias": {
            "variant": "Max",
            "path": str(export_root / "max"),
            "manifest": str(max_manifest),
            "manifest_sha256": sha256(max_manifest),
            "physical_variant": "Balanced",
        },
    }


def write_collapsed_export_policy(
    export_root: Path,
    winners: dict[str, Any],
    recheck_contract: Path,
) -> Path:
    """Write or verify the separate storage correction for collapsed roles."""

    expected = expected_collapsed_export_policy(
        export_root,
        winners,
        recheck_contract,
    )
    if expected is None:
        raise RuntimeError("Collapsed export policy has incomplete bound artifacts")
    path = export_root / "export_policy.json"
    current = load_json_object(path)
    if current is not None:
        if current != expected:
            raise RuntimeError(f"Export policy changed unexpectedly: {path}")
        return path
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite invalid export policy: {path}")
    write_text_atomic(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return path


def versioned_directory(base: Path, version: int) -> Path:
    if version == 1:
        return base
    return base.with_name(f"{base.name}_v{version}")


def versioned_file(base: Path, version: int) -> Path:
    if version == 1:
        return base
    return base.with_name(f"{base.stem}_v{version}{base.suffix}")


def load_finalization_overrides_contract(
    root: Path,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Load and fingerprint the run-local winner-policy override, if present."""

    path = root / "finalization_overrides.json"
    if not path.is_file():
        return {}, None
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 1:
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
    return record, {"path": str(path.resolve()), "sha256": sha256(path)}


def build_finalization_contract(
    args: argparse.Namespace,
    *,
    root: Path,
    source_journal_sha256: str,
    base_config_sha256: str,
) -> dict[str, Any]:
    """Build the complete immutable recheck and export reuse fingerprint."""

    overrides, override_fingerprint = load_finalization_overrides_contract(root)
    balanced_srg_gate = overrides.get(
        "balanced_srg_gate", args.balanced_srg_gate
    )
    baseline_srg_override = overrides.get("baseline_srg")
    removal_fraction = float(
        overrides.get(
            "balanced_removal_fraction",
            args.balanced_removal_fraction,
        )
    )
    return {
        "schema_version": 1,
        "source_journal_sha256": source_journal_sha256,
        "base_config_sha256": base_config_sha256,
        "top_n": args.finalist_top_n,
        "selection_policy": args.finalist_selection_policy,
        "ppl": {
            "chunks": args.recheck_ppl_chunks,
            "window": args.recheck_ppl_window,
        },
        "gates": {
            "max_ppl_drift": args.max_ppl_drift,
            "max_keyword_rate": args.max_keywords / args.keyword_total,
            "max_keywords": args.max_keywords,
            "keyword_total": args.keyword_total,
            "keyword_near_gate_extra": args.keyword_near_gate_extra,
            "balanced_srg_gate": (
                None
                if balanced_srg_gate is None
                else float(balanced_srg_gate)
            ),
            "balanced_removal_fraction": removal_fraction,
            "baseline_srg_override": (
                None
                if baseline_srg_override is None
                else float(baseline_srg_override)
            ),
        },
        "overrides": override_fingerprint,
        "export": {
            "root": str((args.export_root or root / "exports").resolve()),
            "strategy": args.export_strategy,
            "variants": ["Balanced", "Max"],
            # This is the immutable recheck-era release contract. A later
            # weight-storage correction for collapsed roles is recorded
            # separately under the versioned export root, never by rewriting
            # an already frozen finalist contract.
            "distinct_winners": True,
            "heretic_executable": str(args.heretic_path),
            "heretic_executable_sha256": args.heretic_sha256,
        },
    }


def finalization_manifest_matches(
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> bool:
    try:
        top_n = int(manifest.get("top_n", -1))
        manifest_gates = manifest["gates"]
        expected_gates = contract["gates"]
        if not isinstance(manifest_gates, dict):
            return False
        gate_keys = (
            "max_ppl_drift",
            "max_keyword_rate",
            "max_keywords",
            "keyword_total",
            "keyword_near_gate_extra",
            "balanced_srg_gate",
            "balanced_removal_fraction",
        )
        gates_match = all(
            manifest_gates.get(key) == expected_gates[key] for key in gate_keys
        )
        baseline_override = expected_gates["baseline_srg_override"]
        if baseline_override is not None:
            gates_match = (
                gates_match
                and manifest_gates.get("baseline_srg") == baseline_override
            )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        manifest.get("version") == 1
        and manifest.get("status") == "prepared"
        and manifest.get("source_journal_sha256")
        == contract["source_journal_sha256"]
        and manifest.get("base_config_sha256")
        == contract["base_config_sha256"]
        and top_n == contract["top_n"]
        and manifest.get("selection_policy") == contract["selection_policy"]
        and manifest.get("ppl") == contract["ppl"]
        and manifest.get("finalization_overrides") == contract["overrides"]
        and gates_match
    )


def prepared_finalization_artifacts_match(manifest: dict[str, Any]) -> bool:
    try:
        config = Path(str(manifest["config"]))
        journal = Path(str(manifest["journal"]))
        return (
            config.is_file()
            and sha256(config) == manifest["config_sha256"]
            and journal.is_file()
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def finalization_contract_record(
    contract: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Bind the controller contract to the complete prepared selection manifest."""

    return {
        "schema_version": 1,
        "contract": contract,
        "prepared_manifest_sha256": sha256(manifest_path),
    }


def path_is_occupied(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def load_valid_winners_report(path: Path) -> dict[str, Any] | None:
    report = load_json_object(path)
    if report is None or report.get("status") != "PASS":
        return None
    contract = report.get("contract")
    if contract not in {
        "two_distinct_extremes_from_one_high_fidelity_top_n",
        "pareto_extremes_or_single_winner",
    }:
        return None
    winners = report.get("winners")
    if not isinstance(winners, dict) or set(winners) != {"Balanced", "Max"}:
        return None
    if not all(isinstance(winners[variant], dict) for variant in winners):
        return None
    try:
        balanced_number = int(winners["Balanced"]["trial_number"])
        max_number = int(winners["Max"]["trial_number"])
        balanced_source = int(winners["Balanced"]["source_trial_index"])
        max_source = int(winners["Max"]["source_trial_index"])
    except (KeyError, TypeError, ValueError):
        return None
    distinct = balanced_number != max_number and balanced_source != max_source
    if contract == "two_distinct_extremes_from_one_high_fidelity_top_n":
        if not distinct:
            return None
    elif (
        report.get("winners_distinct") is not distinct
        or not isinstance(report.get("pareto_source_trial_indices"), list)
        or balanced_source not in report["pareto_source_trial_indices"]
        or max_source not in report["pareto_source_trial_indices"]
        or report.get("expansion_recommended") is not (not distinct)
    ):
        return None
    return report


def finalization_outputs_resumable(
    args: argparse.Namespace,
    *,
    version: int,
    finalist_dir: Path,
    export_root: Path,
    workflow_path: Path,
    source_journal_sha256: str,
) -> tuple[bool, str]:
    """Reject partial or mismatched immutable outputs before choosing a version."""

    if export_root.exists() and not export_root.is_dir():
        return False, "export_root_not_directory"
    if export_root.is_dir():
        unexpected = sorted(
            path.name
            for path in export_root.iterdir()
            if path.name not in {"balanced", "max", "export_policy.json"}
        )
        if unexpected:
            return False, f"unexpected_export_entries:{','.join(unexpected[:8])}"

    winners_path = finalist_dir / "winners.json"
    winners_report = load_valid_winners_report(winners_path)
    if winners_path.exists() and winners_report is None:
        return False, "invalid_winners_report"
    winners = (
        winners_report.get("winners")
        if winners_report is not None
        else None
    )
    collapsed_roles = (
        isinstance(winners, dict) and winners_share_physical_export(winners)
    )
    complete_variants: set[str] = set()
    for variant in ("Balanced", "Max"):
        output = export_root / variant.lower()
        if not output.exists():
            continue
        if not output.is_dir():
            return False, f"export_not_directory:{variant}"
        if not any(output.iterdir()):
            continue
        if not isinstance(winners, dict) or not isinstance(
            winners.get(variant), dict
        ):
            return False, f"export_without_winner_contract:{variant}"
        if collapsed_roles and variant == "Max":
            valid_export = export_alias_is_complete(
                output,
                variant=variant,
                winner=winners[variant],
                export_strategy=args.export_strategy,
                physical_directory=export_root / "balanced",
                physical_variant="Balanced",
                physical_winner=winners["Balanced"],
            )
        else:
            valid_export = export_is_complete(
                output,
                variant=variant,
                winner=winners[variant],
                export_strategy=args.export_strategy,
            )
        if not valid_export:
            return False, f"incomplete_or_mismatched_export:{variant}"
        complete_variants.add(variant)

    export_policy_path = export_root / "export_policy.json"
    expected_policy = (
        expected_collapsed_export_policy(
            export_root,
            winners,
            finalist_dir / "finalization_contract.json",
        )
        if isinstance(winners, dict) and collapsed_roles
        else None
    )
    recorded_policy = load_json_object(export_policy_path)
    if collapsed_roles:
        if export_policy_path.exists() and recorded_policy != expected_policy:
            return False, "collapsed_export_policy_mismatch"
    elif export_policy_path.exists():
        return False, "unexpected_collapsed_export_policy"

    if not workflow_path.exists():
        return True, "resumable"
    if collapsed_roles and (
        expected_policy is None or recorded_policy != expected_policy
    ):
        return False, "workflow_without_collapsed_export_policy"
    workflow = load_json_object(workflow_path)
    if workflow is None:
        return False, "invalid_workflow_report"
    if complete_variants != {"Balanced", "Max"} or winners_report is None:
        return False, "workflow_without_complete_exports"
    expected_exports = {
        variant: str(export_root / variant.lower())
        for variant in ("Balanced", "Max")
    }
    expected_policy_reference = (
        {
            "path": str(export_policy_path),
            "sha256": sha256(export_policy_path),
        }
        if collapsed_roles and export_policy_path.is_file()
        else None
    )
    if (
        workflow.get("schema_version") != 1
        or workflow.get("status") != "PASS"
        or workflow.get("finalization_version") != version
        or workflow.get("search_journal_sha256") != source_journal_sha256
        or workflow.get("finalist_report") != str(winners_path)
        or workflow.get("finalist_report_sha256") != sha256(winners_path)
        or workflow.get("finalization_contract")
        != str(finalist_dir / "finalization_contract.json")
        or workflow.get("finalization_contract_sha256")
        != sha256(finalist_dir / "finalization_contract.json")
        or workflow.get("exports") != expected_exports
        or (
            collapsed_roles
            and workflow.get("export_roles")
            != expected_export_roles(export_root, winners_report["winners"])
        )
        or workflow.get("export_policy") != expected_policy_reference
    ):
        return False, "workflow_contract_mismatch"
    return True, "resumable_complete"


def select_finalization_paths(
    args: argparse.Namespace,
    *,
    root: Path,
    source_journal_sha256: str,
    base_config: Path,
) -> tuple[int, Path, Path, Path, dict[str, Any]]:
    """Return a resumable matching version without mutating older results."""

    base_export = (args.export_root or root / "exports").resolve()
    base_config_sha256 = sha256(base_config)
    expected_contract = build_finalization_contract(
        args,
        root=root,
        source_journal_sha256=source_journal_sha256,
        base_config_sha256=base_config_sha256,
    )
    version = 1
    while True:
        finalist_dir = versioned_directory(root / "finalist_recheck", version)
        export_root = versioned_directory(base_export, version)
        workflow_path = versioned_file(root / "heretic_moe_workflow.json", version)
        manifest_path = finalist_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = load_json_object(manifest_path)
            recorded_contract = load_json_object(
                finalist_dir / "finalization_contract.json"
            )
            expected_record = (
                finalization_contract_record(expected_contract, manifest_path)
                if manifest is not None
                else None
            )
            contract_matches = (
                manifest is not None
                and recorded_contract == expected_record
                and finalization_manifest_matches(manifest, expected_contract)
                and prepared_finalization_artifacts_match(manifest)
            )
            if contract_matches:
                resumable, reason = finalization_outputs_resumable(
                    args,
                    version=version,
                    finalist_dir=finalist_dir,
                    export_root=export_root,
                    workflow_path=workflow_path,
                    source_journal_sha256=source_journal_sha256,
                )
                if resumable:
                    return (
                        version,
                        finalist_dir,
                        export_root,
                        workflow_path,
                        expected_contract,
                    )
            else:
                reason = "source_or_contract_mismatch"
            print(
                json.dumps(
                    {
                        "event": "immutable_finalization_skipped",
                        "version": version,
                        "path": str(finalist_dir),
                        "reason": reason,
                        "recorded_source_sha256": (
                            manifest.get("source_journal_sha256")
                            if manifest is not None
                            else None
                        ),
                        "current_source_sha256": source_journal_sha256,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            version += 1
            continue
        if path_is_occupied(finalist_dir):
            print(
                json.dumps(
                    {
                        "event": "immutable_finalization_skipped",
                        "version": version,
                        "path": str(finalist_dir),
                        "reason": "nonempty_without_manifest",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            version += 1
            continue
        if workflow_path.exists():
            print(
                json.dumps(
                    {
                        "event": "immutable_workflow_skipped",
                        "version": version,
                        "path": str(workflow_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            version += 1
            continue
        if path_is_occupied(export_root):
            print(
                json.dumps(
                    {
                        "event": "immutable_export_skipped",
                        "version": version,
                        "path": str(export_root),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            version += 1
            continue
        return (
            version,
            finalist_dir,
            export_root,
            workflow_path,
            expected_contract,
        )


def finalize_and_export(
    args: argparse.Namespace,
    *,
    root: Path,
    base_config: Path,
    shared_stage: Stage,
    executable: Path,
    export_models: bool = True,
) -> None:
    finalist_script = Path(__file__).with_name("finalist_recheck.py")
    devices = assigned_devices(args)
    if args.dry_run:
        finalist_dir = root / "finalist_recheck"
        export_root = (args.export_root or root / "exports").resolve()
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
                    "export_models": export_models,
                    "export_root": str(export_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    source_journal_sha256 = sha256(shared_stage.journal)
    (
        finalization_version,
        finalist_dir,
        export_root,
        workflow_path,
        finalization_contract,
    ) = select_finalization_paths(
        args,
        root=root,
        source_journal_sha256=source_journal_sha256,
        base_config=base_config,
    )
    args.finalization_version = finalization_version
    args.resolved_finalist_dir = str(finalist_dir)
    args.resolved_export_root = str(export_root)
    print(
        json.dumps(
            {
                "event": "finalization_version_selected",
                "version": finalization_version,
                "finalist_dir": str(finalist_dir),
                "export_root": str(export_root),
                "source_journal_sha256": source_journal_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
            "--keyword-near-gate-extra",
            str(args.keyword_near_gate_extra),
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

    prepared_manifest = load_json_object(finalist_manifest)
    if prepared_manifest is None or not finalization_manifest_matches(
        prepared_manifest,
        finalization_contract,
    ) or not prepared_finalization_artifacts_match(prepared_manifest):
        raise RuntimeError(
            f"Prepared finalist manifest violates the selected contract: "
            f"{finalist_manifest}"
        )
    contract_path = finalist_dir / "finalization_contract.json"
    recorded_contract = load_json_object(contract_path)
    expected_contract_record = finalization_contract_record(
        finalization_contract,
        finalist_manifest,
    )
    if recorded_contract is None:
        write_text_atomic(
            contract_path,
            json.dumps(expected_contract_record, indent=2, sort_keys=True) + "\n",
        )
    elif recorded_contract != expected_contract_record:
        raise RuntimeError(
            f"Finalization contract changed unexpectedly: {contract_path}"
        )
    args.finalization_contract = str(contract_path)
    args.finalization_contract_sha256 = sha256(contract_path)

    if workflow_path.is_file():
        print(
            json.dumps(
                {
                    "event": "workflow_resume",
                    "version": finalization_version,
                    "report": str(workflow_path),
                    "report_sha256": sha256(workflow_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    winners_path = finalist_dir / "winners.json"
    winners_report = load_valid_winners_report(winners_path)
    if winners_report is None:
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
        run_checked(
            recheck_command,
            cwd=Path(__file__).parents[2],
            event="finalist_recheck",
        )
        winners_report = load_valid_winners_report(winners_path)
        if winners_report is None:
            raise RuntimeError(f"Invalid finalist winners report: {winners_path}")
    else:
        print(
            json.dumps(
                {"event": "finalist_recheck_resume", "report": str(winners_path)},
                sort_keys=True,
            ),
            flush=True,
        )
    winners = winners_report.get("winners", {})

    if not export_models:
        print(
            json.dumps(
                {
                    "event": "recheck_only_complete",
                    "version": finalization_version,
                    "finalist_dir": str(finalist_dir),
                    "winners": str(winners_path),
                    "winners_sha256": sha256(winners_path),
                    "source_journal_sha256": source_journal_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    export_root.mkdir(parents=True, exist_ok=True)
    export_jobs: list[
        tuple[str, str, Path, dict[str, Any], subprocess.Popen[bytes]]
    ] = []
    collapsed_roles = winners_share_physical_export(winners)
    export_assignments = (
        (("Balanced", devices[0]),)
        if collapsed_roles
        else (("Balanced", devices[0]), ("Max", devices[-1]))
    )
    for variant, device in export_assignments:
        output = export_root / variant.lower()
        winner = winners[variant]
        if export_is_complete(
            output,
            variant=variant,
            winner=winner,
            export_strategy=args.export_strategy,
        ):
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

        # Distinct physical winners export concurrently on distinct GPUs. With
        # one GPU, finish each payload before loading the next full model.
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

    if collapsed_roles:
        alias_variant = "Max"
        alias_output = export_root / alias_variant.lower()
        alias_winner = winners[alias_variant]
        physical_output = export_root / "balanced"
        if export_alias_is_complete(
            alias_output,
            variant=alias_variant,
            winner=alias_winner,
            export_strategy=args.export_strategy,
            physical_directory=physical_output,
            physical_variant="Balanced",
            physical_winner=winners["Balanced"],
        ):
            print(
                json.dumps(
                    {
                        "event": "export_role_alias_resume",
                        "variant": alias_variant,
                        "path": str(alias_output),
                        "physical_variant": "Balanced",
                        "physical_path": str(physical_output),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            if alias_output.exists() and any(alias_output.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite incomplete {alias_variant} role alias: "
                    f"{alias_output}"
                )
            alias_manifest = write_export_alias_manifest(
                alias_output,
                variant=alias_variant,
                winner=alias_winner,
                export_strategy=args.export_strategy,
                physical_directory=physical_output,
                physical_variant="Balanced",
                physical_winner=winners["Balanced"],
            )
            print(
                json.dumps(
                    {
                        "event": "export_role_alias_complete",
                        "variant": alias_variant,
                        "path": str(alias_output),
                        "manifest": str(alias_manifest),
                        "manifest_sha256": sha256(alias_manifest),
                        "physical_variant": "Balanced",
                        "physical_path": str(physical_output),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    export_policy_reference = None
    if collapsed_roles:
        export_policy_path = write_collapsed_export_policy(
            export_root,
            winners,
            contract_path,
        )
        export_policy_reference = {
            "path": str(export_policy_path),
            "sha256": sha256(export_policy_path),
        }
        print(
            json.dumps(
                {
                    "event": "collapsed_export_policy_complete",
                    **export_policy_reference,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    workflow_report = {
        "schema_version": 1,
        "status": "PASS",
        "finalization_version": finalization_version,
        "search_journal": str(shared_stage.journal),
        "search_journal_sha256": sha256(shared_stage.journal),
        "finalist_report": str(winners_path),
        "finalist_report_sha256": sha256(winners_path),
        "finalization_contract": str(contract_path),
        "finalization_contract_sha256": sha256(contract_path),
        "exports": {
            variant: str(export_root / variant.lower())
            for variant in ("Balanced", "Max")
        },
        "export_roles": expected_export_roles(export_root, winners),
        "export_policy": export_policy_reference,
    }
    if workflow_path.exists():
        raise FileExistsError(f"Refusing to overwrite workflow report: {workflow_path}")
    write_text_atomic(
        workflow_path,
        json.dumps(workflow_report, indent=2, sort_keys=True) + "\n",
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
    if (
        args.max_keywords < 0
        or args.keyword_total <= 0
        or args.keyword_near_gate_extra < 0
    ):
        raise ValueError("Keyword gate values are invalid")
    if not 0 <= args.balanced_removal_fraction <= 1:
        raise ValueError("--balanced-removal-fraction must be in [0, 1]")

    source_base_config = args.base_config.resolve()
    executable_value = args.heretic or shutil.which("hereticMOE")
    if not executable_value:
        raise FileNotFoundError(
            "No HereticMOE executable found in PATH; provide --heretic explicitly"
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
    random_device = devices[0]
    sobol_device = devices[1] if len(devices) > 1 else devices[0]
    response_archive = root / "trial-responses.sqlite3"
    scorer_updates = (
        frozenset({"scorer"})
        if args.allow_scorer_config_update
        else frozenset()
    )
    random_stage: Stage | None = None
    sobol_stage: Stage | None = None
    if not args.dynamic_worker_queue:
        random_stage = build_stage(
            root,
            "random_branch",
            base,
            n_trials=branch_trials,
            n_startup_trials=branch_trials,
            startup_design="random",
            device=random_device,
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
            device=sobol_device,
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
        device=random_device,
        response_archive=response_archive,
        response_number_offset=0,
        response_number_stride=1,
        parallel_workers=shared_worker_count,
        dry_run=args.dry_run,
        allowed_config_updates=frozenset({"n_trials"}) | scorer_updates,
    )
    manifest_stages = [shared_stage]
    if random_stage is not None and sobol_stage is not None:
        manifest_stages = [random_stage, sobol_stage, shared_stage]
    manifest_path = root / "adaptive_run_manifest.json"
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=manifest_stages,
            status=(
                "tpe_preparing"
                if args.continue_shared_only
                else "exploration_running"
            ),
        )

    if not args.continue_shared_only and not args.dynamic_worker_queue:
        assert random_stage is not None
        assert sobol_stage is not None
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
                stages=manifest_stages,
                status="exploration_merged",
            )
    elif (
        args.continue_shared_only
        and not args.dry_run
        and not shared_stage.journal.is_file()
    ):
        raise FileNotFoundError(
            f"No shared journal to continue: {shared_stage.journal}"
        )

    completed_trials, waiting_trials = controller_trial_counts(
        shared_stage.journal,
        dry_run=args.dry_run,
        continue_shared_only=args.continue_shared_only,
        dynamic_worker_queue=args.dynamic_worker_queue,
        exploration_trials=args.exploration_trials,
    )

    selected_queue: TrialWorkQueue | None = None
    selected_queue_path = queue_path_for_version(root, args.target_trials, 1)
    queue_rejections: list[dict[str, Any]] = []
    if (
        args.dynamic_worker_queue
        and not args.dry_run
    ):
        selected_queue, candidate_path, queue_rejections = select_verified_queue(
            root,
            shared_stage.journal,
            target_trial_count=args.target_trials,
            tpe_concurrency=shared_worker_count,
        )
        assert candidate_path is not None
        selected_queue_path = candidate_path
    args.worker_queue_rejections = queue_rejections
    resuming_dynamic_queue = selected_queue is not None
    if completed_trials > args.target_trials and not resuming_dynamic_queue:
        raise ValueError(
            f"Shared study already has {completed_trials} trials, above target "
            f"{args.target_trials}"
        )
    # WAITING trials already occupy trial numbers but still need one optimization
    # call each. Add them back so queued remeasurements do not silently reduce the
    # requested number of actual evaluations.
    remaining_trials = max(0, args.target_trials - completed_trials) + (
        0 if args.dry_run else waiting_trials
    )
    queue: TrialWorkQueue | None = selected_queue
    queue_expected_tasks = remaining_trials
    if args.dynamic_worker_queue and (remaining_trials or resuming_dynamic_queue):
        first_task_id = completed_trials - (0 if args.dry_run else waiting_trials)
        queue_path = selected_queue_path
        args.worker_queue_path = str(queue_path.resolve())
        if not args.dry_run:
            if resuming_dynamic_queue:
                assert queue is not None
                contract = queue.contract()
                if (
                    contract.last_task_id_exclusive != args.target_trials
                    or contract.target_trial_count != args.target_trials
                    or contract.tpe_concurrency != shared_worker_count
                ):
                    raise RuntimeError(
                        "Verified queue contract changed unexpectedly: "
                        f"{contract}"
                    )
                queue_expected_tasks = contract.task_count
            else:
                running = (
                    [
                        trial.number
                        for trial in load_journal_trials(shared_stage.journal)
                        if trial.state == optuna.trial.TrialState.RUNNING
                    ]
                    if shared_stage.journal.is_file()
                    else []
                )
                if running:
                    raise RuntimeError(
                        "Refusing to allocate a replacement queue while the journal "
                        f"contains RUNNING trials: {running[:8]}"
                    )
                if shared_stage.journal.is_file():
                    journal_base_size_bytes = shared_stage.journal.stat().st_size
                    journal_base_sha256 = sha256(shared_stage.journal)
                else:
                    journal_base_size_bytes = 0
                    journal_base_sha256 = hashlib.sha256().hexdigest()
                queue = TrialWorkQueue(queue_path)
                queue.initialize(
                    first_task_id=first_task_id,
                    task_count=remaining_trials,
                    exploration_task_count=(
                        min(args.exploration_trials, remaining_trials)
                        if not args.continue_shared_only and first_task_id == 0
                        else 0
                    ),
                    target_trial_count=args.target_trials,
                    tpe_concurrency=shared_worker_count,
                    journal_base_trial_count=completed_trials,
                    journal_base_size_bytes=journal_base_size_bytes,
                    journal_base_sha256=journal_base_sha256,
                )
                queue_expected_tasks = remaining_trials
            prelaunch_recoveries: list[dict[str, Any]] = []
            for stale_worker_id in queue.claimed_workers():
                orphaned_trials = fail_running_trials_for_worker(
                    shared_stage.journal,
                    stale_worker_id,
                )
                released_tasks = queue.release_worker(stale_worker_id)
                recovery = {
                    "event": "prelaunch_worker_recovery",
                    "worker_id": stale_worker_id,
                    "orphaned_trials_failed": orphaned_trials,
                    "released_tasks": released_tasks,
                }
                prelaunch_recoveries.append(recovery)
                print(json.dumps(recovery, sort_keys=True), flush=True)
            args.worker_recoveries = prelaunch_recoveries
            queue_stats = queue.stats()
            if queue_stats.failed:
                raise RuntimeError(f"Dynamic worker queue has failed tasks: {queue_stats}")
            remaining_trials = queue_stats.pending + queue_stats.claimed
    else:
        args.worker_queue_path = None
    if should_require_constraint_metadata(
        dry_run=args.dry_run,
        journal_has_trials=completed_trials > 0,
        remaining_trials=remaining_trials,
    ):
        require_constraint_metadata(shared_stage)
    # Dynamic workers receive the same safety ceiling but consume exact work
    # permits from the queue. The legacy path still splits fixed budgets.
    worker_budgets = (
        [remaining_trials] * min(shared_worker_count, remaining_trials)
        if args.dynamic_worker_queue and remaining_trials
        else split_worker_budget(remaining_trials, shared_worker_count)
    )
    active_devices = devices[: len(worker_budgets)]
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=manifest_stages,
            status="tpe_running" if remaining_trials else "complete",
        )
    worker_seed_base = int(base.get("seed") or 0) + 10_000
    worker_processes: list[
        tuple[Stage, subprocess.Popen, str, tuple[str, ...], int]
    ] = []
    for worker_index, (device, budget) in enumerate(
        zip(active_devices, worker_budgets, strict=True)
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
        worker_id = f"gpu-{device}"
        worker_arguments = [
            "--n-trials",
            str(args.target_trials),
            "--n-startup-trials",
            "0",
            "--parallel-workers",
            str(len(active_devices)),
            f"--seed={worker_seed_base + worker_index}",
        ]
        if args.dynamic_worker_queue:
            worker_arguments.extend(
                (
                    "--worker-queue-path",
                    str(args.worker_queue_path),
                    f"--worker-id={worker_id}",
                )
            )
        else:
            worker_arguments.extend(("--worker-trial-budget", str(budget)))
        process = start_stage(
            worker_stage,
            executable,
            dry_run=args.dry_run,
            display_name=display_name,
            device=device,
            command_args=tuple(worker_arguments),
            visible_worker_window=args.visible_worker_windows,
        )
        worker_processes.append(
            (worker_stage, process, worker_id, tuple(worker_arguments), 0)
        )
    if args.dry_run:
        for _, process, _, _, _ in worker_processes:
            process.wait()
    else:
        if queue is not None:
            recoveries = [
                *getattr(args, "worker_recoveries", []),
                *wait_dynamic_workers(
                worker_processes,
                queue=queue,
                executable=executable,
                expected_tasks=queue_expected_tasks,
                visible_worker_window=args.visible_worker_windows,
                ),
            ]
        else:
            wait_parallel(
                [(stage, process) for stage, process, _, _, _ in worker_processes]
            )
            recoveries = []
        args.worker_recoveries = recoveries
        if queue is not None:
            queue_stats = queue.stats()
            if (
                queue_stats.pending
                or queue_stats.claimed
                or queue_stats.failed
                or queue_stats.complete != queue_expected_tasks
            ):
                raise RuntimeError(f"Dynamic worker queue incomplete: {queue_stats}")
            queue_valid, queue_reason = verify_queue_against_journal(
                queue,
                shared_stage.journal,
                target_trial_count=args.target_trials,
                tpe_concurrency=shared_worker_count,
            )
            if not queue_valid:
                raise RuntimeError(
                    "Completed dynamic queue failed journal verification: "
                    f"{queue_reason}"
                )
        final_trial_count = journal_trial_count(shared_stage.journal)
        if final_trial_count < args.target_trials:
            raise RuntimeError(
                f"Shared study ended at {final_trial_count}, below target "
                f"{args.target_trials}"
            )
    if not args.dry_run:
        write_run_manifest(
            manifest_path,
            args=args,
            base_config=base_config,
            stages=manifest_stages,
            status="complete",
        )
    if args.post_search_mode in {"export", "recheck"}:
        finalize_and_export(
            args,
            root=root,
            base_config=base_config,
            shared_stage=shared_stage,
            executable=executable,
            export_models=args.finalize,
        )
        if not args.dry_run:
            write_run_manifest(
                manifest_path,
                args=args,
                base_config=base_config,
                stages=manifest_stages,
                status=post_search_completion_status(args.post_search_mode),
            )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        mark_existing_run_failed(sys.argv[1:], error)
        raise
