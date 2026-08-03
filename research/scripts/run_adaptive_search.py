#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the bounded Random/Sobol -> shared TPE Heretic workflow.

The controller owns only configuration, processes, journals, and text-free
provenance. Prompt and response payloads remain inside the scorer processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w


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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--heretic", type=Path, required=True)
    parser.add_argument("--branch-trials", type=int, default=120)
    parser.add_argument("--startup-trials", type=int, default=60)
    parser.add_argument("--target-trials", type=int, default=600)
    parser.add_argument("--random-device", default="0")
    parser.add_argument("--sobol-device", default="1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--continue-shared-only",
        action="store_true",
        help="Skip branch creation/merge and extend an existing shared journal.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_model_name(model: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "--" for c in model)


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"Base config has no non-empty model: {path}")
    return config


def stage_config(
    base: dict[str, Any],
    *,
    checkpoint_dir: Path,
    n_trials: int,
    n_startup_trials: int,
    startup_design: str,
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
    if device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = device
    return environment


def start_stage(stage: Stage, executable: Path, *, dry_run: bool) -> subprocess.Popen:
    command = [str(executable)]
    print(
        json.dumps(
            {
                "event": "stage_start",
                "stage": stage.name,
                "cwd": str(stage.directory),
                "device": stage.device,
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
    return subprocess.Popen(
        command,
        cwd=stage.directory,
        env=process_environment(stage.device),
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


def run_stage(stage: Stage, executable: Path, *, dry_run: bool) -> None:
    process = start_stage(stage, executable, dry_run=dry_run)
    if dry_run:
        process.wait()
        return
    wait_stage(stage, process)


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
) -> None:
    record = {
        "schema_version": 1,
        "created_unix": time.time(),
        "base_config": str(base_config),
        "base_config_sha256": sha256(base_config),
        "branch_trials": args.branch_trials,
        "startup_trials": args.startup_trials,
        "target_trials": args.target_trials,
        "parallel": args.parallel,
        "continue_shared_only": args.continue_shared_only,
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
                "device": stage.device,
            }
            for stage in stages
        ],
    }
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == payload:
        return
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.branch_trials <= 0 or args.startup_trials < 0 or args.target_trials <= 0:
        raise ValueError("Trial budgets must be positive (startup may be zero)")
    if args.startup_trials > args.branch_trials:
        raise ValueError("--startup-trials cannot exceed --branch-trials")
    if not args.continue_shared_only and args.target_trials <= 2 * args.branch_trials:
        raise ValueError(
            "--target-trials must exceed the combined two-branch prefix"
        )

    base_config = args.base_config.resolve()
    executable = args.heretic.resolve()
    root = args.run_root.resolve()
    if not base_config.is_file():
        raise FileNotFoundError(base_config)
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    base = read_config(base_config)
    random_stage = build_stage(
        root,
        "random_branch",
        base,
        n_trials=args.branch_trials,
        n_startup_trials=args.startup_trials,
        startup_design="random",
        device=args.random_device,
        dry_run=args.dry_run,
    )
    sobol_stage = build_stage(
        root,
        "sobol_branch",
        base,
        n_trials=args.branch_trials,
        n_startup_trials=args.startup_trials,
        startup_design="sobol",
        device=args.sobol_device,
        dry_run=args.dry_run,
    )
    shared_stage = build_stage(
        root,
        "shared_tpe",
        base,
        n_trials=args.target_trials,
        n_startup_trials=0,
        startup_design="random",
        device=args.random_device,
        dry_run=args.dry_run,
        allowed_config_updates=frozenset({"n_trials"}),
    )

    if not args.continue_shared_only:
        if args.parallel:
            random_process = start_stage(random_stage, executable, dry_run=args.dry_run)
            sobol_process = start_stage(sobol_stage, executable, dry_run=args.dry_run)
            if args.dry_run:
                random_process.wait()
                sobol_process.wait()
            else:
                wait_stage(random_stage, random_process)
                wait_stage(sobol_stage, sobol_process)
        else:
            run_stage(random_stage, executable, dry_run=args.dry_run)
            run_stage(sobol_stage, executable, dry_run=args.dry_run)
        merge_branches(
            random_stage,
            sobol_stage,
            shared_stage,
            target_trials=args.target_trials,
            dry_run=args.dry_run,
        )
    elif not args.dry_run and not shared_stage.journal.is_file():
        raise FileNotFoundError(
            f"No shared journal to continue: {shared_stage.journal}"
        )

    run_stage(shared_stage, executable, dry_run=args.dry_run)
    if not args.dry_run:
        write_run_manifest(
            root / "adaptive_run_manifest.json",
            args=args,
            base_config=base_config,
            stages=[random_stage, sobol_stage, shared_stage],
        )


if __name__ == "__main__":
    main()
