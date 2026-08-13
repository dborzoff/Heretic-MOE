# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-command dynamic multi-GPU supervisor for HereticMOE."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import BinaryIO

from .launch_config import (
    LaunchOverrides,
    build_internal_settings,
    load_effective_launch_config,
    resolve_run_root,
    write_effective_config_bundle,
)


@dataclass(frozen=True)
class GpuInfo:
    index: str
    name: str
    total_mib: int
    free_mib: int
    utilization: int

    @property
    def free_fraction(self) -> float:
        return self.free_mib / self.total_mib if self.total_mib else 0.0


class AdaptiveRunLock(AbstractContextManager["AdaptiveRunLock"]):
    """Keep two supervisors from launching workers into the same run root."""

    def __init__(self, run_root: Path):
        resolved = run_root.resolve()
        self.path = resolved.parent / f".{resolved.name}.hereticmoe-controller.lock"
        self.handle: BinaryIO | None = None

    def __enter__(self) -> AdaptiveRunLock:  # noqa: PYI034 - Python 3.10 support
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.path.open("x+b")
        except FileExistsError:
            handle = self.path.open("r+b")
        self.handle = handle
        handle.seek(0, os.SEEK_END)
        if handle.tell() < 64:
            handle.write(b"\0" * (64 - handle.tell()))
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as error:
            handle.close()
            self.handle = None
            raise RuntimeError(
                f"Another HereticMOE supervisor is already using {self.path.parent}"
            ) from error
        handle.seek(0)
        lock_record = f"pid={os.getpid()}\n".encode().ljust(64, b" ")
        handle.write(lock_record)
        handle.truncate(64)
        handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        handle = self.handle
        assert handle is not None
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(),
                fcntl.LOCK_UN,  # type: ignore[attr-defined]
            )
        handle.close()
        self.handle = None


def detect_nvidia_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Unable to detect NVIDIA GPUs with nvidia-smi") from error

    devices: list[GpuInfo] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi row: {raw_line!r}")
        try:
            total_mib = int(fields[2])
            free_mib = int(fields[3])
            utilization = int(fields[4])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi returned an unavailable numeric field for GPU "
                f"{fields[0]}: {raw_line!r}"
            ) from error
        devices.append(
            GpuInfo(
                index=fields[0],
                name=fields[1],
                total_mib=total_mib,
                free_mib=free_mib,
                utilization=utilization,
            )
        )
    if not devices:
        raise RuntimeError("No NVIDIA GPUs detected")
    return devices


def select_devices(
    available: list[GpuInfo],
    specification: str,
    *,
    min_free_fraction: float,
    min_free_gib: float,
    max_workers: int | None,
) -> list[GpuInfo]:
    by_index = {device.index: device for device in available}
    if specification.lower() == "auto":
        required_mib = int(min_free_gib * 1024)
        selected = [
            device
            for device in available
            if device.free_mib >= required_mib
            and device.free_fraction >= min_free_fraction
        ]
    else:
        indices = [part.strip() for part in specification.split(",")]
        if any(not index for index in indices):
            raise ValueError("--devices contains an empty GPU index")
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise ValueError(f"Unknown GPU indices: {', '.join(missing)}")
        selected = [by_index[index] for index in dict.fromkeys(indices)]

    if max_workers is not None:
        selected = selected[:max_workers]
    if not selected:
        raise RuntimeError(
            "No GPU passes the free-memory gate; select devices explicitly or "
            "lower --min-free-fraction/--min-free-gib"
        )
    return selected


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    controller = root / "research" / "scripts" / "run_adaptive_search.py"
    if not controller.is_file():
        raise RuntimeError(
            "Adaptive supervisor requires a source checkout containing "
            "research/scripts/run_adaptive_search.py"
        )
    return root


def executable_path(override: Path | None) -> Path:
    if override is not None:
        result = override.resolve()
    else:
        # Child processes receive HERETIC_MOE_INTERNAL=1 from the controller,
        # so the same public launcher safely dispatches them to worker_main.
        # Keeping one executable also makes a clean install independent of the
        # legacy upstream ``heretic`` console entry point.
        launcher_name = "hereticMOE.exe" if os.name == "nt" else "hereticMOE"
        candidates = (
            Path(sys.argv[0]),
            Path(sys.prefix)
            / ("Scripts" if os.name == "nt" else "bin")
            / launcher_name,
            Path(sys.executable).resolve().parent / launcher_name,
        )
        discovered = next(
            (
                str(candidate.resolve())
                for candidate in candidates
                if candidate.is_file()
                and candidate.name.lower() == launcher_name.lower()
            ),
            None,
        )
        if not discovered:
            discovered = shutil.which("hereticMOE")
        if not discovered:
            raise FileNotFoundError("Cannot locate hereticMOE worker launcher")
        result = Path(discovered).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hereticMOE",
        description="Dynamic render-queue search across available GPUs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Public Heretic-MOE YAML configuration (default: ./config.yaml).",
    )
    parser.add_argument("--model", help="Override model.path from YAML.")
    parser.add_argument("--run-root", type=Path, help="Override run.root from YAML.")
    parser.add_argument("--devices", help="Override devices.include from YAML.")
    parser.add_argument("--target-trials", type=int)
    parser.add_argument("--exploration-trials", type=int)
    parser.add_argument(
        "--post-search",
        choices=("export", "recheck", "none"),
    )
    parser.add_argument(
        "--incompatible-contract",
        choices=("archive", "new_run", "replace", "fail"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_effective_launch_config(
        args.config,
        LaunchOverrides(
            model=args.model,
            run_root=args.run_root,
            devices=args.devices,
            target_trials=args.target_trials,
            exploration_trials=args.exploration_trials,
            post_search=args.post_search,
            incompatible_contract=args.incompatible_contract,
        ),
    )

    root = repository_root()
    worker_executable = executable_path(None)
    available = detect_nvidia_gpus()
    selected = select_devices(
        available,
        config.devices.selection_specification(),
        min_free_fraction=config.devices.min_free_fraction,
        min_free_gib=config.devices.min_free_gib,
        max_workers=config.devices.max_workers,
    )

    print(
        f"HereticMOE v{version('heretic-llm')} multi-GPU supervisor",
        flush=True,
    )
    print(f"Selected {len(selected)} GPU worker(s):", flush=True)
    for device in selected:
        print(
            f"  GPU {device.index}: {device.name} | "
            f"free {device.free_mib / 1024:.2f}/{device.total_mib / 1024:.2f} GiB | "
            f"utilization {device.utilization}%",
            flush=True,
        )

    if args.dry_run:
        resolution = resolve_run_root(
            config.run.root,
            config,
            config.run.incompatible_contract,
            mutate=False,
        )
        print(
            f"Dry run: would use {resolution.run_root} "
            f"with {config.run.target_trials} trial(s); no files were changed.",
            flush=True,
        )
        return

    environment = os.environ.copy()
    environment["HERETIC_SUPERVISED"] = "1"
    with AdaptiveRunLock(config.run.root):
        preview = resolve_run_root(
            config.run.root,
            config,
            config.run.incompatible_contract,
            mutate=False,
        )
        run_config = config.model_copy(
            update={"run": config.run.model_copy(update={"root": preview.run_root})}
        )
        internal_settings = build_internal_settings(run_config)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{preview.run_root.name}.config-staging-",
                dir=preview.run_root.parent,
            )
        )
        try:
            staged_bundle = write_effective_config_bundle(
                args.config,
                run_config,
                staging_root,
            )
            resolution = resolve_run_root(
                config.run.root,
                config,
                config.run.incompatible_contract,
            )
            run_root = resolution.run_root
            run_root.mkdir(parents=True, exist_ok=True)
            for staged_file in staging_root.iterdir():
                staged_file.replace(run_root / staged_file.name)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        if resolution.archived_root is not None:
            print(
                f"Archived incompatible run: {resolution.archived_root}",
                flush=True,
            )
        elif resolution.replaced:
            print(f"Replaced incompatible Heretic-MOE run: {run_root}", flush=True)
        elif resolution.run_root != config.run.root.resolve():
            print(f"Using new run root: {run_root}", flush=True)
        effective_toml = run_root / staged_bundle.effective_toml.name
        controller = root / "research" / "scripts" / "run_adaptive_search.py"
        device_indices = ",".join(device.index for device in selected)
        command = [
            sys.executable,
            str(controller),
            "--base-config",
            str(effective_toml),
            "--model",
            internal_settings.model,
            "--run-root",
            str(run_root),
            "--heretic",
            str(worker_executable),
            "--exploration-trials",
            str(config.run.exploration_trials),
            "--target-trials",
            str(config.run.target_trials),
            "--devices",
            device_indices,
            "--random-device",
            selected[0].index,
            "--sobol-device",
            selected[1].index if len(selected) > 1 else selected[0].index,
            "--dynamic-worker-queue",
            "--finalist-top-n",
            str(config.finalists.top_n),
            "--balanced-removal-fraction",
            str(config.finalists.balanced_removal_fraction),
            "--max-worker-restarts",
            str(config.recovery.max_restarts_per_gpu),
            "--heartbeat-interval-seconds",
            str(config.recovery.worker_heartbeat_seconds),
            "--lease-timeout-seconds",
            str(config.recovery.worker_timeout_seconds),
        ]
        multilingual = internal_settings.multilingual_search
        if multilingual.dataset_root:
            command.extend(
                ("--data-root", str(Path(multilingual.dataset_root).resolve()))
            )
        command.append(
            {
                "export": "--finalize",
                "recheck": "--recheck-only",
                "none": "--no-finalize",
            }[config.run.post_search]
        )
        result = subprocess.run(command, cwd=root, env=environment, check=False)
    raise SystemExit(result.returncode)
