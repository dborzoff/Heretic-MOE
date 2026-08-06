# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-command adaptive multi-GPU supervisor for HereticMOE."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import BinaryIO


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
        self.path = run_root.resolve() / ".hereticmoe-controller.lock"
        self.handle: BinaryIO | None = None

    def __enter__(self) -> "AdaptiveRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
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
        devices.append(
            GpuInfo(
                index=fields[0],
                name=fields[1],
                total_mib=int(fields[2]),
                free_mib=int(fields[3]),
                utilization=int(fields[4]),
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
        discovered = shutil.which("hereticMOE")
        if not discovered:
            candidate = Path(sys.executable).resolve().parent / "hereticMOE.exe"
            discovered = str(candidate) if candidate.is_file() else None
        if not discovered:
            raise FileNotFoundError(
                "Cannot locate hereticMOE executable; provide --worker-executable"
            )
        result = Path(discovered).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hereticMOE adaptive",
        description="Adaptive render-queue style search across available GPUs.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--min-free-fraction", type=float, default=0.70)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--exploration-trials", type=int, default=120)
    parser.add_argument("--n-trials", type=int, default=600)
    parser.add_argument("--continue-shared-only", action="store_true")
    parser.add_argument(
        "--finalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--worker-executable", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0 <= args.min_free_fraction <= 1:
        raise ValueError("--min-free-fraction must be in [0, 1]")
    if args.min_free_gib < 0:
        raise ValueError("--min-free-gib cannot be negative")
    if args.max_workers is not None and args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    root = repository_root()
    base_config = args.base_config or (
        root
        / "research"
        / "configs"
        / "adaptive_search"
        / "gemma2_sparse_geometry.toml"
    )
    worker_executable = executable_path(args.worker_executable)
    available = detect_nvidia_gpus()
    selected = select_devices(
        available,
        args.devices,
        min_free_fraction=args.min_free_fraction,
        min_free_gib=args.min_free_gib,
        max_workers=args.max_workers,
    )

    print(
        f"HereticMOE v{version('heretic-llm')} adaptive supervisor",
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

    controller = root / "research" / "scripts" / "run_adaptive_search.py"
    device_indices = ",".join(device.index for device in selected)
    command = [
        sys.executable,
        str(controller),
        "--base-config",
        str(base_config.resolve()),
        "--model",
        args.model,
        "--run-root",
        str(args.run_root.resolve()),
        "--heretic",
        str(worker_executable),
        "--exploration-trials",
        str(args.exploration_trials),
        "--target-trials",
        str(args.n_trials),
        "--devices",
        device_indices,
        "--random-device",
        selected[0].index,
        "--sobol-device",
        selected[1].index if len(selected) > 1 else selected[0].index,
        "--dynamic-worker-queue",
    ]
    if args.data_root:
        command.extend(("--data-root", str(args.data_root.resolve())))
    if args.continue_shared_only:
        command.append("--continue-shared-only")
    command.append("--finalize" if args.finalize else "--no-finalize")
    if args.dry_run:
        command.append("--dry-run")

    environment = os.environ.copy()
    environment["HERETIC_SUPERVISED"] = "1"
    with AdaptiveRunLock(args.run_root):
        result = subprocess.run(command, cwd=root, env=environment, check=False)
    raise SystemExit(result.returncode)
