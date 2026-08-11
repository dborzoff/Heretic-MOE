# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-console process controller for resident geometry-map GPU workers."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Thread


@dataclass(frozen=True)
class GeometryWorkerSpec:
    device: str
    worker_id: str
    command: tuple[str, ...]
    environment: Mapping[str, str]


def worker_environment(
    base: Mapping[str, str] | None,
    *,
    device: str,
    cpu_threads: int,
) -> dict[str, str]:
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device),
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "RAYON_NUM_THREADS": str(cpu_threads),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def build_worker_command(
    job_path: str | Path,
    device: str,
    worker_id: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        "-c",
        "from heretic.language_map_worker import main; main()",
        "--job",
        str(job_path),
        "--device",
        str(device),
        "--worker-id",
        worker_id,
    )


def run_worker_processes(
    specifications: Sequence[GeometryWorkerSpec],
    *,
    line_sink: Callable[[str], None] = print,
) -> dict[str, int]:
    if not specifications:
        raise ValueError("at least one worker specification is required")
    processes: list[tuple[GeometryWorkerSpec, subprocess.Popen[str]]] = []
    readers: list[Thread] = []

    def stream(specification: GeometryWorkerSpec, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        prefix = f"[GPU {specification.device}]"
        for raw_line in process.stdout:
            line_sink(f"{prefix} {raw_line.rstrip()}")

    try:
        for specification in specifications:
            process = subprocess.Popen(
                specification.command,
                env=dict(specification.environment),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append((specification, process))
            reader = Thread(
                target=stream,
                args=(specification, process),
                name=f"geometry-output-{specification.worker_id}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)
        exits = {
            specification.worker_id: int(process.wait())
            for specification, process in processes
        }
        for reader in readers:
            reader.join()
        return exits
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
