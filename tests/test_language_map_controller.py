from __future__ import annotations

import os
import sys

from heretic.language_map_controller import (
    GeometryWorkerSpec,
    build_worker_command,
    run_worker_processes,
    worker_environment,
)


def test_worker_environment_isolates_gpu_and_bounds_cpu_threads() -> None:
    environment = worker_environment(
        {"UNCHANGED": "yes"}, device="3", cpu_threads=5
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["OMP_NUM_THREADS"] == "5"
    assert environment["MKL_NUM_THREADS"] == "5"
    assert environment["RAYON_NUM_THREADS"] == "5"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["UNCHANGED"] == "yes"


def test_worker_command_carries_job_device_and_stable_worker_id(tmp_path) -> None:
    command = build_worker_command(tmp_path / "job.json", "7", "gpu-7")

    assert command[:3] == (
        sys.executable,
        "-u",
        "-c",
    )
    assert command[-6:] == (
        "--job",
        str(tmp_path / "job.json"),
        "--device",
        "7",
        "--worker-id",
        "gpu-7",
    )


def test_two_real_worker_processes_stream_into_one_prefixed_output() -> None:
    lines: list[str] = []
    base = os.environ.copy()
    specs = [
        GeometryWorkerSpec(
            device=device,
            worker_id=f"gpu-{device}",
            command=(sys.executable, "-u", "-c", f"print('worker-{device}-ready')"),
            environment=base,
        )
        for device in ("0", "1")
    ]

    exits = run_worker_processes(specs, line_sink=lines.append)

    assert exits == {"gpu-0": 0, "gpu-1": 0}
    assert sorted(lines) == [
        "[GPU 0] worker-0-ready",
        "[GPU 1] worker-1-ready",
    ]

