from __future__ import annotations

import os
import sys

from heretic.language_map_controller import (
    GeometryWorkerSpec,
    build_worker_command,
    format_worker_event,
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


def test_worker_batch_and_phase_events_are_rendered_as_human_status() -> None:
    lines: list[str] = []
    payload = (
        "import json; "
        "print(json.dumps({'event':'worker_phase','phase':'model_load'})); "
        "print(json.dumps({'event':'batch_probe','mode':'generation','batch_size':64})); "
        "print(json.dumps({'event':'batch_backoff','mode':'generation','batch_size':64,"
        "'next_batch_size':32,'reason':'OOM'})); "
        "print(json.dumps({'event':'batch_selected','mode':'generation','batch_size':32}))"
    )
    spec = GeometryWorkerSpec(
        device="0",
        worker_id="gpu-0",
        command=(sys.executable, "-u", "-c", payload),
        environment=os.environ.copy(),
    )

    exits = run_worker_processes([spec], line_sink=lines.append)

    assert exits == {"gpu-0": 0}
    assert lines == [
        "[GPU 0] Loading model...",
        "[GPU 0] Batch probe (generation): 64",
        "[GPU 0] Batch 64 OOM; retrying 32",
        "[GPU 0] Batch selected (generation): 32",
    ]


def test_batch_validation_event_explains_long_full_generation() -> None:
    assert format_worker_event(
        "[GPU 1]",
        {
            "event": "batch_validation",
            "mode": "generation",
            "batch_size": 40,
            "max_new_tokens": 100,
        },
    ) == "[GPU 1] Validating batch 40 with 100 generated tokens..."


def test_batch_validation_result_shows_minimum_and_recovered_vram() -> None:
    assert format_worker_event(
        "[GPU 1]",
        {
            "event": "batch_validation_result",
            "mode": "generation",
            "batch_size": 144,
            "status": "PASS",
            "free_gib": 4.12,
            "recovered_gib": 14.12,
        },
    ) == (
        "[GPU 1] Batch validation PASS: 144; "
        "min 4.12 GiB, recovered 14.12 GiB"
    )


def test_batch_validation_result_shows_measured_throughput() -> None:
    assert format_worker_event(
        "[GPU 0]",
        {
            "event": "batch_validation_result",
            "mode": "generation",
            "batch_size": 224,
            "status": "PASS",
            "free_gib": 4.83,
            "recovered_gib": 14.12,
            "tokens_per_second": 1840.5,
            "elapsed_seconds": 12.17,
        },
    ) == (
        "[GPU 0] Batch validation PASS: 224; 1840.5 tok/s in 12.17s; "
        "min 4.83 GiB, recovered 14.12 GiB"
    )


def test_batch_reserve_floor_explains_successful_batch_one_fallback() -> None:
    assert format_worker_event(
        "[GPU 0]",
        {
            "event": "batch_reserve_floor",
            "mode": "conditional NLL",
            "batch_size": 1,
            "free_gib": 1.75,
            "required_gib": 2.40,
        },
    ) == (
        "[GPU 0] Conditional NLL batch 1 completed; "
        "reserve 1.75/2.40 GiB, continuing longest-first"
    )


def test_worker_prewarm_phases_explain_compile_pause() -> None:
    assert format_worker_event(
        "[GPU 0]", {"event": "worker_phase", "phase": "prewarm"}
    ) == "[GPU 0] Compiling resident generation shapes..."
    assert format_worker_event(
        "[GPU 0]", {"event": "worker_phase", "phase": "prewarm_ready"}
    ) == "[GPU 0] Generation backend ready; starting trials..."
