# SPDX-License-Identifier: AGPL-3.0-or-later

"""One-console process controller for resident geometry-map GPU workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread

from .pipeline_ui import PipelineUI


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


def format_worker_event(prefix: str, event: Mapping[str, object]) -> str | None:
    kind = str(event.get("event", ""))
    if kind == "worker_phase":
        phase = str(event.get("phase", ""))
        labels = {
            "model_load": "Loading model...",
            "model_ready": "Model loaded; preparing token cache...",
            "tokenize": "Tokenizing frozen rows...",
            "prewarm": "Compiling resident generation shapes...",
            "prewarm_ready": "Generation backend ready; starting trials...",
        }
        label = labels.get(phase)
        return f"{prefix} {label}" if label is not None else None
    if kind == "token_cache_ready":
        rows = int(event.get("rows", 0))
        tokens = int(event.get("tokens", 0))
        pinned = "pinned" if bool(event.get("pinned", False)) else "CPU"
        return f"{prefix} Token cache ready: {rows} rows, {tokens} tokens ({pinned})"
    mode = str(event.get("mode", "batch"))
    batch_size = int(event.get("batch_size", 0))
    if kind == "batch_probe":
        return f"{prefix} Batch probe ({mode}): {batch_size}"
    if kind == "batch_backoff":
        next_batch_size = int(event.get("next_batch_size", 0))
        reason = str(event.get("reason", "retry"))
        return f"{prefix} Batch {batch_size} {reason}; retrying {next_batch_size}"
    if kind == "batch_probe_result":
        free_gib = float(event.get("free_gib", 0.0))
        return f"{prefix} Batch {batch_size} fits; {free_gib:.2f} GiB free"
    if kind == "batch_validation":
        tokens = int(event.get("max_new_tokens", 0))
        return f"{prefix} Validating batch {batch_size} with {tokens} generated tokens..."
    if kind == "batch_validation_result":
        free_gib = float(event.get("free_gib", 0.0))
        recovered_gib = event.get("recovered_gib")
        status = str(event.get("status", "unknown"))
        throughput = float(event.get("tokens_per_second", 0.0))
        elapsed = float(event.get("elapsed_seconds", 0.0))
        speed = (
            f"{throughput:.1f} tok/s in {elapsed:.2f}s; "
            if throughput > 0.0 and elapsed > 0.0
            else ""
        )
        if recovered_gib is not None:
            return (
                f"{prefix} Batch validation {status}: {batch_size}; "
                f"{speed}min {free_gib:.2f} GiB, "
                f"recovered {float(recovered_gib):.2f} GiB"
            )
        return (
            f"{prefix} Batch validation {status}: {batch_size}; "
            f"{speed}min {free_gib:.2f} GiB"
        )
    if kind == "batch_selected":
        return f"{prefix} Batch selected ({mode}): {batch_size}"
    return None


def run_worker_processes(
    specifications: Sequence[GeometryWorkerSpec],
    *,
    line_sink: Callable[[str], None] = print,
    stage_name: str = "GPU work",
    total_rows: int | None = None,
    next_action: str | None = None,
) -> dict[str, int]:
    if not specifications:
        raise ValueError("at least one worker specification is required")
    processes: list[tuple[GeometryWorkerSpec, subprocess.Popen[str]]] = []
    readers: list[Thread] = []
    progress_by_worker: dict[str, tuple[int, int]] = {}
    selected_batch_by_worker: dict[str, int] = {}
    progress_started = time.monotonic()
    progress_lock = Lock()
    ui = PipelineUI() if line_sink is print and total_rows is not None else None
    if ui is not None:
        ui.stage(
            stage_name,
            total=total_rows,
            workers=tuple(specification.worker_id for specification in specifications),
            description=f"Resident workers: {len(specifications)} GPU(s)",
        )

    def stream(specification: GeometryWorkerSpec, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        prefix = f"[GPU {specification.device}]"
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                event = None
            if isinstance(event, dict):
                human_event = format_worker_event(prefix, event)
                if human_event is not None:
                    if event.get("event") == "batch_selected":
                        with progress_lock:
                            selected_batch_by_worker[specification.worker_id] = int(
                                event["batch_size"]
                            )
                    if ui is not None:
                        ui.note(human_event)
                    else:
                        line_sink(human_event)
                    continue
            if isinstance(event, dict) and event.get("event") in {
                "worker_progress",
                "reference_worker_progress",
                "final_holdout_worker_progress",
            }:
                completed = int(event["completed"])
                total = int(event["total"])
                with progress_lock:
                    progress_by_worker[specification.worker_id] = (completed, total)
                    if ui is not None:
                        ui.update_worker(
                            specification.worker_id,
                            completed=completed,
                            total=total,
                        )
                        if "global_completed" in event:
                            ui.update_overall(
                                completed=int(event["global_completed"]),
                                total=int(event["global_total"]),
                            )
                        continue
                    if event.get("scope") == "global":
                        global_completed = max(
                            value[0] for value in progress_by_worker.values()
                        )
                        global_total = max(
                            value[1] for value in progress_by_worker.values()
                        )
                    else:
                        global_completed = sum(
                            value[0] for value in progress_by_worker.values()
                        )
                        global_total = sum(
                            value[1] for value in progress_by_worker.values()
                        )
                    elapsed = max(time.monotonic() - progress_started, 1e-6)
                    rate = global_completed / elapsed
                    remaining = max(global_total - global_completed, 0)
                    eta = remaining / rate if rate > 0 else 0.0
                    progress_line = (
                        f"{prefix} {global_completed}/{global_total} | "
                        f"{rate:.1f} rows/s | ETA {eta / 60:.1f}m"
                    )
                    if line_sink is print:
                        sys.stdout.write(f"\r{progress_line:<100}")
                        sys.stdout.flush()
                    else:
                        line_sink(progress_line)
                continue
            if progress_by_worker and line_sink is print:
                sys.stdout.write("\n")
                sys.stdout.flush()
            line_sink(f"{prefix} {line}")

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
        if progress_by_worker and line_sink is print:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if ui is not None:
            summary: dict[str, object] = {
                "status": "PASS" if not any(exits.values()) else "FAIL",
                "workers": len(specifications),
                "rows": total_rows,
            }
            if next_action is not None and not any(exits.values()):
                summary["next"] = next_action
            if selected_batch_by_worker:
                selected = set(selected_batch_by_worker.values())
                summary["batch"] = (
                    str(next(iter(selected)))
                    if len(selected) == 1
                    else ", ".join(
                        f"{worker.removeprefix('gpu-')}:{batch}"
                        for worker, batch in sorted(selected_batch_by_worker.items())
                    )
                )
            ui.finish_stage(summary)
            ui.close()
        return exits
    except BaseException:
        if ui is not None:
            ui.finish_stage({"status": "FAIL", "error": "worker stage failed"})
            ui.close()
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
