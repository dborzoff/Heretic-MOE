# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resident-worker capture and canonical merge for multilingual geometry."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .language_map_data import GeometryRow, text_free_row_index
from .range_work_queue import RangeWorkQueue
from .utils import Prompt

WorkerProgress = Callable[[str, int, int, int, int], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def capture_claimed_ranges(
    model: Any,
    rows: list[GeometryRow],
    queue: RangeWorkQueue,
    parts_dir: str | Path,
    *,
    worker_id: str,
    batch_size: int,
    system_prompt: str,
    progress: WorkerProgress | None = None,
) -> dict[str, int | str]:
    """Keep one model resident while dynamically claiming global row ranges."""

    if batch_size < 0:
        raise ValueError("batch_size must be nonnegative")
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed_tasks = 0
    completed_rows = 0
    while (item := queue.claim(worker_id)) is not None:
        temporary: Path | None = None
        try:
            prompts = [
                Prompt(system=system_prompt, user=row.prompt)
                for row in rows[item.start : item.end]
            ]
            batches: list[Tensor] = []
            expected_shape: tuple[int, int] | None = None
            for batch in model.iter_residual_batches(prompts, batch_size):
                if not isinstance(batch, Tensor) or batch.ndim != 3:
                    raise ValueError(
                        "residual batch must have shape [rows, layers, hidden]"
                    )
                shape = (int(batch.shape[1]), int(batch.shape[2]))
                if expected_shape is None:
                    expected_shape = shape
                elif expected_shape != shape:
                    raise ValueError("residual layer/hidden shape drift")
                cpu_batch = (
                    batch.detach().to(device="cpu", dtype=torch.float32).contiguous()
                )
                if not bool(torch.isfinite(cpu_batch).all()):
                    raise ValueError("residual range contains non-finite values")
                batches.append(cpu_batch)
            if not batches:
                raise ValueError("model returned no residual batches")
            residuals = torch.cat(batches, dim=0).contiguous()
            expected_rows = item.end - item.start
            if residuals.shape[0] != expected_rows:
                raise ValueError(
                    f"model returned {residuals.shape[0]} rows, expected {expected_rows}"
                )
            part_name = f"part_{item.start:08d}_{item.end:08d}.safetensors"
            part_path = parts_dir / part_name
            temporary = parts_dir / f"{part_name}.{worker_id}.tmp"
            temporary.unlink(missing_ok=True)
            save_file({"residuals": residuals}, str(temporary))
            temporary.replace(part_path)
            queue.complete(
                item,
                part_file=part_name,
                sha256=_sha256(part_path),
                shape=tuple(int(value) for value in residuals.shape),
            )
            completed_tasks += 1
            completed_rows += expected_rows
            if progress is not None:
                progress(
                    worker_id,
                    completed_rows,
                    len(rows),
                    queue.stats().complete_rows,
                    len(rows),
                )
        except BaseException as error:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            queue.fail(item, type(error).__name__)
            raise
    return {
        "worker_id": worker_id,
        "tasks": completed_tasks,
        "rows": completed_rows,
    }


def finalize_range_cache(
    rows: list[GeometryRow],
    queue: RangeWorkQueue,
    parts_dir: str | Path,
    output_dir: str | Path,
    *,
    metadata: dict[str, object],
    capture_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Verify every range and publish one canonical cache with manifest last."""

    parts_dir = Path(parts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = {
        name: output_dir / name
        for name in ("residuals.safetensors", "row_index.jsonl", "manifest.json")
    }
    existing = [path for path in final.values() if path.exists()]
    if existing:
        raise FileExistsError(f"cache output already exists: {existing[0].name}")
    temporary = {name: path.with_name(f"{path.name}.tmp") for name, path in final.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    records = queue.records()
    stats = queue.stats()
    if stats.complete_rows != len(rows) or any(
        record.state != "complete" for record in records
    ):
        raise ValueError("range queue is not complete")

    expected_start = 0
    expected_shape: tuple[int, int] | None = None
    tensors: list[Tensor] = []
    worker_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tasks": 0, "rows": 0}
    )
    try:
        for record in records:
            if record.start != expected_start:
                raise ValueError("range coverage is not contiguous")
            if record.part_file is None or record.sha256 is None or record.shape is None:
                raise ValueError("complete range is missing part metadata")
            path = parts_dir / record.part_file
            if not path.is_file() or _sha256(path) != record.sha256:
                raise ValueError(f"part hash mismatch: {record.task_id}")
            tensor = load_file(str(path))["residuals"]
            if tuple(int(value) for value in tensor.shape) != record.shape:
                raise ValueError(f"part shape mismatch: {record.task_id}")
            if tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"invalid residual part: {record.task_id}")
            shape = (int(tensor.shape[1]), int(tensor.shape[2]))
            if expected_shape is None:
                expected_shape = shape
            elif expected_shape != shape:
                raise ValueError("residual layer/hidden shape drift")
            tensors.append(tensor)
            expected_start = record.end
            worker = record.worker_id or "unknown"
            worker_stats[worker]["tasks"] += 1
            worker_stats[worker]["rows"] += record.end - record.start
        if expected_start != len(rows):
            raise ValueError("range coverage does not match row count")

        residuals = torch.cat(tensors, dim=0).contiguous()
        save_file({"residuals": residuals}, str(temporary["residuals.safetensors"]))
        _write_jsonl(temporary["row_index.jsonl"], text_free_row_index(rows))
        files = {
            name: {
                "bytes": temporary[name].stat().st_size,
                "sha256": _sha256(temporary[name]),
            }
            for name in ("residuals.safetensors", "row_index.jsonl")
        }
        manifest: dict[str, Any] = {
            "status": "PASS",
            "schema_version": 2,
            "measurement_position": "first_generated_token",
            "rows": len(rows),
            "layers": int(residuals.shape[1]),
            "hidden_size": int(residuals.shape[2]),
            "tensor_dtype": str(residuals.dtype).removeprefix("torch."),
            "files": files,
            "capture": {
                "mode": "resident_range_workers",
                "parts": len(records),
                "workers": dict(sorted(worker_stats.items())),
            },
            "metadata": dict(metadata),
        }
        if capture_fingerprint is not None:
            manifest["capture_fingerprint"] = capture_fingerprint
        _write_json(temporary["manifest.json"], manifest)
        temporary["residuals.safetensors"].replace(final["residuals.safetensors"])
        temporary["row_index.jsonl"].replace(final["row_index.jsonl"])
        temporary["manifest.json"].replace(final["manifest.json"])
        for record in records:
            if record.part_file is not None:
                (parts_dir / record.part_file).unlink(missing_ok=True)
        try:
            parts_dir.rmdir()
        except OSError:
            pass
        return manifest
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
