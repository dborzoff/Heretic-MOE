# SPDX-License-Identifier: AGPL-3.0-or-later

"""Immutable residual-cache capture for multilingual geometry diagnostics."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .language_map_data import GeometryRow, text_free_row_index
from .utils import Prompt


ProgressCallback = Callable[[int, int], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _capture_fingerprint(
    rows: list[GeometryRow],
    batch_size: int,
    system_prompt: str,
    metadata: dict[str, object] | None,
) -> str:
    payload = {
        "batch_size": batch_size,
        "index": text_free_row_index(rows),
        "metadata": dict(metadata or {}),
        "prompt_sha256": [
            hashlib.sha256(row.prompt.encode("utf-8")).hexdigest() for row in rows
        ],
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_capture_state(
    state_path: Path,
    parts_dir: Path,
    fingerprint: str,
) -> tuple[dict[str, Any], list[Tensor]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("fingerprint") != fingerprint:
        raise ValueError("capture checkpoint does not match current inputs")
    tensors: list[Tensor] = []
    expected_start = 0
    expected_shape: tuple[int, int] | None = None
    for part in state.get("parts", []):
        if part.get("start") != expected_start or part.get("end", 0) <= expected_start:
            raise ValueError("capture checkpoint has invalid row ranges")
        path = parts_dir / str(part["file"])
        if not path.is_file() or _sha256(path) != part.get("sha256"):
            raise ValueError("capture checkpoint part hash mismatch")
        tensor = load_file(str(path))["residuals"]
        if tensor.ndim != 3 or tensor.shape[0] != part["end"] - part["start"]:
            raise ValueError("capture checkpoint part shape mismatch")
        shape = (int(tensor.shape[1]), int(tensor.shape[2]))
        if expected_shape is None:
            expected_shape = shape
        elif expected_shape != shape:
            raise ValueError("capture checkpoint layer/hidden shape drift")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("capture checkpoint contains non-finite values")
        tensors.append(tensor)
        expected_start = int(part["end"])
    if state.get("completed_rows") != expected_start:
        raise ValueError("capture checkpoint completed row count mismatch")
    return state, tensors


def capture_residual_cache(
    model: Any,
    rows: list[GeometryRow],
    batch_size: int,
    output_dir: Path,
    *,
    system_prompt: str,
    metadata: dict[str, object] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Measure every row once and atomically publish a verified cache."""

    if not rows:
        raise ValueError("rows must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "residuals.safetensors": output_dir / "residuals.safetensors",
        "row_index.jsonl": output_dir / "row_index.jsonl",
        "manifest.json": output_dir / "manifest.json",
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"cache output already exists: {existing[0].name}")
    temporary_paths = {
        name: path.with_name(f"{path.name}.tmp")
        for name, path in final_paths.items()
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    state_path = output_dir / "capture_state.json"
    parts_dir = output_dir / "capture_parts"
    fingerprint = _capture_fingerprint(rows, batch_size, system_prompt, metadata)
    if state_path.exists():
        if not parts_dir.is_dir():
            raise ValueError("capture checkpoint is missing its parts directory")
        state, residual_batches = _load_capture_state(
            state_path, parts_dir, fingerprint
        )
    else:
        if parts_dir.exists():
            raise ValueError("capture parts exist without a checkpoint")
        parts_dir.mkdir()
        state = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "rows": len(rows),
            "batch_size": batch_size,
            "completed_rows": 0,
            "parts": [],
        }
        _write_json_atomic(state_path, state)
        residual_batches = []
    completed = int(state["completed_rows"])
    if state.get("rows") != len(rows) or completed > len(rows):
        raise ValueError("capture checkpoint row count mismatch")
    resumed_rows = completed
    expected_shape: tuple[int, int] | None = (
        (int(residual_batches[0].shape[1]), int(residual_batches[0].shape[2]))
        if residual_batches
        else None
    )
    prompts = [Prompt(system=system_prompt, user=row.prompt) for row in rows]
    if progress is not None and completed:
        progress(completed, len(rows))
    try:
        for batch in model.iter_residual_batches(prompts[completed:], batch_size):
            if not isinstance(batch, Tensor) or batch.ndim != 3:
                raise ValueError("residual batch must have shape [rows, layers, hidden]")
            if batch.shape[0] <= 0:
                raise ValueError("residual batch must not be empty")
            shape = (int(batch.shape[1]), int(batch.shape[2]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError("residual layer/hidden shape drift")
            cpu_batch = batch.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(cpu_batch).all()):
                raise ValueError("residual cache contains non-finite values")
            completed += int(cpu_batch.shape[0])
            if completed > len(rows):
                raise ValueError("model returned more residual rows than requested")
            start = completed - int(cpu_batch.shape[0])
            part_name = f"part_{start:08d}_{completed:08d}.safetensors"
            part_path = parts_dir / part_name
            part_temporary = parts_dir / f"{part_name}.tmp"
            part_temporary.unlink(missing_ok=True)
            try:
                save_file({"residuals": cpu_batch}, str(part_temporary))
                part_temporary.replace(part_path)
            except BaseException:
                part_temporary.unlink(missing_ok=True)
                raise
            state["parts"].append(
                {
                    "file": part_name,
                    "start": start,
                    "end": completed,
                    "sha256": _sha256(part_path),
                }
            )
            state["completed_rows"] = completed
            _write_json_atomic(state_path, state)
            if progress is not None:
                progress(completed, len(rows))
        if completed != len(rows):
            raise ValueError(
                f"model returned {completed} residual rows, expected {len(rows)}"
            )

        # Reload every immutable part from disk so finalization verifies the
        # same representation that a resumed process would use.
        state, residual_batches = _load_capture_state(
            state_path, parts_dir, fingerprint
        )
        residuals = torch.cat(residual_batches, dim=0).contiguous()
        index = text_free_row_index(rows)
        save_file({"residuals": residuals}, str(temporary_paths["residuals.safetensors"]))
        _write_jsonl(temporary_paths["row_index.jsonl"], index)

        files = {
            name: {
                "bytes": temporary_paths[name].stat().st_size,
                "sha256": _sha256(temporary_paths[name]),
            }
            for name in ("residuals.safetensors", "row_index.jsonl")
        }
        manifest: dict[str, Any] = {
            "status": "PASS",
            "schema_version": 1,
            "measurement_position": "first_generated_token",
            "rows": len(rows),
            "layers": int(residuals.shape[1]),
            "hidden_size": int(residuals.shape[2]),
            "tensor_dtype": str(residuals.dtype).removeprefix("torch."),
            "files": files,
            "capture": {
                "batch_size": batch_size,
                "parts": len(state["parts"]),
                "resumed_rows": resumed_rows,
                "fingerprint": fingerprint,
            },
            "metadata": dict(metadata or {}),
        }
        _write_json(temporary_paths["manifest.json"], manifest)

        # The manifest is the completion marker and is published last.
        temporary_paths["residuals.safetensors"].replace(
            final_paths["residuals.safetensors"]
        )
        temporary_paths["row_index.jsonl"].replace(final_paths["row_index.jsonl"])
        temporary_paths["manifest.json"].replace(final_paths["manifest.json"])
        for part in state["parts"]:
            (parts_dir / str(part["file"])).unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        with suppress(OSError):
            parts_dir.rmdir()
        return manifest
    except BaseException:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise


def load_residual_cache(
    output_dir: Path,
) -> tuple[list[dict[str, object]], Tensor, dict[str, Any]]:
    """Load a cache only after verifying its completion marker and hashes."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("cache manifest is not PASS")
    for name in ("residuals.safetensors", "row_index.jsonl"):
        path = output_dir / name
        expected = manifest["files"][name]["sha256"]
        if _sha256(path) != expected:
            raise ValueError(f"cache hash mismatch: {name}")

    with (output_dir / "row_index.jsonl").open(encoding="utf-8") as stream:
        index = [json.loads(line) for line in stream if line.strip()]
    residuals = load_file(str(output_dir / "residuals.safetensors"))["residuals"]
    if len(index) != manifest["rows"] or residuals.shape[0] != manifest["rows"]:
        raise ValueError("cache row count mismatch")
    if tuple(residuals.shape[1:]) != (
        manifest["layers"],
        manifest["hidden_size"],
    ):
        raise ValueError("cache tensor shape mismatch")
    return index, residuals, manifest
