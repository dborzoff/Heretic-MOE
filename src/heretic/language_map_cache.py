# SPDX-License-Identifier: AGPL-3.0-or-later

"""Immutable residual-cache capture for multilingual geometry diagnostics."""

from __future__ import annotations

import hashlib
import json
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

    prompts = [Prompt(system=system_prompt, user=row.prompt) for row in rows]
    residual_batches: list[Tensor] = []
    completed = 0
    expected_shape: tuple[int, int] | None = None
    try:
        for batch in model.iter_residual_batches(prompts, batch_size):
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
            residual_batches.append(cpu_batch)
            completed += int(cpu_batch.shape[0])
            if completed > len(rows):
                raise ValueError("model returned more residual rows than requested")
            if progress is not None:
                progress(completed, len(rows))
        if completed != len(rows):
            raise ValueError(
                f"model returned {completed} residual rows, expected {len(rows)}"
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
            "metadata": dict(metadata or {}),
        }
        _write_json(temporary_paths["manifest.json"], manifest)

        # The manifest is the completion marker and is published last.
        temporary_paths["residuals.safetensors"].replace(
            final_paths["residuals.safetensors"]
        )
        temporary_paths["row_index.jsonl"].replace(final_paths["row_index.jsonl"])
        temporary_paths["manifest.json"].replace(final_paths["manifest.json"])
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
