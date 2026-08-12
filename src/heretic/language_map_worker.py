# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal resident worker entry point for geometry-map capture."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from .language_map_cache import _capture_fingerprint
from .language_map_data import GeometryRow, LanguageFile, load_aligned_corpus
from .language_map_parallel import capture_claimed_ranges
from .range_work_queue import RangeWorkQueue
from .utils import Prompt


def _selected_rows(
    rows: list[GeometryRow], limit_per_cell: int | None
) -> list[GeometryRow]:
    if limit_per_cell is None:
        return rows
    selected: list[GeometryRow] = []
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.direction, row.language)
        count = counts.get(key, 0)
        if count < limit_per_cell:
            selected.append(row)
            counts[key] = count + 1
    return selected


def _load_model(job: dict[str, Any]):
    import torch

    from .config import QuantizationMethod, Settings
    from .model import Model

    cpu_threads = int(job["cpu_threads"])
    torch.set_num_threads(cpu_threads)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    settings = Settings(
        model=str(job["model"]),
        dtypes=[str(job["dtype"])],
        quantization=QuantizationMethod.NONE,
        device_map="auto",
        batch_size=int(job["batch_size"]),
        residual_batch_size=int(job["batch_size"]),
        max_batch_size=int(job.get("max_batch_size", 64)),
        offload_outputs_to_cpu=True,
        seed=int(job["seed"]),
        system_prompt=str(job["system_prompt"]),
    )
    return Model(settings)


def run_worker_job(
    job_path: str | Path,
    *,
    device: str,
    worker_id: str,
    model_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, int | str]:
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("schema_version") != 1:
        raise ValueError("unsupported geometry worker job schema")
    files = [
        LanguageFile(
            language=str(entry["language"]),
            direction=cast(Any, str(entry["direction"])),
            path=Path(entry["path"]),
        )
        for entry in job["files"]
    ]
    languages = tuple(str(language) for language in job["languages"])
    rows = load_aligned_corpus(files, languages, int(job["rows_per_cell"]))
    rows = _selected_rows(rows, job.get("limit_per_cell"))
    fingerprint = _capture_fingerprint(
        rows,
        int(job["batch_size"]),
        str(job["system_prompt"]),
        dict(job["metadata"]),
    )
    if fingerprint != job.get("fingerprint"):
        raise ValueError("geometry worker fingerprint mismatch")
    queue = RangeWorkQueue(job["queue_path"])
    model = (model_factory or _load_model)(job)
    cache_prompts = [
        Prompt(system=str(job["system_prompt"]), user=row.prompt) for row in rows
    ]
    if hasattr(model, "prepare_prompt_cache"):
        prepared = model.prepare_prompt_cache(cache_prompts)
        packed = None
        if hasattr(model, "pin_prompt_cache"):
            packed = model.pin_prompt_cache()
        print(
            json.dumps(
                {
                    "event": "token_cache_ready",
                    "worker_id": worker_id,
                    "rows": int(prepared.get("rows", len(cache_prompts)))
                    if isinstance(prepared, dict)
                    else len(cache_prompts),
                    "tokens": int(packed.get("tokens", 0))
                    if isinstance(packed, dict)
                    else 0,
                    "pinned": bool(packed.get("pinned", False))
                    if isinstance(packed, dict)
                    else False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def progress(_worker_id: str, completed: int, total: int) -> None:
        print(
            json.dumps(
                {
                    "event": "worker_progress",
                    "scope": "global",
                    "worker_id": worker_id,
                    "completed": completed,
                    "total": total,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return capture_claimed_ranges(
        model,
        rows,
        queue,
        job["parts_dir"],
        worker_id=worker_id,
        batch_size=int(job["batch_size"]),
        system_prompt=str(job["system_prompt"]),
        progress=progress,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hereticMOE geometry-map worker")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--worker-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    result = run_worker_job(
        args.job,
        device=args.device,
        worker_id=args.worker_id,
    )
    print(
        json.dumps({"event": "worker_complete", **result}, sort_keys=True), flush=True
    )
