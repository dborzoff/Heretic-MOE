from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from heretic.language_map_cache import load_residual_cache
from heretic.language_map_data import GeometryRow
from heretic.language_map_parallel import (
    capture_claimed_ranges,
    finalize_range_cache,
)
from heretic.range_work_queue import RangeWorkQueue


class IndexedResidualModel:
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.measured: list[int] = []

    def iter_residual_batches(self, prompts, batch_size):
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            time.sleep(self.delay)
            indices = [int(prompt.user.removeprefix("row-")) for prompt in batch]
            self.measured.extend(indices)
            yield torch.tensor(indices, dtype=torch.float32).view(-1, 1, 1).expand(
                -1, 2, 3
            )


def _rows(count: int) -> list[GeometryRow]:
    return [
        GeometryRow(
            canonical_id=f"C{index:04d}",
            row_id=f"EN-C{index:04d}",
            language="en",
            direction="safe",
            category_id="C01",
            prompt=f"row-{index}",
            source_path=Path("fixture.jsonl"),
            source_line=index + 1,
        )
        for index in range(count)
    ]


def _initialized_queue(tmp_path: Path, rows: int, rows_per_task: int) -> RangeWorkQueue:
    queue = RangeWorkQueue(tmp_path / "capture_queue.sqlite3")
    queue.initialize(
        row_count=rows,
        rows_per_task=rows_per_task,
        fingerprint="d" * 64,
    )
    return queue


def test_resident_workers_dynamically_share_ranges_and_merge_canonical_order(
    tmp_path: Path,
) -> None:
    rows = _rows(40)
    queue = _initialized_queue(tmp_path, rows=40, rows_per_task=2)
    parts = tmp_path / "parts"
    slow = IndexedResidualModel(delay=0.02)
    fast = IndexedResidualModel(delay=0.001)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                capture_claimed_ranges,
                model,
                rows,
                queue,
                parts,
                worker_id=worker_id,
                batch_size=2,
                system_prompt="system",
            )
            for model, worker_id in ((slow, "gpu-0"), (fast, "gpu-1"))
        ]
        results = [future.result() for future in futures]

    assert results[1]["tasks"] > results[0]["tasks"]
    assert sorted(slow.measured + fast.measured) == list(range(40))
    manifest = finalize_range_cache(
        rows,
        queue,
        parts,
        tmp_path / "cache",
        metadata={"mode": "test"},
    )
    index, residuals, loaded = load_residual_cache(tmp_path / "cache")
    assert manifest["rows"] == loaded["rows"] == len(index) == 40
    assert residuals[:, 0, 0].tolist() == list(range(40))
    assert manifest["capture"]["workers"]["gpu-1"]["tasks"] > 1


def test_one_range_contains_multiple_model_batches_but_writes_one_part(
    tmp_path: Path,
) -> None:
    rows = _rows(5)
    queue = _initialized_queue(tmp_path, rows=5, rows_per_task=5)
    parts = tmp_path / "parts"
    model = IndexedResidualModel()

    result = capture_claimed_ranges(
        model,
        rows,
        queue,
        parts,
        worker_id="gpu-0",
        batch_size=2,
        system_prompt="system",
    )

    assert result == {"worker_id": "gpu-0", "tasks": 1, "rows": 5}
    assert model.measured == [0, 1, 2, 3, 4]
    assert len(list(parts.glob("*.safetensors"))) == 1


def test_finalization_rejects_tampered_range_before_publishing_manifest(
    tmp_path: Path,
) -> None:
    rows = _rows(4)
    queue = _initialized_queue(tmp_path, rows=4, rows_per_task=4)
    parts = tmp_path / "parts"
    capture_claimed_ranges(
        IndexedResidualModel(),
        rows,
        queue,
        parts,
        worker_id="gpu-0",
        batch_size=2,
        system_prompt="system",
    )
    part = next(parts.glob("*.safetensors"))
    part.write_bytes(part.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="part hash mismatch"):
        finalize_range_cache(rows, queue, parts, tmp_path / "cache", metadata={})

    assert not (tmp_path / "cache" / "manifest.json").exists()

