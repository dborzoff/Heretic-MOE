from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from heretic.range_work_queue import RangeWorkQueue


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queue(tmp_path: Path, *, rows: int = 10, rows_per_task: int = 4) -> RangeWorkQueue:
    queue = RangeWorkQueue(tmp_path / "capture.sqlite3")
    queue.initialize(
        row_count=rows,
        rows_per_task=rows_per_task,
        fingerprint="a" * 64,
    )
    return queue


def test_queue_covers_every_row_once_in_global_order(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    assert [(item.start, item.end) for item in queue.records()] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert queue.stats().total_rows == 10


def test_faster_worker_claims_more_ranges_without_static_shards(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    first = queue.claim("gpu-0")
    second = queue.claim("gpu-1")
    assert first is not None and second is not None

    queue.complete(
        first,
        part_file="part_00000000_00000004.safetensors",
        sha256="b" * 64,
        shape=(4, 2, 3),
    )
    third = queue.claim("gpu-0")

    assert third is not None
    assert (first.start, first.end, third.start, third.end) == (0, 4, 8, 10)
    assert queue.claim("gpu-1") is None


def test_release_worker_returns_only_its_claim_to_pending(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    first = queue.claim("gpu-0")
    second = queue.claim("gpu-1")
    assert first is not None and second is not None

    assert queue.release_worker("gpu-0") == 1
    reclaimed = queue.claim("gpu-2")

    assert reclaimed is not None
    assert (reclaimed.start, reclaimed.end) == (first.start, first.end)
    assert queue.record(second.task_id).worker_id == "gpu-1"


def test_contract_mismatch_cannot_reuse_existing_queue(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    with pytest.raises(RuntimeError, match="contract mismatch"):
        queue.initialize(
            row_count=11,
            rows_per_task=4,
            fingerprint="a" * 64,
        )


def test_missing_or_tampered_part_is_requeued(tmp_path: Path) -> None:
    queue = _queue(tmp_path, rows=4)
    parts = tmp_path / "parts"
    parts.mkdir()
    item = queue.claim("gpu-0")
    assert item is not None
    part = parts / "part_00000000_00000004.safetensors"
    part.write_bytes(b"verified")
    queue.complete(
        item,
        part_file=part.name,
        sha256=_sha256(part),
        shape=(4, 2, 3),
    )

    part.write_bytes(b"tampered")
    assert queue.requeue_invalid_parts(parts) == 1

    replacement = queue.claim("gpu-1")
    assert replacement is not None
    assert replacement.task_id == item.task_id
    assert replacement.attempt == item.attempt + 1


def test_stale_attempt_cannot_complete_a_reclaimed_range(tmp_path: Path) -> None:
    queue = _queue(tmp_path, rows=4)
    stale = queue.claim("gpu-0")
    assert stale is not None
    queue.release_worker("gpu-0")
    current = queue.claim("gpu-1")
    assert current is not None

    with pytest.raises(RuntimeError, match="Lost queue claim"):
        queue.complete(
            stale,
            part_file="stale.safetensors",
            sha256="c" * 64,
            shape=(4, 2, 3),
        )

