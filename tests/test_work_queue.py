from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from heretic.work_queue import TrialWorkQueue


def _queue(tmp_path: Path) -> TrialWorkQueue:
    queue = TrialWorkQueue(tmp_path / "queue.sqlite3")
    queue.initialize(
        first_task_id=0,
        task_count=1,
        exploration_task_count=1,
        target_trial_count=1,
        tpe_concurrency=1,
        journal_base_trial_count=0,
        journal_base_complete_count=0,
        journal_base_size_bytes=0,
        journal_base_sha256=sha256().hexdigest(),
        queue_seed=7,
    )
    return queue


def test_heartbeat_keeps_claim_alive_and_stale_detection_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((100.0, 100.0, 105.0, 120.0, 140.0))
    monkeypatch.setattr("heretic.work_queue.time.time", lambda: next(clock))
    queue = _queue(tmp_path)
    item = queue.claim("gpu-0")
    assert item is not None

    queue.heartbeat(item, worker_id="gpu-0")
    assert queue.stale_claimed_workers(lease_timeout_seconds=30) == []
    assert queue.stale_claimed_workers(lease_timeout_seconds=30) == ["gpu-0"]


def test_retry_releases_claim_ownership(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    item = queue.claim("gpu-0")
    assert item is not None

    queue.fail(item, error_type="synthetic", retry=True)

    record = queue.task_records()[0]
    assert record.state == "pending"
    assert record.worker_id is None


def test_only_complete_optuna_trials_can_complete_a_permit(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    item = queue.claim("gpu-0")
    assert item is not None

    with pytest.raises(ValueError, match="COMPLETE"):
        queue.finish(item, trial_number=0, trial_state="PRUNED")

    assert queue.task_records()[0].state == "claimed"
