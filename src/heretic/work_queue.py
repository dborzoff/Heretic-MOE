# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable render-farm style trial queue for parallel HereticMOE workers."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkItem:
    task_id: int
    attempt: int
    task_kind: str


@dataclass(frozen=True)
class QueueStats:
    pending: int
    claimed: int
    complete: int
    failed: int

    @property
    def total(self) -> int:
        return self.pending + self.claimed + self.complete + self.failed


@dataclass(frozen=True)
class QueueContract:
    first_task_id: int
    task_count: int
    last_task_id_exclusive: int
    exploration_task_count: int


class TrialWorkQueue:
    """A small SQLite queue shared by long-lived GPU worker processes.

    SQLite's ``BEGIN IMMEDIATE`` transaction is the dispatch lock. A fast worker
    claims another row as soon as it completes a trial, while slower workers keep
    their model resident and simply finish fewer rows. The queue contains work
    permits, not model parameters; Optuna still creates the actual adaptive trial
    only after a worker has claimed a permit.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def initialize(
        self,
        *,
        first_task_id: int,
        task_count: int,
        exploration_task_count: int = 0,
    ) -> None:
        if first_task_id < 0:
            raise ValueError("first_task_id cannot be negative")
        if task_count < 0:
            raise ValueError("task_count cannot be negative")
        if not 0 <= exploration_task_count <= task_count:
            raise ValueError("exploration_task_count must be within the task range")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        expected_last = first_task_id + task_count
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS queue_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id INTEGER PRIMARY KEY,
                    task_kind TEXT NOT NULL CHECK (
                        task_kind IN ('random', 'sobol', 'tpe')
                    ),
                    state TEXT NOT NULL CHECK (
                        state IN ('pending', 'claimed', 'complete', 'failed')
                    ),
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    claimed_at REAL,
                    finished_at REAL,
                    trial_number INTEGER,
                    trial_state TEXT,
                    error_type TEXT
                )
                """
            )
            metadata = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM queue_meta")
            }
            expected = {
                "first_task_id": str(first_task_id),
                "task_count": str(task_count),
                "last_task_id_exclusive": str(expected_last),
                "exploration_task_count": str(exploration_task_count),
            }
            if metadata and metadata != expected:
                raise RuntimeError(
                    f"Queue contract mismatch for {self.path}: "
                    f"found {metadata}, expected {expected}"
                )
            if not metadata:
                connection.executemany(
                    "INSERT INTO queue_meta(key, value) VALUES (?, ?)",
                    expected.items(),
                )
                connection.executemany(
                    """
                    INSERT INTO tasks(task_id, task_kind, state)
                    VALUES (?, ?, 'pending')
                    """,
                    (
                        (
                            task_id,
                            (
                                "random"
                                if offset < exploration_task_count and offset % 2 == 0
                                else (
                                    "sobol"
                                    if offset < exploration_task_count
                                    else "tpe"
                                )
                            ),
                        )
                        for offset, task_id in enumerate(
                            range(first_task_id, expected_last)
                        )
                    ),
                )
            actual = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            if actual != task_count:
                raise RuntimeError(
                    f"Queue {self.path} contains {actual} tasks, expected {task_count}"
                )
            connection.commit()

    def claim(self, worker_id: str) -> WorkItem | None:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT task_id, attempt, task_kind
                FROM tasks
                WHERE state = 'pending'
                ORDER BY task_id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            task_id = int(row["task_id"])
            task_kind = str(row["task_kind"])
            if task_kind == "tpe":
                blocked = connection.execute(
                    """
                    SELECT 1
                    FROM tasks
                    WHERE task_id < ? AND state != 'complete'
                    LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()
                if blocked is not None:
                    connection.commit()
                    return None
            attempt = int(row["attempt"]) + 1
            connection.execute(
                """
                UPDATE tasks
                SET state = 'claimed', worker_id = ?, attempt = ?, claimed_at = ?,
                    finished_at = NULL, trial_number = NULL, trial_state = NULL,
                    error_type = NULL
                WHERE task_id = ? AND state = 'pending'
                """,
                (worker_id, attempt, time.time(), task_id),
            )
            connection.commit()
            return WorkItem(
                task_id=task_id,
                attempt=attempt,
                task_kind=task_kind,
            )

    def finish(
        self,
        item: WorkItem,
        *,
        trial_number: int,
        trial_state: str,
    ) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE tasks
                SET state = 'complete', finished_at = ?, trial_number = ?,
                    trial_state = ?
                WHERE task_id = ? AND state = 'claimed' AND attempt = ?
                """,
                (
                    time.time(),
                    int(trial_number),
                    str(trial_state),
                    item.task_id,
                    item.attempt,
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError(f"Lost queue claim for task {item.task_id}")

    def fail(self, item: WorkItem, *, error_type: str, retry: bool) -> None:
        state = "pending" if retry else "failed"
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE tasks
                SET state = ?, finished_at = ?, error_type = ?
                WHERE task_id = ? AND state = 'claimed' AND attempt = ?
                """,
                (state, time.time(), error_type, item.task_id, item.attempt),
            ).rowcount
            if updated != 1:
                raise RuntimeError(f"Lost queue claim for task {item.task_id}")

    def release_worker(self, worker_id: str) -> int:
        """Return claims owned by a dead worker to the pending queue."""

        with self._connect() as connection:
            return connection.execute(
                """
                UPDATE tasks
                SET state = 'pending', worker_id = NULL, claimed_at = NULL,
                    error_type = 'worker_released'
                WHERE state = 'claimed' AND worker_id = ?
                """,
                (worker_id,),
            ).rowcount

    def claimed_workers(self) -> list[str]:
        """Return stable worker identifiers that still own queue claims."""

        with self._connect() as connection:
            return [
                str(row["worker_id"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT worker_id
                    FROM tasks
                    WHERE state = 'claimed' AND worker_id IS NOT NULL
                    ORDER BY worker_id
                    """
                )
            ]

    def contract(self) -> QueueContract:
        """Read the immutable range and exploration prefix of this queue."""

        with self._connect() as connection:
            metadata = {
                str(row["key"]): int(row["value"])
                for row in connection.execute("SELECT key, value FROM queue_meta")
            }
        required = {
            "first_task_id",
            "task_count",
            "last_task_id_exclusive",
            "exploration_task_count",
        }
        if set(metadata) != required:
            raise RuntimeError(f"Invalid queue metadata in {self.path}: {metadata}")
        return QueueContract(**metadata)

    def stats(self) -> QueueStats:
        counts = {"pending": 0, "claimed": 0, "complete": 0, "failed": 0}
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM tasks GROUP BY state"
            ):
                counts[str(row["state"])] = int(row["count"])
        return QueueStats(**counts)
