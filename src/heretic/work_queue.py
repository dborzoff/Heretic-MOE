# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable render-farm style trial queue for parallel HereticMOE workers."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
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
    schema_version: int
    first_task_id: int
    task_count: int
    last_task_id_exclusive: int
    exploration_task_count: int
    target_trial_count: int
    tpe_concurrency: int
    journal_base_trial_count: int
    journal_base_size_bytes: int
    journal_base_sha256: str


@dataclass(frozen=True)
class QueueTaskRecord:
    task_id: int
    task_kind: str
    state: str
    worker_id: str | None
    attempt: int
    trial_number: int | None
    trial_state: str | None


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

    def _connect_readonly(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.path.as_posix()}?mode=ro",
            uri=True,
            timeout=60.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    def initialize(
        self,
        *,
        first_task_id: int,
        task_count: int,
        exploration_task_count: int = 0,
        target_trial_count: int,
        tpe_concurrency: int,
        journal_base_trial_count: int,
        journal_base_size_bytes: int,
        journal_base_sha256: str,
    ) -> None:
        if first_task_id < 0:
            raise ValueError("first_task_id cannot be negative")
        if task_count < 0:
            raise ValueError("task_count cannot be negative")
        if not 0 <= exploration_task_count <= task_count:
            raise ValueError("exploration_task_count must be within the task range")
        if target_trial_count != first_task_id + task_count:
            raise ValueError(
                "target_trial_count must equal first_task_id + task_count"
            )
        if tpe_concurrency <= 0:
            raise ValueError("tpe_concurrency must be positive")
        if journal_base_trial_count < 0:
            raise ValueError("journal_base_trial_count cannot be negative")
        if not first_task_id <= journal_base_trial_count <= target_trial_count:
            raise ValueError(
                "journal_base_trial_count must be between the completed-task "
                "prefix and target_trial_count"
            )
        if journal_base_size_bytes < 0:
            raise ValueError("journal_base_size_bytes cannot be negative")
        normalized_base_sha256 = journal_base_sha256.strip().lower()
        if len(normalized_base_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_base_sha256
        ):
            raise ValueError("journal_base_sha256 must be a SHA-256 hex digest")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        expected_last = first_task_id + task_count
        with closing(self._connect()) as connection:
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
                "schema_version": "3",
                "first_task_id": str(first_task_id),
                "task_count": str(task_count),
                "last_task_id_exclusive": str(expected_last),
                "exploration_task_count": str(exploration_task_count),
                "target_trial_count": str(target_trial_count),
                "tpe_concurrency": str(tpe_concurrency),
                "journal_base_trial_count": str(journal_base_trial_count),
                "journal_base_size_bytes": str(journal_base_size_bytes),
                "journal_base_sha256": normalized_base_sha256,
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
        with closing(self._connect()) as connection:
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
                # TPE may run with a bounded constant-liar batch, but it must
                # never start before the complete Random/Sobol prefix. Once
                # exploration is complete, keep at most the immutable contract
                # limit in flight. This preserves exact task permits while a
                # faster GPU naturally consumes more of the queue.
                exploration_blocked = connection.execute(
                    """
                    SELECT 1
                    FROM tasks
                    WHERE task_id < ? AND task_kind != 'tpe'
                        AND state != 'complete'
                    LIMIT 1
                    """,
                    (task_id,),
                ).fetchone()
                tpe_concurrency_row = connection.execute(
                    "SELECT value FROM queue_meta WHERE key = 'tpe_concurrency'"
                ).fetchone()
                if tpe_concurrency_row is None:
                    raise RuntimeError(
                        f"Queue {self.path} has no tpe_concurrency contract"
                    )
                claimed_tpe = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM tasks
                        WHERE task_kind = 'tpe' AND state = 'claimed'
                        """
                    ).fetchone()[0]
                )
                if exploration_blocked is not None or claimed_tpe >= int(
                    tpe_concurrency_row["value"]
                ):
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
        with closing(self._connect()) as connection:
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
        with closing(self._connect()) as connection:
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

        with closing(self._connect()) as connection:
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

        with closing(self._connect()) as connection:
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

        with closing(self._connect_readonly()) as connection:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM queue_meta")
            }
        required = {
            "schema_version",
            "first_task_id",
            "task_count",
            "last_task_id_exclusive",
            "exploration_task_count",
            "target_trial_count",
            "tpe_concurrency",
            "journal_base_trial_count",
            "journal_base_size_bytes",
            "journal_base_sha256",
        }
        if set(metadata) != required:
            raise RuntimeError(f"Invalid queue metadata in {self.path}: {metadata}")
        numeric = required - {"journal_base_sha256"}
        try:
            values = {key: int(metadata[key]) for key in numeric}
        except ValueError as error:
            raise RuntimeError(
                f"Invalid numeric queue metadata in {self.path}: {metadata}"
            ) from error
        return QueueContract(
            **values,
            journal_base_sha256=metadata["journal_base_sha256"],
        )

    def task_records(self) -> list[QueueTaskRecord]:
        """Return text-free task state for journal reconciliation."""

        with closing(self._connect_readonly()) as connection:
            rows = connection.execute(
                """
                SELECT task_id, task_kind, state, worker_id, attempt,
                    trial_number, trial_state
                FROM tasks
                ORDER BY task_id
                """
            ).fetchall()
        return [
            QueueTaskRecord(
                task_id=int(row["task_id"]),
                task_kind=str(row["task_kind"]),
                state=str(row["state"]),
                worker_id=(
                    None if row["worker_id"] is None else str(row["worker_id"])
                ),
                attempt=int(row["attempt"]),
                trial_number=(
                    None
                    if row["trial_number"] is None
                    else int(row["trial_number"])
                ),
                trial_state=(
                    None if row["trial_state"] is None else str(row["trial_state"])
                ),
            )
            for row in rows
        ]

    def stats(self) -> QueueStats:
        counts = {"pending": 0, "claimed": 0, "complete": 0, "failed": 0}
        with closing(self._connect()) as connection:
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM tasks GROUP BY state"
            ):
                counts[str(row["state"])] = int(row["count"])
        return QueueStats(**counts)
