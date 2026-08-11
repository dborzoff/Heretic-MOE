# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable global row-range queue for resident geometry-map workers."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RangeWorkItem:
    task_id: int
    start: int
    end: int
    attempt: int
    worker_id: str


@dataclass(frozen=True)
class RangeTaskRecord:
    task_id: int
    start: int
    end: int
    state: str
    worker_id: str | None
    attempt: int
    part_file: str | None
    sha256: str | None
    shape: tuple[int, int, int] | None


@dataclass(frozen=True)
class RangeQueueStats:
    pending: int
    claimed: int
    complete: int
    failed: int
    total_rows: int
    complete_rows: int

    @property
    def total_tasks(self) -> int:
        return self.pending + self.claimed + self.complete + self.failed


class RangeWorkQueue:
    """SQLite-backed queue whose tasks are non-overlapping global row ranges."""

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
        row_count: int,
        rows_per_task: int,
        fingerprint: str,
    ) -> None:
        if row_count <= 0:
            raise ValueError("row_count must be positive")
        if rows_per_task <= 0:
            raise ValueError("rows_per_task must be positive")
        normalized_fingerprint = fingerprint.strip().lower()
        if len(normalized_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_fingerprint
        ):
            raise ValueError("fingerprint must be a SHA-256 hex digest")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        expected = {
            "schema_version": "1",
            "row_count": str(row_count),
            "rows_per_task": str(rows_per_task),
            "fingerprint": normalized_fingerprint,
        }
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS queue_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id INTEGER PRIMARY KEY,
                    start_row INTEGER NOT NULL,
                    end_row INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('pending', 'claimed', 'complete', 'failed')
                    ),
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    claimed_at REAL,
                    finished_at REAL,
                    part_file TEXT,
                    sha256 TEXT,
                    shape_rows INTEGER,
                    shape_layers INTEGER,
                    shape_hidden INTEGER,
                    error_type TEXT,
                    CHECK (start_row >= 0 AND end_row > start_row)
                )
                """
            )
            found = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM queue_meta")
            }
            if found and found != expected:
                raise RuntimeError(
                    f"Range queue contract mismatch: found {found}, expected {expected}"
                )
            if not found:
                connection.executemany(
                    "INSERT INTO queue_meta(key, value) VALUES (?, ?)",
                    expected.items(),
                )
                connection.executemany(
                    """
                    INSERT INTO tasks(task_id, start_row, end_row, state)
                    VALUES (?, ?, ?, 'pending')
                    """,
                    (
                        (task_id, start, min(start + rows_per_task, row_count))
                        for task_id, start in enumerate(
                            range(0, row_count, rows_per_task)
                        )
                    ),
                )
            count, covered = connection.execute(
                "SELECT COUNT(*), SUM(end_row - start_row) FROM tasks"
            ).fetchone()
            if int(count) != (row_count + rows_per_task - 1) // rows_per_task:
                raise RuntimeError("Range queue task count mismatch")
            if int(covered) != row_count:
                raise RuntimeError("Range queue row coverage mismatch")
            connection.commit()

    def claim(self, worker_id: str) -> RangeWorkItem | None:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT task_id, start_row, end_row, attempt
                FROM tasks WHERE state = 'pending'
                ORDER BY task_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempt = int(row["attempt"]) + 1
            changed = connection.execute(
                """
                UPDATE tasks
                SET state = 'claimed', worker_id = ?, attempt = ?,
                    claimed_at = ?, finished_at = NULL, error_type = NULL
                WHERE task_id = ? AND state = 'pending'
                """,
                (worker_id, attempt, time.time(), int(row["task_id"])),
            ).rowcount
            if changed != 1:
                raise RuntimeError("Range claim race")
            connection.commit()
            return RangeWorkItem(
                task_id=int(row["task_id"]),
                start=int(row["start_row"]),
                end=int(row["end_row"]),
                attempt=attempt,
                worker_id=worker_id,
            )

    def complete(
        self,
        item: RangeWorkItem,
        *,
        part_file: str,
        sha256: str,
        shape: tuple[int, int, int],
    ) -> None:
        normalized_sha = sha256.strip().lower()
        if len(normalized_sha) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_sha
        ):
            raise ValueError("sha256 must be a SHA-256 hex digest")
        if len(shape) != 3 or shape[0] != item.end - item.start:
            raise ValueError("part shape does not match claimed row range")
        if any(dimension <= 0 for dimension in shape):
            raise ValueError("part shape dimensions must be positive")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE tasks
                SET state = 'complete', finished_at = ?, part_file = ?, sha256 = ?,
                    shape_rows = ?, shape_layers = ?, shape_hidden = ?, error_type = NULL
                WHERE task_id = ? AND state = 'claimed' AND attempt = ?
                    AND worker_id = ?
                """,
                (
                    time.time(),
                    part_file,
                    normalized_sha,
                    *shape,
                    item.task_id,
                    item.attempt,
                    item.worker_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"Lost queue claim for task {item.task_id}")
            connection.commit()

    def fail(self, item: RangeWorkItem, error_type: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE tasks
                SET state = 'failed', finished_at = ?, error_type = ?
                WHERE task_id = ? AND state = 'claimed' AND attempt = ?
                    AND worker_id = ?
                """,
                (
                    time.time(),
                    error_type[:200],
                    item.task_id,
                    item.attempt,
                    item.worker_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"Lost queue claim for task {item.task_id}")
            connection.commit()

    def release_worker(self, worker_id: str) -> int:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE tasks
                SET state = 'pending', worker_id = NULL, claimed_at = NULL,
                    part_file = NULL, sha256 = NULL, shape_rows = NULL,
                    shape_layers = NULL, shape_hidden = NULL,
                    error_type = 'worker_released'
                WHERE state = 'claimed' AND worker_id = ?
                """,
                (worker_id,),
            ).rowcount
            connection.commit()
            return int(changed)

    def requeue_invalid_parts(self, parts_dir: str | Path) -> int:
        parts_dir = Path(parts_dir)
        invalid: list[int] = []
        for record in self.records():
            if record.state != "complete":
                continue
            path = parts_dir / str(record.part_file)
            if (
                record.part_file is None
                or record.sha256 is None
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256
            ):
                invalid.append(record.task_id)
        if not invalid:
            return 0
        placeholders = ",".join("?" for _ in invalid)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                f"""
                UPDATE tasks
                SET state = 'pending', worker_id = NULL, claimed_at = NULL,
                    finished_at = NULL, part_file = NULL, sha256 = NULL,
                    shape_rows = NULL, shape_layers = NULL, shape_hidden = NULL,
                    error_type = 'invalid_part'
                WHERE task_id IN ({placeholders})
                """,
                invalid,
            )
            connection.commit()
        return len(invalid)

    def record(self, task_id: int) -> RangeTaskRecord:
        with closing(self._connect_readonly()) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._record_from_row(row)

    def records(self) -> list[RangeTaskRecord]:
        with closing(self._connect_readonly()) as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
        return [self._record_from_row(row) for row in rows]

    def stats(self) -> RangeQueueStats:
        with closing(self._connect_readonly()) as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count, "
                "SUM(end_row - start_row) AS rows FROM tasks GROUP BY state"
            ).fetchall()
        counts = {state: 0 for state in ("pending", "claimed", "complete", "failed")}
        row_counts = {state: 0 for state in counts}
        for row in rows:
            counts[str(row["state"])] = int(row["count"])
            row_counts[str(row["state"])] = int(row["rows"])
        return RangeQueueStats(
            pending=counts["pending"],
            claimed=counts["claimed"],
            complete=counts["complete"],
            failed=counts["failed"],
            total_rows=sum(row_counts.values()),
            complete_rows=row_counts["complete"],
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> RangeTaskRecord:
        shape = None
        if row["shape_rows"] is not None:
            shape = (
                int(row["shape_rows"]),
                int(row["shape_layers"]),
                int(row["shape_hidden"]),
            )
        return RangeTaskRecord(
            task_id=int(row["task_id"]),
            start=int(row["start_row"]),
            end=int(row["end_row"]),
            state=str(row["state"]),
            worker_id=None if row["worker_id"] is None else str(row["worker_id"]),
            attempt=int(row["attempt"]),
            part_file=None if row["part_file"] is None else str(row["part_file"]),
            sha256=None if row["sha256"] is None else str(row["sha256"]),
            shape=shape,
        )
