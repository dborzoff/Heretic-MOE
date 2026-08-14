# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unified text-private Rich stage UI for Heretic-MOE controllers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

_SENSITIVE_KEY_PARTS = ("prompt", "response", "answer", "text", "payload")


def _clean_label(value: object, *, fallback: str, max_length: int = 64) -> str:
    text = " ".join(str(value).split())
    if not text:
        text = fallback
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def _sensitive_key(key: object) -> bool:
    lowered = str(key).strip().lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def _public_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("summary values must be finite")
        return f"{value:.6g}"
    if isinstance(value, str):
        return _clean_label(value, fallback="value", max_length=120)
    raise TypeError(f"summary value must be a scalar, got {type(value).__name__}")


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    if seconds >= 90:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


class PipelineUI:
    """Small controller-facing API for one active pipeline stage."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        transient: bool = True,
        refresh_per_second: float = 10.0,
        non_tty_update_interval: float = 5.0,
    ) -> None:
        self.console = console or Console()
        self.non_tty_update_interval = max(0.0, float(non_tty_update_interval))
        self._rich = bool(self.console.is_terminal)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.fields[rate]:.1f} rows/s"),
            TimeRemainingColumn(),
            console=self.console,
            transient=transient,
            refresh_per_second=refresh_per_second,
            disable=not self._rich,
        )
        self._closed = False
        self._active = False
        self._stage = ""
        self._started = 0.0
        self._overall: TaskID | None = None
        self._overall_total = 0
        self._workers: dict[str, dict[str, Any]] = {}
        self._last_compact_update = 0.0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PipelineUI is closed")

    def _require_active(self) -> None:
        self._require_open()
        if not self._active:
            raise RuntimeError("no active stage")

    def stage(
        self,
        name: object,
        *,
        total: int,
        workers: Sequence[object] = (),
        description: object | None = None,
    ) -> None:
        """Start a stage header and one overall progress with per-worker rows."""
        self._require_open()
        if self._active:
            raise RuntimeError("finish the active stage before starting another")
        if int(total) <= 0:
            raise ValueError("stage total must be positive")
        self._stage = _clean_label(name, fallback="stage")
        self._started = time.monotonic()
        self._overall_total = int(total)
        self._active = True
        self._workers = {}
        self._last_compact_update = 0.0
        title = (
            self._stage
            if description is None
            else _clean_label(description, fallback=self._stage, max_length=120)
        )
        if self._rich:
            self.console.print(f"▶ {self._stage} | {title}", style="bold cyan")
            self._progress.start()
            self._overall = self._progress.add_task(
                self._stage, total=int(total), rate=0.0
            )
        else:
            self.console.print(f"STAGE {self._stage} | total {int(total)}")
            self._overall = None
        for worker in workers:
            self.add_worker(worker, total=int(total))

    def note(self, message: object) -> None:
        """Print one short text-private status without breaking the live display."""

        self._require_active()
        value = _clean_label(message, fallback="working", max_length=180)
        self.console.print(f"  {value}", style="dim")

    def add_worker(
        self,
        worker_id: object,
        *,
        total: int,
        label: object | None = None,
    ) -> None:
        """Add one GPU/worker row to the active stage."""
        self._require_active()
        if int(total) <= 0:
            raise ValueError("worker total must be positive")
        worker = _clean_label(worker_id, fallback="worker")
        if worker in self._workers:
            raise ValueError(f"duplicate worker: {worker}")
        display = _clean_label(label, fallback=worker) if label is not None else worker
        state: dict[str, Any] = {
            "completed": 0,
            "total": int(total),
            "started": time.monotonic(),
            "label": display,
        }
        if self._rich:
            state["task"] = self._progress.add_task(display, total=int(total), rate=0.0)
        else:
            self.console.print(f"WORKER {self._stage} | {display} | total {int(total)}")
        self._workers[worker] = state

    def _worker_rate(self, state: Mapping[str, Any]) -> float:
        elapsed = max(time.monotonic() - float(state["started"]), 1e-6)
        return float(state["completed"]) / elapsed

    def _refresh_overall(self) -> None:
        if not self._rich or self._overall is None:
            return
        completed = sum(int(state["completed"]) for state in self._workers.values())
        total = self._overall_total
        elapsed = max(time.monotonic() - self._started, 1e-6)
        self._progress.update(
            self._overall,
            completed=min(completed, total),
            total=total,
            rate=completed / elapsed,
        )

    def update_worker(
        self,
        worker_id: object,
        *,
        completed: int,
        total: int | None = None,
        rate: float | None = None,
    ) -> None:
        """Update one worker row and the shared overall progress."""
        self._require_active()
        worker = _clean_label(worker_id, fallback="worker")
        if worker not in self._workers:
            raise KeyError(f"unknown worker: {worker}")
        state = self._workers[worker]
        new_total = int(state["total"] if total is None else total)
        if new_total <= 0:
            raise ValueError("worker total must be positive")
        new_completed = int(completed)
        if not 0 <= new_completed <= new_total:
            raise ValueError("worker completed must be within [0, total]")
        state["completed"] = new_completed
        state["total"] = new_total
        effective_rate = self._worker_rate(state) if rate is None else float(rate)
        if not math.isfinite(effective_rate) or effective_rate < 0:
            raise ValueError("worker rate must be a nonnegative finite value")
        if self._rich:
            self._progress.update(
                state["task"],
                completed=new_completed,
                total=new_total,
                rate=effective_rate,
            )
            self._refresh_overall()
            return
        # Multi-worker controllers print one authoritative global line from
        # update_overall(); individual rows would otherwise scroll the console
        # twice per refresh and still omit the true shared queue count.
        if len(self._workers) > 1:
            return
        now = time.monotonic()
        if (
            self.non_tty_update_interval == 0.0
            or new_completed == new_total
            or now - self._last_compact_update >= self.non_tty_update_interval
        ):
            remaining = max(new_total - new_completed, 0)
            eta = remaining / effective_rate if effective_rate > 0 else math.inf
            self.console.print(
                f"PROGRESS {self._stage} | {state['label']} {new_completed}/{new_total} | "
                f"{effective_rate:.1f} rows/s | ETA {_format_duration(eta)}"
            )
            self._last_compact_update = now

    def update_overall(
        self,
        *,
        completed: int,
        total: int | None = None,
        rate: float | None = None,
    ) -> None:
        """Set authoritative global progress for dynamic shared queues."""

        self._require_active()
        new_total = self._overall_total if total is None else int(total)
        new_completed = int(completed)
        if new_total <= 0 or not 0 <= new_completed <= new_total:
            raise ValueError("overall progress must be within [0, total]")
        self._overall_total = new_total
        elapsed = max(time.monotonic() - self._started, 1e-6)
        effective_rate = new_completed / elapsed if rate is None else float(rate)
        if not math.isfinite(effective_rate) or effective_rate < 0:
            raise ValueError("overall rate must be a nonnegative finite value")
        if self._rich and self._overall is not None:
            self._progress.update(
                self._overall,
                completed=new_completed,
                total=new_total,
                rate=effective_rate,
            )
            return
        now = time.monotonic()
        if (
            self.non_tty_update_interval == 0.0
            or new_completed == new_total
            or now - self._last_compact_update >= self.non_tty_update_interval
        ):
            workers = " ".join(
                f"{state['label']} {int(state['completed'])}"
                for state in self._workers.values()
            )
            worker_suffix = f" | {workers}" if workers else ""
            remaining = max(new_total - new_completed, 0)
            eta = remaining / effective_rate if effective_rate > 0 else math.inf
            self.console.print(
                f"PROGRESS {self._stage} | {new_completed}/{new_total}"
                f"{worker_suffix} | {effective_rate * 60:.1f} trials/min | "
                f"ETA {_format_duration(eta)}"
            )
            self._last_compact_update = now

    def finish_stage(self, summary: Mapping[object, object]) -> dict[str, str]:
        """Stop the stage and render a concise sanitized PASS/FAIL summary."""
        self._require_active()
        if not isinstance(summary, Mapping):
            raise TypeError("stage summary must be a mapping")
        rows: list[tuple[str, str]] = []
        status = "PASS"
        next_action: str | None = None
        for key, value in summary.items():
            if _sensitive_key(key):
                continue
            name = _clean_label(key, fallback="metric", max_length=48)
            rendered = _public_value(value)
            if name.lower() == "status":
                status = rendered.upper()
            elif name.lower() == "next":
                next_action = rendered
            else:
                rows.append((name, rendered))
        elapsed = time.monotonic() - self._started
        rows.append(("elapsed", _format_duration(elapsed)))
        if self._rich:
            if self._progress.live.is_started:
                self._progress.stop()
            suffix = " | ".join(f"{name} {value}" for name, value in rows)
            icon = "✓" if status == "PASS" else "✗"
            line = f"{icon} {status} {self._stage}"
            if suffix:
                line += f" | {suffix}"
            if next_action is not None:
                line += f" → {next_action}"
            self.console.print(
                line,
                style="bold green" if status == "PASS" else "bold red",
            )
        else:
            suffix = " ".join(f"{name}={value}" for name, value in rows)
            next_suffix = f" next={next_action}" if next_action is not None else ""
            self.console.print(
                f"SUMMARY {self._stage} | {status} | {suffix}{next_suffix}"
            )
        self._active = False
        self._overall = None
        self._workers = {}
        return {"stage": self._stage, "status": status, "elapsed": _format_duration(elapsed)}

    def result(
        self,
        label: object,
        values: Mapping[object, object],
    ) -> None:
        """Print one compact, sanitized result line after a completed stage."""

        self._require_open()
        if not isinstance(values, Mapping):
            raise TypeError("result values must be a mapping")
        rendered: list[str] = []
        for key, value in values.items():
            if _sensitive_key(key):
                continue
            name = _clean_label(key, fallback="metric", max_length=48)
            rendered.append(f"{name} {_public_value(value)}")
        title = _clean_label(label, fallback="Result", max_length=48)
        suffix = " | ".join(rendered) if rendered else "none"
        self.console.print(f"  {title}: {suffix}", style="dim")

    def fail_stage(self, error: object | None = None) -> dict[str, str] | None:
        """Finish an active stage as failed without masking the original exception."""

        if self._closed or not self._active:
            return None
        summary: dict[str, object] = {"status": "FAIL"}
        if error is not None:
            summary["reason"] = type(error).__name__
        return self.finish_stage(summary)

    def close(self) -> None:
        """Release the live progress; idempotent."""
        if self._closed:
            return
        if self._active and self._rich and self._progress.live.is_started:
            self._progress.stop()
        self._active = False
        self._closed = True
