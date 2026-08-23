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
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Column, Table
from rich.text import Text

_SENSITIVE_KEY_PARTS = ("prompt", "response", "answer", "text", "payload")


class _PipelineProgress(Progress):
    """Progress rows with one optional persistent footer renderable."""

    def __init__(self, *columns: Any, **kwargs: Any) -> None:
        self.footer: Any | None = None
        super().__init__(*columns, **kwargs)

    def set_footer(self, renderable: Any | None) -> None:
        self.footer = renderable
        if self.live.is_started:
            self.refresh()

    def get_renderables(self) -> Any:
        yield self.make_tasks_table(self.tasks)
        if self.footer is not None:
            yield self.footer


class _OverallBarColumn(BarColumn):
    def render(self, task: Task) -> Any:
        if task.fields.get("worker"):
            return Text("")
        return super().render(task)


class _OverallPercentColumn(TaskProgressColumn):
    def render(self, task: Task) -> Text:
        if task.fields.get("worker"):
            return Text("")
        return super().render(task)


class _QueueCountColumn(ProgressColumn):
    def __init__(self) -> None:
        super().__init__(table_column=Column(min_width=10, no_wrap=True))

    def render(self, task: Task) -> Text:
        completed = int(task.completed)
        if task.fields.get("worker"):
            return Text(f"{completed} trials")
        total = int(task.total or 0)
        return Text(f"{completed}/{total}")


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
    if seconds >= 3_600:
        total_minutes = int((seconds + 30) // 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}h {minutes}m"
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
        self._progress = _PipelineProgress(
            SpinnerColumn(),
            TextColumn(
                "[bold]{task.description}",
                table_column=Column(min_width=7, max_width=18, no_wrap=True),
            ),
            _OverallBarColumn(bar_width=16),
            _OverallPercentColumn(
                table_column=Column(width=4, no_wrap=True, justify="right")
            ),
            _QueueCountColumn(),
            TextColumn(
                "{task.fields[detail]}",
                table_column=Column(ratio=1, overflow="ellipsis", no_wrap=True),
            ),
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
                self._stage, total=int(total), detail="starting", worker=False
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
            state["task"] = self._progress.add_task(
                display, total=int(total), detail="waiting", worker=True
            )
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
        self._progress.update(
            self._overall,
            completed=min(completed, total),
            total=total,
        )

    def update_worker(
        self,
        worker_id: object,
        *,
        completed: int,
        total: int | None = None,
        rate: float | None = None,
        last_trial_seconds: float | None = None,
        gpu_utilization: float | None = None,
        memory_used_gib: float | None = None,
        memory_total_gib: float | None = None,
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
        telemetry = (
            last_trial_seconds,
            gpu_utilization,
            memory_used_gib,
            memory_total_gib,
        )
        if all(value is not None for value in telemetry):
            if any(not math.isfinite(float(value)) for value in telemetry):
                raise ValueError("worker telemetry must be finite")
            detail = (
                f"last {float(last_trial_seconds):.1f}s | "
                f"GPU {float(gpu_utilization):.0f}% | "
                f"VRAM {float(memory_used_gib):.1f}/"
                f"{float(memory_total_gib):.1f} GiB"
            )
        else:
            remaining = max(new_total - new_completed, 0)
            eta = remaining / effective_rate if effective_rate > 0 else math.inf
            detail = f"{effective_rate:.1f} rows/s | ETA {_format_duration(eta)}"
        if self._rich:
            self._progress.update(
                state["task"],
                completed=new_completed,
                total=new_total,
                detail=detail,
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
            self.console.print(
                f"PROGRESS {self._stage} | {state['label']} {new_completed}/{new_total} | "
                f"{detail}"
            )
            self._last_compact_update = now

    def update_overall(
        self,
        *,
        completed: int,
        total: int | None = None,
        rate: float | None = None,
        elapsed_seconds: float | None = None,
        estimated_total_seconds: float | None = None,
        eta_seconds: float | None = None,
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
        explicit_times = (
            elapsed_seconds,
            estimated_total_seconds,
            eta_seconds,
        )
        if all(value is not None for value in explicit_times):
            detail = (
                f"{effective_rate * 60:.1f} trials/min | "
                f"elapsed {_format_duration(float(elapsed_seconds))} | "
                f"total {_format_duration(float(estimated_total_seconds))} | "
                f"ETA {_format_duration(float(eta_seconds))}"
            )
        else:
            remaining = max(new_total - new_completed, 0)
            eta = remaining / effective_rate if effective_rate > 0 else math.inf
            detail = (
                f"{effective_rate * 60:.1f} trials/min | "
                f"ETA {_format_duration(eta)}"
            )
        if self._rich and self._overall is not None:
            self._progress.update(
                self._overall,
                completed=new_completed,
                total=new_total,
                detail=detail,
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
            self.console.print(
                f"PROGRESS {self._stage} | {new_completed}/{new_total}"
                f"{worker_suffix} | {detail}"
            )
            self._last_compact_update = now

    def update_leaderboard(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Render a shared text-private TOP list below the active progress."""

        self._require_active()
        normalized: list[dict[str, object]] = []
        for source in rows[:6]:
            row = {
                "rank": int(source["rank"]),
                "trial": int(source["trial"]),
                "feasible": bool(source["feasible"]),
                "cost_up": float(source["cost_up"]),
                "removal": float(source["removal"]),
                "preservation_loss": float(source["preservation_loss"]),
                "ppl_drift": float(source["ppl_drift"]),
                "gate": _clean_label(source["gate"], fallback="unknown", max_length=52),
            }
            numeric = (
                row["cost_up"],
                row["removal"],
                row["preservation_loss"],
                row["ppl_drift"],
            )
            if any(not math.isfinite(float(value)) for value in numeric):
                raise ValueError("leaderboard metrics must be finite")
            normalized.append(row)

        if not self._rich:
            return
        table = Table(
            title=f"Current TOP-{len(normalized)}",
            show_edge=False,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column("#", justify="right")
        table.add_column("Trial", justify="right")
        table.add_column("Cost↑", justify="right")
        table.add_column("Removal↑", justify="right")
        table.add_column("Preserve↓", justify="right")
        table.add_column("PPL↓", justify="right")
        table.add_column("Gate")
        for row in normalized:
            feasible = bool(row["feasible"])
            table.add_row(
                str(row["rank"]),
                f"T{row['trial']}",
                f"{float(row['cost_up']):.3f}",
                f"{float(row['removal']):+.5f}",
                f"{float(row['preservation_loss']):.5f}",
                f"{float(row['ppl_drift']) * 100:.2f}%",
                "PASS" if feasible else str(row["gate"]),
                style=None if feasible else "yellow",
            )
        self._progress.set_footer(table)

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
            preserve_progress = bool(self._workers) and bool(
                self._progress.live.transient
            )
            if self._progress.live.is_started:
                self._progress.stop()
            if preserve_progress:
                self.console.print(self._progress.get_renderable())
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
