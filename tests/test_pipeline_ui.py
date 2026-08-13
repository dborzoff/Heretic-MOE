from __future__ import annotations

import io

import pytest
from rich.console import Console

from heretic.pipeline_ui import PipelineUI


def _console(*, terminal: bool) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return (
        Console(
            file=stream,
            force_terminal=terminal,
            width=100,
            record=True,
            color_system=None,
        ),
        stream,
    )


def test_tty_stage_workers_and_summary_render_rich() -> None:
    console, _ = _console(terminal=True)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Search", total=4, workers=("gpu-0", "gpu-1"))
    ui.update_worker("gpu-0", completed=2, total=4, rate=1.5)
    ui.update_worker("gpu-1", completed=4, total=4, rate=2.5)
    ui.finish_stage(
        {
            "status": "PASS",
            "rows": 8,
            "prompt_payload": "secret",
            "notes": "ok",
        }
    )
    ui.close()

    output = console.export_text(styles=False)
    assert "Search" in output
    assert "gpu-0" in output
    assert "gpu-1" in output
    assert "4/4" in output
    assert "PASS" in output
    assert "rows" in output
    assert "8" in output
    assert "notes" in output
    assert "ok" in output
    assert "Metric" not in output
    assert "Value" not in output
    assert "secret" not in output
    assert "prompt_payload" not in output


def test_tty_stage_summary_is_one_compact_result_line() -> None:
    console, _ = _console(terminal=True)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Direction map", total=10)
    ui.finish_stage(
        {
            "status": "PASS",
            "workers": 2,
            "rows": 10_000,
            "next": "Clean reference",
        }
    )
    ui.close()

    output = console.export_text(styles=False)
    assert "PASS Direction map" in output
    assert "workers 2" in output
    assert "rows 10000" in output
    assert "→ Clean reference" in output
    assert "Metric" not in output


def test_stage_can_print_compact_findings_after_summary() -> None:
    console, _ = _console(terminal=True)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Direction analysis", total=3)
    ui.finish_stage(
        {"status": "PASS", "rows": 10_000, "next": "Clean reference"}
    )
    ui.result(
        "Found",
        {
            "layers": "6-36",
            "strongest": "29,30,23",
            "cross-lang": "94.9%",
        },
    )
    ui.result("Report", {"HTML": "analysis/report.html"})
    ui.close()

    lines = [line.strip() for line in console.export_text(styles=False).splitlines()]
    assert any(line.startswith("✓ PASS Direction analysis") for line in lines)
    assert "Found: layers 6-36 | strongest 29,30,23 | cross-lang 94.9%" in lines
    assert "Report: HTML analysis/report.html" in lines


def test_non_tty_fallback_is_compact_and_text_private() -> None:
    console, stream = _console(terminal=False)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Prepare", total=2, workers=("gpu-0",))
    ui.update_worker("gpu-0", completed=1, total=2, rate=2.5)
    ui.finish_stage(
        {
            "status": "FAIL",
            "error": "boom",
            "response": "should-not-print",
        }
    )
    ui.close()

    output = stream.getvalue()
    assert "Prepare" in output
    assert "gpu-0 1/2" in output
    assert "2.5 rows/s" in output
    assert "FAIL" in output
    assert "error=boom" in output
    assert "should-not-print" not in output
    assert "response" not in output
    assert (chr(27) + "[") not in output


def test_stage_lifecycle_and_unknown_worker() -> None:
    console, _ = _console(terminal=False)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Search", total=1, workers=("gpu-0",))
    with pytest.raises(RuntimeError, match="active stage"):
        ui.stage("Again", total=1)
    with pytest.raises(KeyError, match="unknown worker"):
        ui.update_worker("gpu-9", completed=1, total=1)
    ui.finish_stage({"status": "PASS"})
    ui.close()
    ui.close()
    with pytest.raises(RuntimeError, match="closed"):
        ui.stage("Closed", total=1)


def test_overall_total_stays_global_for_multiple_worker_rows() -> None:
    console, _ = _console(terminal=True)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)

    ui.stage("Geometry", total=10, workers=("gpu-0", "gpu-1"))
    ui.update_worker("gpu-0", completed=3, total=10, rate=1.0)
    ui.update_worker("gpu-1", completed=2, total=10, rate=1.0)
    ui.update_overall(completed=7, total=10, rate=2.0)

    assert ui._progress.tasks[0].total == 10
    assert ui._progress.tasks[0].completed == 7
    ui.finish_stage({"status": "PASS"})
    ui.close()


def test_fail_stage_is_safe_and_idempotent() -> None:
    console, stream = _console(terminal=False)
    ui = PipelineUI(console=console, non_tty_update_interval=0.0)
    ui.stage("Search", total=2)

    result = ui.fail_stage(RuntimeError("private detail"))

    assert result is not None and result["status"] == "FAIL"
    assert ui.fail_stage(RuntimeError("again")) is None
    assert "private detail" not in stream.getvalue()
    ui.close()
