from __future__ import annotations

import sys
from pathlib import Path

from heretic import cli
from heretic.multilingual_prepare_cli import build_clean_reference_worker_specs


def test_cli_dispatches_multilingual_preparation(monkeypatch) -> None:
    observed: list[list[str]] = []

    def fake_main(arguments):
        observed.append(list(arguments))

    monkeypatch.setattr(
        "heretic.multilingual_prepare_cli.main",
        fake_main,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hereticMOE", "prepare-multilingual", "--config", "config.toml"],
    )

    cli.main()

    assert observed == [["--config", "config.toml"]]


def test_cli_dispatches_final_holdout_preparation(monkeypatch) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(
        "heretic.multilingual_final_holdout_cli.main",
        lambda arguments: observed.append(list(arguments)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["hereticMOE", "prepare-final-holdout", "--config", "final.toml"],
    )

    cli.main()

    assert observed == [["--config", "final.toml"]]


def test_clean_reference_launches_all_requested_gpu_workers_once() -> None:
    specs, boundaries = build_clean_reference_worker_specs(
        ("0", "2", "5", "7"),
        total_rows=40,
        job_path=Path("F:/run/job.json"),
        base_environment={},
        cpu_count=32,
    )

    assert [spec.worker_id for spec in specs] == ["gpu-0", "gpu-2", "gpu-5", "gpu-7"]
    assert boundaries == [0, 10, 20, 30, 40]
    assert [spec.command[-4:] for spec in specs] == [
        ("--start", "0", "--end", "10"),
        ("--start", "10", "--end", "20"),
        ("--start", "20", "--end", "30"),
        ("--start", "30", "--end", "40"),
    ]
