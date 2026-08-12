from __future__ import annotations

import sys

from heretic import cli


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
