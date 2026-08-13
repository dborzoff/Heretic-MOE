from __future__ import annotations

import pytest

from heretic import cli


def test_public_launcher_reports_version_without_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["hereticMOE", "--version"])

    cli.main()

    assert "Heretic-MOE 1.5.0" in capsys.readouterr().out
