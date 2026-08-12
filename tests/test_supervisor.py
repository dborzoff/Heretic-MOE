from __future__ import annotations

import pytest

from heretic import supervisor


def _parse(*extra: str):
    return supervisor.parse_args(
        [
            "--model",
            "example/model",
            "--run-root",
            "run",
            *extra,
        ]
    )


def test_supervisor_defaults_to_export_after_recheck() -> None:
    args = _parse()

    assert args.post_search_mode == "export"


def test_supervisor_exposes_recheck_without_export() -> None:
    args = _parse("--recheck-only")

    assert args.post_search_mode == "recheck"


def test_supervisor_post_search_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse("--recheck-only", "--no-finalize")
