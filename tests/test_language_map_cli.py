from __future__ import annotations

import json
from pathlib import Path

import pytest

from heretic import cli
from heretic import language_map_cli


def _write_cell(path: Path, *, language: str, direction: str) -> None:
    rows = []
    prefix = "A" if direction == "safe" else "B"
    for number in range(1, 3):
        canonical_id = f"{prefix}{number:04d}"
        rows.append(
            {
                "canonical_id": canonical_id,
                "row_id": f"{language.upper()}-{canonical_id}",
                "language": language,
                "direction_class": direction,
                "category_id": "C01",
                "prompt": f"synthetic-{language}-{canonical_id}",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_dispatches_geometry_map_without_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[list[str]] = []
    monkeypatch.setattr(language_map_cli, "main", lambda argv: received.append(argv))
    monkeypatch.setattr(cli.sys, "argv", ["hereticMOE", "geometry-map", "analyze"])

    cli.main()

    assert received == [["analyze"]]


def test_dry_run_validates_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[tuple[str, str], Path] = {}
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = tmp_path / f"{language}-{direction}.jsonl"
            _write_cell(path, language=language, direction=direction)
            files[(language, direction)] = path

    def fail_if_loaded(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not load a model")

    monkeypatch.setattr(language_map_cli, "_capture", fail_if_loaded)
    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--group-a",
            f"en={files[('en', 'safe')]}",
            "--group-a",
            f"ru={files[('ru', 'safe')]}",
            "--group-b",
            f"en={files[('en', 'unsafe')]}",
            "--group-b",
            f"ru={files[('ru', 'unsafe')]}",
        ]
    )

    assert result == {
        "status": "PASS",
        "mode": "dry-run",
        "languages": ["en", "ru"],
        "rows": 8,
        "rows_per_cell": 2,
        "source_rows_per_cell": 2,
    }


def test_dry_run_can_limit_each_validated_cell(tmp_path: Path) -> None:
    files: dict[tuple[str, str], Path] = {}
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            path = tmp_path / f"{language}-{direction}.jsonl"
            _write_cell(path, language=language, direction=direction)
            files[(language, direction)] = path

    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--limit-per-cell",
            "1",
            "--group-a",
            f"en={files[('en', 'safe')]}",
            "--group-a",
            f"ru={files[('ru', 'safe')]}",
            "--group-b",
            f"en={files[('en', 'unsafe')]}",
            "--group-b",
            f"ru={files[('ru', 'unsafe')]}",
        ]
    )

    assert result["rows"] == 4
    assert result["rows_per_cell"] == 1
    assert result["source_rows_per_cell"] == 2


def test_dry_run_discovers_frozen_layout_from_corpus_root(tmp_path: Path) -> None:
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            _write_cell(
                tmp_path / f"direction_{language}_{direction}_train.jsonl",
                language=language,
                direction=direction,
            )

    result = language_map_cli.main(
        [
            "run",
            "--dry-run",
            "--languages",
            "en,ru",
            "--rows-per-cell",
            "2",
            "--corpus-root",
            str(tmp_path),
        ]
    )

    assert result["status"] == "PASS"
    assert result["rows"] == 8
