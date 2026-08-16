# SPDX-License-Identifier: AGPL-3.0-or-later

"""Commands that connect the frozen multilingual map to live trial geometry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def geometry_project_command(
    *,
    executable: Path,
    split_root: Path,
    runtime_sources: Path,
    journal: Path,
    package_dir: Path,
    languages: Sequence[str],
    rows_per_cell: int,
    seed: int,
) -> list[str]:
    command = [
        str(executable),
        "geometry-map",
        "project",
        "--cache-dir",
        str((runtime_sources / "cache").resolve()),
        "--journal",
        str(journal.resolve()),
        "--output-dir",
        str(package_dir.resolve()),
        "--languages",
        ",".join(languages),
        "--rows-per-cell",
        str(rows_per_cell),
        "--seed",
        str(seed),
    ]
    for language in languages:
        command.extend(
            (
                "--group-a",
                f"{language}={split_root / f'map_{language}_safe_{rows_per_cell}.jsonl'}",
            )
        )
    for language in languages:
        command.extend(
            (
                "--group-b",
                f"{language}={split_root / f'map_{language}_unsafe_{rows_per_cell}.jsonl'}",
            )
        )
    return command


def geometry_render_command(*, executable: Path, package_dir: Path) -> list[str]:
    return [
        str(executable),
        "geometry-map",
        "render",
        "--package-dir",
        str(package_dir.resolve()),
    ]
