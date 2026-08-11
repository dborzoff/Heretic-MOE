# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strict, text-safe input contract for multilingual geometry diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Direction = Literal["safe", "unsafe"]
DIRECTIONS: tuple[Direction, ...] = ("safe", "unsafe")


@dataclass(frozen=True)
class LanguageFile:
    language: str
    direction: Direction
    path: Path


@dataclass(frozen=True)
class GeometryRow:
    canonical_id: str
    row_id: str
    language: str
    direction: Direction
    category_id: str
    prompt: str
    source_path: Path
    source_line: int


def _required_string(row: dict[str, object], key: str, path: Path, line: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{path.name}:{line} has invalid {key}")
    if not value.strip():
        description = "blank prompt" if key == "prompt" else f"blank {key}"
        raise ValueError(f"{path.name}:{line} has {description}")
    return value


def _read_file(specification: LanguageFile) -> list[GeometryRow]:
    path = Path(specification.path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[GeometryRow] = []
    seen_canonical: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must be a JSON object")
            language = _required_string(value, "language", path, line_number).lower()
            direction = _required_string(
                value, "direction_class", path, line_number
            ).lower()
            if language != specification.language:
                raise ValueError(
                    f"{path.name}:{line_number} language metadata drift"
                )
            if direction != specification.direction:
                raise ValueError(
                    f"{path.name}:{line_number} direction metadata drift"
                )
            canonical_id = _required_string(
                value, "canonical_id", path, line_number
            )
            if canonical_id in seen_canonical:
                raise ValueError(
                    f"{path.name}:{line_number} duplicate canonical_id"
                )
            seen_canonical.add(canonical_id)
            rows.append(
                GeometryRow(
                    canonical_id=canonical_id,
                    row_id=_required_string(value, "row_id", path, line_number),
                    language=language,
                    direction=direction,  # type: ignore[arg-type]
                    category_id=_required_string(
                        value, "category_id", path, line_number
                    ),
                    prompt=_required_string(value, "prompt", path, line_number),
                    source_path=path,
                    source_line=line_number,
                )
            )
    return rows


def load_aligned_corpus(
    files: list[LanguageFile],
    expected_languages: tuple[str, ...],
    expected_per_cell: int,
) -> list[GeometryRow]:
    """Load a balanced SAFE/UNSAFE corpus with exact translation alignment."""

    languages = tuple(language.lower() for language in expected_languages)
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("expected_languages must contain unique language codes")
    if expected_per_cell <= 0:
        raise ValueError("expected_per_cell must be positive")

    by_cell: dict[tuple[Direction, str], list[GeometryRow]] = {}
    for specification in files:
        normalized = LanguageFile(
            specification.language.lower(),
            specification.direction,
            Path(specification.path),
        )
        if normalized.language not in languages:
            raise ValueError(f"unexpected language file: {normalized.language}")
        if normalized.direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction file: {normalized.direction}")
        key = (normalized.direction, normalized.language)
        if key in by_cell:
            raise ValueError(f"duplicate input cell: {key}")
        cell_rows = _read_file(normalized)
        if len(cell_rows) != expected_per_cell:
            raise ValueError(
                f"{normalized.language}/{normalized.direction} expected "
                f"{expected_per_cell} rows, got {len(cell_rows)}"
            )
        by_cell[key] = cell_rows

    required_cells = {
        (direction, language) for direction in DIRECTIONS for language in languages
    }
    if set(by_cell) != required_cells:
        missing = sorted(required_cells - set(by_cell))
        extra = sorted(set(by_cell) - required_cells)
        raise ValueError(f"invalid input cells: missing={missing}, extra={extra}")

    output: list[GeometryRow] = []
    seen_row_ids: set[str] = set()
    for direction in DIRECTIONS:
        reference = by_cell[(direction, languages[0])]
        reference_ids = [row.canonical_id for row in reference]
        reference_categories = {
            row.canonical_id: row.category_id for row in reference
        }
        for language in languages:
            rows = by_cell[(direction, language)]
            canonical_ids = [row.canonical_id for row in rows]
            if canonical_ids != reference_ids:
                raise ValueError(
                    f"{direction}/{language} canonical coverage or order drift"
                )
            for row in rows:
                if reference_categories[row.canonical_id] != row.category_id:
                    raise ValueError(
                        f"{direction}/{row.canonical_id} category drift"
                    )
                if row.row_id in seen_row_ids:
                    raise ValueError("duplicate row_id")
                seen_row_ids.add(row.row_id)
            output.extend(rows)
    return output


def text_free_row_index(rows: list[GeometryRow]) -> list[dict[str, object]]:
    """Return cache metadata without prompt content."""

    return [
        {
            "index": index,
            "canonical_id": row.canonical_id,
            "row_id": row.row_id,
            "language": row.language,
            "direction_class": row.direction,
            "category_id": row.category_id,
            "source_file": row.source_path.name,
            "source_line": row.source_line,
        }
        for index, row in enumerate(rows)
    ]
