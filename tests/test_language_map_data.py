import json
from pathlib import Path

import pytest

from heretic.language_map_data import (
    LanguageFile,
    load_aligned_corpus,
    text_free_row_index,
)


def write_rows(
    path: Path,
    *,
    language: str,
    direction: str,
    categories: tuple[str, ...] = ("C01", "C02"),
) -> Path:
    prefix = "S" if direction == "safe" else "U"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for offset, category in enumerate(categories, start=1):
            canonical_id = f"{prefix}{offset:04d}"
            row = {
                "canonical_id": canonical_id,
                "row_id": f"{language.upper()}-{canonical_id}",
                "language": language,
                "direction_class": direction,
                "category_id": category,
                "prompt": f"private {language} {direction} {offset}",
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def balanced_files(tmp_path: Path) -> list[LanguageFile]:
    files = []
    for direction in ("safe", "unsafe"):
        for language in ("en", "ru"):
            path = write_rows(
                tmp_path / f"{language}_{direction}.jsonl",
                language=language,
                direction=direction,
            )
            files.append(LanguageFile(language, direction, path))
    return files


def test_loads_balanced_aligned_rows_without_leaking_prompt(tmp_path: Path):
    rows = load_aligned_corpus(
        balanced_files(tmp_path),
        expected_languages=("en", "ru"),
        expected_per_cell=2,
    )

    assert len(rows) == 8
    assert [row.row_id for row in rows] == [
        "EN-S0001",
        "EN-S0002",
        "RU-S0001",
        "RU-S0002",
        "EN-U0001",
        "EN-U0002",
        "RU-U0001",
        "RU-U0002",
    ]
    index = text_free_row_index(rows)
    assert all("prompt" not in row for row in index)
    assert index[0]["canonical_id"] == "S0001"


def test_rejects_missing_translation(tmp_path: Path):
    files = balanced_files(tmp_path)
    ru_unsafe = tmp_path / "ru_unsafe.jsonl"
    ru_unsafe.write_text(
        ru_unsafe.read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2 rows|canonical coverage"):
        load_aligned_corpus(
            files,
            expected_languages=("en", "ru"),
            expected_per_cell=2,
        )


def test_rejects_category_drift_across_translation(tmp_path: Path):
    files = balanced_files(tmp_path)
    ru_safe = tmp_path / "ru_safe.jsonl"
    rows = [json.loads(line) for line in ru_safe.read_text(encoding="utf-8").splitlines()]
    rows[0]["category_id"] = "DRIFT"
    ru_safe.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="category drift"):
        load_aligned_corpus(
            files,
            expected_languages=("en", "ru"),
            expected_per_cell=2,
        )


def test_rejects_blank_prompt_without_echoing_it(tmp_path: Path):
    files = balanced_files(tmp_path)
    en_safe = tmp_path / "en_safe.jsonl"
    rows = [json.loads(line) for line in en_safe.read_text(encoding="utf-8").splitlines()]
    rows[0]["prompt"] = "  "
    en_safe.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blank prompt") as error:
        load_aligned_corpus(
            files,
            expected_languages=("en", "ru"),
            expected_per_cell=2,
        )
    assert "private" not in str(error.value)
