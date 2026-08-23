# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build a verified self-classification view of the main Heretic corpus."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .utils import get_file_sha256

_SOURCE_FILES = (
    ("direction_{language}_safe.jsonl", "safe", "canonical_id"),
    ("direction_{language}_unsafe.jsonl", "unsafe", "canonical_id"),
    ("search_unsafe_{language}.jsonl", "unsafe", "base_id"),
    ("srg_calibration_{language}.jsonl", "unsafe", "base_id"),
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path.name}:{line_number}")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError(f"non-object JSONL row at {path.name}:{line_number}")
            rows.append(raw)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _normalized_rows(
    source_root: Path,
    language: str,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {"safe": [], "unsafe": []}
    for template, direction, id_field in _SOURCE_FILES:
        path = source_root / template.format(language=language)
        if not path.is_file():
            raise FileNotFoundError(path)
        for raw in _read_jsonl(path):
            if str(raw.get("language", "")).lower() != language:
                raise ValueError(f"source language drift: {path.name}")
            canonical_id = str(raw[id_field])
            category = str(raw.get("category_id", "uncategorized"))
            category_values = raw.get("category_ids")
            if not isinstance(category_values, list) or not category_values:
                category_values = [category]
            row: dict[str, object] = {
                "canonical_id": canonical_id,
                "row_id": str(raw["row_id"]),
                "language": language,
                "direction_class": direction,
                "category_id": category,
                "category_ids": [str(value) for value in category_values],
                "prompt": str(raw["prompt"]),
            }
            result[direction].append(row)
    return result


def materialize_main_classification_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    languages: Sequence[str],
) -> dict[str, Any]:
    """Combine direction/search/SRG inputs without changing their aligned IDs."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    normalized_languages = tuple(str(value).strip().lower() for value in languages)
    if not normalized_languages or len(set(normalized_languages)) != len(
        normalized_languages
    ):
        raise ValueError("languages must be non-empty and unique")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        files: list[dict[str, object]] = []
        canonical_order: dict[str, tuple[str, ...]] = {}
        direction_counts: dict[str, int] = {}
        for language in normalized_languages:
            rows_by_direction = _normalized_rows(source_root, language)
            for direction in ("safe", "unsafe"):
                rows = rows_by_direction[direction]
                ordered = tuple(str(row["canonical_id"]) for row in rows)
                previous = canonical_order.setdefault(direction, ordered)
                if previous != ordered:
                    raise ValueError(
                        f"source canonical order drift: {language}/{direction}"
                    )
                direction_counts.setdefault(direction, len(rows))
                if direction_counts[direction] != len(rows):
                    raise ValueError(f"source row count drift: {language}/{direction}")
                name = f"direction_{language}_{direction}_strict.jsonl"
                path = temporary / name
                _write_jsonl(path, rows)
                files.append(
                    {
                        "language": language,
                        "direction": direction,
                        "path": name,
                        "rows": len(rows),
                        "sha256": get_file_sha256(path),
                    }
                )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "source_root": str(source_root),
            "source_files": {
                path.name: get_file_sha256(path)
                for language in normalized_languages
                for template, _, _ in _SOURCE_FILES
                if (path := source_root / template.format(language=language)).is_file()
            },
            "canonical_rows": sum(direction_counts.values()),
            "directions": direction_counts,
            "languages": list(normalized_languages),
            "rows": sum(direction_counts.values()) * len(normalized_languages),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.replace(output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def apply_aligned_translations(
    source_root: str | Path,
    translations_path: str | Path,
    output_root: str | Path,
    *,
    target_language: str,
) -> dict[str, object]:
    """Apply a complete translation sidecar while preserving canonical coverage."""

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    language = target_language.strip().lower()
    if not language or language == "en":
        raise ValueError("target language must be non-English")

    source_rows = _normalized_rows(source_root, "en")
    expected = {
        (direction, str(row["canonical_id"]))
        for direction, rows in source_rows.items()
        for row in rows
    }
    translations: dict[tuple[str, str], str] = {}
    for raw in _read_jsonl(Path(translations_path)):
        key = (str(raw["direction_class"]), str(raw["canonical_id"]))
        prompt = str(raw.get("prompt", "")).strip()
        if key in translations or not prompt:
            raise ValueError("translation coverage is duplicate or empty")
        translations[key] = prompt
    if set(translations) != expected:
        raise ValueError("translation coverage does not match the canonical corpus")

    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"safe": 0, "unsafe": 0}
    for template, direction, id_field in _SOURCE_FILES:
        source_path = source_root / template.format(language="en")
        output_path = output_root / template.format(language=language)
        translated_rows: list[dict[str, object]] = []
        for raw in _read_jsonl(source_path):
            canonical_id = str(raw[id_field])
            translated = dict(raw)
            translated["row_id"] = f"{language.upper()}-{canonical_id}"
            translated["language"] = language
            translated["prompt"] = translations[(direction, canonical_id)]
            translated["canonical_language"] = "en"
            translated["translation_status"] = "aligned"
            translated_rows.append(translated)
            counts[direction] += 1
        _write_jsonl(output_path, translated_rows)
    return {"safe": counts["safe"], "unsafe": counts["unsafe"], "language": language}
