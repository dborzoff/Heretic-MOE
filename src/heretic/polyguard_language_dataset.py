# SPDX-License-Identifier: AGPL-3.0-or-later

"""Materialize a strict, aligned PolyGuard geometry corpus without responses."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd

Direction = Literal["safe", "unsafe"]

POLYGUARD_LANGUAGE_CODES: dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Italian": "it",
    "German": "de",
    "Portuguese": "pt",
    "Thai": "th",
    "Spanish": "es",
    "Czech": "cs",
    "Swedish": "sv",
    "Chinese": "zh",
    "Arabic": "ar",
    "Dutch": "nl",
    "Korean": "ko",
    "Polish": "pl",
    "Russian": "ru",
    "Japanese": "ja",
}

PRODUCTION_COUNTS: dict[Direction, int] = {"safe": 482, "unsafe": 238}

_SOURCE_COLUMNS = (
    "id",
    "language",
    "prompt",
    "prompt_harm_label",
    "response_refusal_label",
    "prompt_harm_agreement",
    "response_refusal_agreement",
    "prompt_categories",
    "subcategory",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_id(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"PG-{value}"


def _category_ids(value: object) -> tuple[str, ...]:
    raw = "" if pd.isna(value) else str(value)
    categories = {part for part in raw.split() if part.startswith("S")}
    if not categories:
        return ("benign",)
    return tuple(
        sorted(
            categories,
            key=lambda item: (
                int(item[1:]) if item[1:].isdigit() else 10_000,
                item,
            ),
        )
    )


def _direction(row: pd.Series) -> Direction | None:
    unanimous = (
        float(row["prompt_harm_agreement"]) == 3.0
        and float(row["response_refusal_agreement"]) == 3.0
    )
    if not unanimous:
        return None
    if (
        row["prompt_harm_label"] == "unharmful"
        and row["response_refusal_label"] == "compliance"
    ):
        return "safe"
    if (
        row["prompt_harm_label"] == "harmful"
        and row["response_refusal_label"] == "refusal"
    ):
        return "unsafe"
    return None


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(temporary, path)


def materialize_polyguard_language_dataset(
    source: Path,
    output_dir: Path,
    *,
    language_codes: Mapping[str, str] = POLYGUARD_LANGUAGE_CODES,
    expected_counts: Mapping[str, int] = PRODUCTION_COUNTS,
) -> dict[str, object]:
    """Create strict per-language SAFE/UNSAFE JSONL files and a public manifest."""

    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    language_codes = dict(language_codes)
    if not language_codes or len(set(language_codes.values())) != len(language_codes):
        raise ValueError("language codes must be non-empty and unique")

    frame = pd.read_parquet(source, columns=list(_SOURCE_COLUMNS))
    observed_languages = set(frame["language"].astype(str))
    if observed_languages != set(language_codes):
        raise ValueError("source language coverage does not match the frozen contract")

    source_ids: list[object] | None = None
    by_language: dict[str, pd.DataFrame] = {}
    for source_language in language_codes:
        language_frame = frame[frame["language"] == source_language].copy()
        if language_frame["id"].duplicated().any():
            raise ValueError("duplicate canonical id in source language")
        ids = language_frame["id"].tolist()
        if source_ids is None:
            source_ids = ids
        elif ids != source_ids:
            raise ValueError("source is not exactly aligned across languages")
        by_language[source_language] = language_frame.set_index("id", drop=False)
    assert source_ids is not None

    selected: dict[Direction, list[object]] = {"safe": [], "unsafe": []}
    reference_language = next(iter(language_codes))
    for source_id in source_ids:
        reference = by_language[reference_language].loc[source_id]
        direction = _direction(reference)
        if direction is None:
            continue
        reference_categories = _category_ids(reference["prompt_categories"])
        reference_subcategory = str(reference["subcategory"])
        for source_language in language_codes:
            translated = by_language[source_language].loc[source_id]
            if (
                _direction(translated) != direction
                or _category_ids(translated["prompt_categories"])
                != reference_categories
                or str(translated["subcategory"]) != reference_subcategory
            ):
                raise ValueError("aligned source metadata drift")
        selected[direction].append(source_id)

    normalized_expected = {
        direction: int(expected_counts[direction]) for direction in ("safe", "unsafe")
    }
    actual_counts = {direction: len(selected[direction]) for direction in selected}
    if actual_counts != normalized_expected:
        raise ValueError(
            f"strict selection count mismatch: expected={normalized_expected}, "
            f"actual={actual_counts}"
        )

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    files: list[dict[str, object]] = []
    try:
        for source_language, language_code in language_codes.items():
            language_frame = by_language[source_language]
            for direction in ("safe", "unsafe"):
                rows: list[dict[str, object]] = []
                for source_id in selected[direction]:
                    source_row = language_frame.loc[source_id]
                    prompt = str(source_row["prompt"])
                    if not prompt.strip():
                        raise ValueError("source contains blank prompt")
                    categories = _category_ids(source_row["prompt_categories"])
                    canonical_id = _canonical_id(source_id)
                    rows.append(
                        {
                            "canonical_id": canonical_id,
                            "row_id": f"{language_code.upper()}-{canonical_id}",
                            "language": language_code,
                            "direction_class": direction,
                            "category_id": categories[0],
                            "category_ids": list(categories),
                            "source_category": str(source_row["subcategory"]),
                            "prompt": prompt,
                        }
                    )
                name = f"direction_{language_code}_{direction}_strict.jsonl"
                path = temporary / name
                _write_jsonl(path, rows)
                files.append(
                    {
                        "language": language_code,
                        "direction": direction,
                        "path": name,
                        "rows": len(rows),
                        "sha256": _sha256(path),
                    }
                )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "languages": list(language_codes.values()),
            "directions": actual_counts,
            "canonical_rows": sum(actual_counts.values()),
            "rows": sum(actual_counts.values()) * len(language_codes),
            "files": files,
            "selection_contract": {
                "safe": "unharmful+compliance+agreement3",
                "unsafe": "harmful+refusal+agreement3",
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
