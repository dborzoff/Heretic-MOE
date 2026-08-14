# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verified input and text-free checkpoints for self-classification."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from .self_classification import ClassificationInput, ClassificationResult
from .utils import get_file_sha256

ResultKey = tuple[str, str, str]
_PROHIBITED_RESULT_FIELDS = frozenset({"prompt", "response", "raw_output", "text"})


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path.name}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path.name}:{line_number}")
            rows.append(value)
    return rows


def load_classification_rows(
    manifest_path: str | Path,
    languages: Sequence[str],
) -> list[ClassificationInput]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise ValueError("dataset manifest is not a verified schema-v1 artifact")
    selected = tuple(str(value).strip().lower() for value in languages)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected languages must be non-empty and unique")
    available = tuple(str(value).lower() for value in manifest.get("languages", []))
    missing_languages = sorted(set(selected) - set(available))
    if missing_languages:
        raise ValueError(f"dataset manifest lacks languages: {missing_languages}")

    expected_counts = {
        direction: int(manifest["directions"][direction])
        for direction in ("safe", "unsafe")
    }
    entries = [
        entry
        for entry in manifest.get("files", [])
        if str(entry.get("language", "")).lower() in selected
    ]
    entry_keys = [
        (str(entry["language"]).lower(), str(entry["direction"]).lower())
        for entry in entries
    ]
    expected_keys = {
        (language, direction)
        for language in selected
        for direction in ("safe", "unsafe")
    }
    if set(entry_keys) != expected_keys or len(entry_keys) != len(expected_keys):
        raise ValueError("dataset manifest language/direction file coverage mismatch")

    by_key = {key: entry for key, entry in zip(entry_keys, entries, strict=True)}
    result: list[ClassificationInput] = []
    canonical_order: dict[str, tuple[str, ...]] = {}
    for language in selected:
        for direction in ("safe", "unsafe"):
            entry = by_key[(language, direction)]
            path = (manifest_path.parent / str(entry["path"])).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError("dataset manifest file escapes its root")
            if get_file_sha256(path) != str(entry["sha256"]):
                raise ValueError(f"dataset file hash drift: {path.name}")
            raw_rows = _read_jsonl(path)
            expected = expected_counts[direction]
            if int(entry["rows"]) != expected or len(raw_rows) != expected:
                raise ValueError(f"dataset row count drift: {path.name}")
            ids: list[str] = []
            for raw in raw_rows:
                if str(raw.get("language", "")).lower() != language:
                    raise ValueError(f"dataset row language drift: {path.name}")
                if str(raw.get("direction_class", "")).lower() != direction:
                    raise ValueError(f"dataset row direction drift: {path.name}")
                category_values = raw.get("category_ids")
                if not isinstance(category_values, list) or not category_values:
                    category_values = [raw.get("category_id")]
                row = ClassificationInput(
                    canonical_id=str(raw["canonical_id"]),
                    row_id=str(raw["row_id"]),
                    language=language,
                    category_ids=tuple(str(value) for value in category_values),
                    direction_class=direction,
                    prompt=str(raw["prompt"]),
                )
                ids.append(row.canonical_id)
                result.append(row)
            ordered = tuple(ids)
            previous = canonical_order.setdefault(direction, ordered)
            if ordered != previous:
                raise ValueError(f"dataset canonical order drift: {language}/{direction}")
    if len({row.row_id for row in result}) != len(result):
        raise ValueError("dataset contains duplicate row IDs")
    return result


def append_result_atomic(path: str | Path, result: ClassificationResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_public_dict()
    if _PROHIBITED_RESULT_FIELDS & payload.keys():
        raise ValueError("public result contains prohibited text fields")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _result_key(raw: dict[str, object]) -> ResultKey:
    prohibited = _PROHIBITED_RESULT_FIELDS & raw.keys()
    if prohibited:
        raise ValueError(f"result contains prohibited fields: {sorted(prohibited)}")
    return str(raw["model_id"]), str(raw["row_id"]), str(raw["variant"])


def load_completed_keys(path: str | Path) -> set[ResultKey]:
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[ResultKey] = set()
    for raw in _read_jsonl(path):
        key = _result_key(raw)
        if key in keys:
            raise ValueError(f"duplicate result key: {key}")
        keys.add(key)
    return keys


def verify_result_coverage(
    path: str | Path,
    expected_keys: Iterable[ResultKey],
) -> dict[str, int]:
    expected = set(expected_keys)
    actual = load_completed_keys(path)
    missing = expected - actual
    extra = actual - expected
    summary = {
        "expected": len(expected),
        "actual": len(actual),
        "missing": len(missing),
        "extra": len(extra),
    }
    if missing or extra:
        raise ValueError(f"result coverage mismatch: {summary}")
    return summary
