# SPDX-License-Identifier: AGPL-3.0-or-later

"""Frozen, text-safe contract for multilingual Heretic-MOE search inputs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .language_map_data import GeometryRow, LanguageFile, load_aligned_corpus


@dataclass(frozen=True)
class CalibrationRow:
    base_id: str
    row_id: str
    language: str
    category_id: str
    prompt: str
    source_path: Path
    source_line: int


@dataclass(frozen=True)
class MultilingualDatasetBundle:
    direction_rows: tuple[GeometryRow, ...]
    trial_rows: tuple[GeometryRow, ...]
    search_rows: tuple[CalibrationRow, ...]
    final_rows: tuple[CalibrationRow, ...]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _assert_text_free_mapping(value: object, path: str = "contract") -> None:
    forbidden = {"prompt", "response", "answer", "text"}
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in forbidden:
                raise ValueError(f"{path} contains forbidden text field {key!r}")
            _assert_text_free_mapping(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_text_free_mapping(child, f"{path}[{index}]")


def build_frozen_run_contract(
    *,
    dataset_contract_sha256: str,
    model_id: str,
    model_fingerprint_sha256: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    map_sha256: str,
    srg_profile_sha256: str,
    schedule_seed: int,
    schedule_version: int,
    generation_contract: dict[str, object],
    metric_contract: dict[str, object],
    constraint_contract: dict[str, object],
) -> dict[str, Any]:
    """Build the immutable resume contract for one multilingual study."""

    if not model_id.strip():
        raise ValueError("model_id cannot be blank")
    if schedule_version <= 0:
        raise ValueError("schedule_version must be positive")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_contract_sha256": _validate_sha256(
            dataset_contract_sha256, "dataset_contract_sha256"
        ),
        "model_id": model_id,
        "model_fingerprint_sha256": _validate_sha256(
            model_fingerprint_sha256, "model_fingerprint_sha256"
        ),
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "map_sha256": _validate_sha256(map_sha256, "map_sha256"),
        "srg_profile_sha256": _validate_sha256(
            srg_profile_sha256, "srg_profile_sha256"
        ),
        "schedule_seed": int(schedule_seed),
        "schedule_version": int(schedule_version),
        "generation_contract": generation_contract,
        "metric_contract": metric_contract,
        "constraint_contract": constraint_contract,
    }
    _assert_text_free_mapping(payload)
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def write_or_verify_frozen_contract(
    path: str | Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Write a new contract atomically or verify an exact resume match."""

    destination = Path(path)
    _assert_text_free_mapping(contract)
    expected_hash = contract.get("contract_sha256")
    if not isinstance(expected_hash, str):
        raise TypeError("contract mismatch: missing contract_sha256")
    without_hash = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if _canonical_sha256(without_hash) != expected_hash:
        raise RuntimeError("contract mismatch: invalid contract_sha256")
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError(f"contract mismatch for resume: {destination}")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return contract


def _required_string(
    value: dict[str, object], key: str, path: Path, line_number: int
) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field.strip():
        raise ValueError(f"{path.name}:{line_number} has invalid {key}")
    return field


def _read_calibration_file(
    path: Path,
    *,
    language: str,
    expected_rows: int,
) -> list[CalibrationRow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[CalibrationRow] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path.name}:{line_number} must be a JSON object")
            observed_language = _required_string(
                value, "language", path, line_number
            ).lower()
            if observed_language != language:
                raise ValueError(f"{path.name}:{line_number} language metadata drift")
            base_id = _required_string(value, "base_id", path, line_number)
            if base_id in seen_ids:
                raise ValueError(f"{path.name}:{line_number} duplicate base_id")
            seen_ids.add(base_id)
            rows.append(
                CalibrationRow(
                    base_id=base_id,
                    row_id=_required_string(value, "row_id", path, line_number),
                    language=observed_language,
                    category_id=_required_string(
                        value, "category_id", path, line_number
                    ),
                    prompt=_required_string(value, "prompt", path, line_number),
                    source_path=path.resolve(),
                    source_line=line_number,
                )
            )
    if len(rows) != expected_rows:
        raise ValueError(f"{path.name} expected {expected_rows} rows, got {len(rows)}")
    return rows


def _load_calibration_pool(
    root: Path,
    *,
    filename_prefix: str,
    languages: tuple[str, ...],
    expected_rows: int,
) -> list[CalibrationRow]:
    by_language = {
        language: _read_calibration_file(
            root / f"{filename_prefix}_{language}.jsonl",
            language=language,
            expected_rows=expected_rows,
        )
        for language in languages
    }
    reference = by_language[languages[0]]
    reference_ids = [row.base_id for row in reference]
    reference_categories = {row.base_id: row.category_id for row in reference}
    output: list[CalibrationRow] = []
    seen_row_ids: set[str] = set()
    for language in languages:
        rows = by_language[language]
        if [row.base_id for row in rows] != reference_ids:
            raise ValueError(f"{filename_prefix}/{language} coverage or order drift")
        for row in rows:
            if reference_categories[row.base_id] != row.category_id:
                raise ValueError(f"{filename_prefix}/{row.base_id} category drift")
            if row.row_id in seen_row_ids:
                raise ValueError(f"{filename_prefix} duplicate row_id")
            seen_row_ids.add(row.row_id)
        output.extend(rows)
    return output


def _normalized_prompt_hash(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assert_pool_prompt_disjoint(
    pools: dict[str, list[GeometryRow] | list[CalibrationRow]],
) -> None:
    hashed = {
        name: {_normalized_prompt_hash(row.prompt) for row in rows}
        for name, rows in pools.items()
    }
    names = tuple(hashed)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = hashed[left] & hashed[right]
            if overlap:
                raise ValueError(
                    f"prompt overlap between {left} and {right}: {len(overlap)}"
                )


def _file_record(
    path: Path,
    *,
    relative_to: Path,
    rows: int,
    pool: str,
    language: str | None = None,
    direction: str | None = None,
    ids: list[str] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "rows": rows,
        "sha256": _sha256(path),
        "pool": pool,
    }
    if language is not None:
        record["language"] = language
    if direction is not None:
        record["direction_class"] = direction
    if ids is not None:
        record["id_order_sha256"] = _canonical_sha256(ids)
    return record


def load_multilingual_dataset_bundle(
    *,
    dataset_root: str | Path,
    split_root: str | Path | None = None,
    languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
    direction_rows_per_cell: int = 1000,
    trial_rows_per_cell: int = 400,
    calibration_rows_per_language: int = 132,
) -> MultilingualDatasetBundle:
    """Load all four frozen pools and return a prompt-free public manifest."""

    root = Path(dataset_root).resolve()
    split = (
        Path(split_root).resolve()
        if split_root is not None
        else root / "operative_split_1000_400_v1"
    )
    normalized_languages = tuple(language.lower() for language in languages)
    if not normalized_languages or len(set(normalized_languages)) != len(
        normalized_languages
    ):
        raise ValueError("languages must be a non-empty unique sequence")
    if (
        min(
            direction_rows_per_cell,
            trial_rows_per_cell,
            calibration_rows_per_language,
        )
        <= 0
    ):
        raise ValueError("all expected row counts must be positive")

    direction_files = [
        LanguageFile(
            language,
            direction,  # type: ignore[arg-type]
            split / f"direction_{language}_{direction}_{direction_rows_per_cell}.jsonl",
        )
        for direction in ("safe", "unsafe")
        for language in normalized_languages
    ]
    trial_files = [
        LanguageFile(
            language,
            direction,  # type: ignore[arg-type]
            split / f"trial_{language}_{direction}_{trial_rows_per_cell}.jsonl",
        )
        for direction in ("safe", "unsafe")
        for language in normalized_languages
    ]
    direction_rows = load_aligned_corpus(
        direction_files,
        normalized_languages,
        direction_rows_per_cell,
    )
    trial_rows = load_aligned_corpus(
        trial_files,
        normalized_languages,
        trial_rows_per_cell,
    )
    search_rows = _load_calibration_pool(
        root,
        filename_prefix="search_unsafe",
        languages=normalized_languages,
        expected_rows=calibration_rows_per_language,
    )
    final_rows = _load_calibration_pool(
        root,
        filename_prefix="srg_calibration",
        languages=normalized_languages,
        expected_rows=calibration_rows_per_language,
    )
    _assert_pool_prompt_disjoint(
        {
            "direction": direction_rows,
            "trial": trial_rows,
            "srg_calibration": search_rows,
            "final_holdout": final_rows,
        }
    )

    file_records: dict[str, dict[str, object]] = {}
    for pool, specifications, expected in (
        ("direction", direction_files, direction_rows_per_cell),
        ("trial", trial_files, trial_rows_per_cell),
    ):
        for specification in specifications:
            path = Path(specification.path)
            ids = [
                row.canonical_id
                for row in (direction_rows if pool == "direction" else trial_rows)
                if row.language == specification.language
                and row.direction == specification.direction
            ]
            file_records[path.name] = _file_record(
                path,
                relative_to=root,
                rows=expected,
                pool=pool,
                language=specification.language,
                direction=specification.direction,
                ids=ids,
            )
    for pool, prefix, rows in (
        ("srg_calibration", "search_unsafe", search_rows),
        ("final_holdout", "srg_calibration", final_rows),
    ):
        for language in normalized_languages:
            path = root / f"{prefix}_{language}.jsonl"
            ids = [row.base_id for row in rows if row.language == language]
            file_records[path.name] = _file_record(
                path,
                relative_to=root,
                rows=calibration_rows_per_language,
                pool=pool,
                language=language,
                ids=ids,
            )

    dataset_manifest = root / "manifest.json"
    split_manifest = split / "manifest.json"
    source_manifests = {
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "languages": list(normalized_languages),
        "counts": {
            "direction": len(direction_rows),
            "trial": len(trial_rows),
            "srg_calibration": len(search_rows),
            "final_holdout": len(final_rows),
        },
        "rows_per_cell": {
            "direction": direction_rows_per_cell,
            "trial": trial_rows_per_cell,
            "srg_calibration": calibration_rows_per_language,
            "final_holdout": calibration_rows_per_language,
        },
        "source_manifests": source_manifests,
        "files": dict(sorted(file_records.items())),
        "cross_pool_exact_overlap": 0,
    }
    manifest["contract_sha256"] = _canonical_sha256(manifest)
    return MultilingualDatasetBundle(
        direction_rows=tuple(direction_rows),
        trial_rows=tuple(trial_rows),
        search_rows=tuple(search_rows),
        final_rows=tuple(final_rows),
        manifest=manifest,
    )
