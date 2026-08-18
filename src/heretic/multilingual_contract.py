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
    direction: str = "unsafe"


@dataclass(frozen=True)
class MultilingualDatasetBundle:
    direction_rows: tuple[GeometryRow, ...]
    trial_rows: tuple[GeometryRow, ...]
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


def _load_directional_final_pool(
    root: Path,
    *,
    languages: tuple[str, ...],
    expected_rows: int,
) -> list[CalibrationRow]:
    files = [
        LanguageFile(
            language,
            direction,  # type: ignore[arg-type]
            root / f"final_{language}_{direction}_{expected_rows}.jsonl",
        )
        for direction in ("safe", "unsafe")
        for language in languages
    ]
    aligned = load_aligned_corpus(files, languages, expected_rows)
    return [
        CalibrationRow(
            base_id=row.canonical_id,
            row_id=row.row_id,
            language=row.language,
            direction=row.direction,
            category_id=row.category_id,
            prompt=row.prompt,
            source_path=row.source_path,
            source_line=row.source_line,
        )
        for row in aligned
    ]


def _interleave_trial_rows_for_workers(
    rows: list[GeometryRow],
    *,
    languages: tuple[str, ...],
    rows_per_cell: int,
) -> list[GeometryRow]:
    """Balance SAFE/UNSAFE and languages in every contiguous GPU shard."""

    by_cell = {
        (direction, language): [
            row
            for row in rows
            if row.direction == direction and row.language == language
        ]
        for language in languages
        for direction in ("safe", "unsafe")
    }
    if any(len(cell) != rows_per_cell for cell in by_cell.values()):
        raise ValueError("trial rows cannot be interleaved with incomplete cells")
    return [
        by_cell[(direction, language)][index]
        for index in range(rows_per_cell)
        for language in languages
        for direction in ("safe", "unsafe")
    ]


def _normalized_prompt_hash(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _assert_pool_prompt_disjoint(
    pools: dict[str, list[GeometryRow] | list[CalibrationRow]],
) -> int:
    return _assert_pool_prompt_overlap_contract(pools, allowed_map_trial_ids=set())


def _assert_pool_prompt_overlap_contract(
    pools: dict[str, list[GeometryRow] | list[CalibrationRow]],
    *,
    allowed_map_trial_ids: set[str],
) -> int:
    hashed = {
        name: {
            _normalized_prompt_hash(row.prompt): str(
                getattr(row, "canonical_id", getattr(row, "base_id", ""))
            )
            for row in rows
        }
        for name, rows in pools.items()
    }
    names = tuple(hashed)
    observed_allowed: set[str] = set()
    intentional_rows = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = set(hashed[left]) & set(hashed[right])
            if overlap:
                if {left, right} == {"map", "trial"}:
                    for prompt_hash in overlap:
                        left_id = hashed[left][prompt_hash]
                        right_id = hashed[right][prompt_hash]
                        if left_id != right_id or left_id not in allowed_map_trial_ids:
                            raise ValueError(
                                "unmanifested prompt overlap between map and trial"
                            )
                        observed_allowed.add(left_id)
                        intentional_rows += 1
                    continue
                raise ValueError(
                    f"prompt overlap between {left} and {right}: {len(overlap)}"
                )
    if observed_allowed != allowed_map_trial_ids:
        raise ValueError("intentional map/trial overlap manifest coverage mismatch")
    return intentional_rows


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
    languages: tuple[str, ...] = ("en", "ru", "zh", "ja"),
    direction_rows_per_cell: int = 1000,
    trial_rows_per_cell: int = 400,
    final_rows_per_cell: int = 200,
) -> MultilingualDatasetBundle:
    """Load map, rotating trial and independent directional final pools."""

    root = Path(dataset_root).resolve()
    split = (
        Path(split_root).resolve()
        if split_root is not None
        else root
    )
    dataset_manifest = root / "manifest.json"
    source_dataset_manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    raw_allowed_overlap = source_dataset_manifest.get(
        "intentional_map_trial_overlap_canonical_ids", []
    )
    if not isinstance(raw_allowed_overlap, list) or any(
        not isinstance(value, str) or not value for value in raw_allowed_overlap
    ):
        raise TypeError("intentional map/trial overlap IDs must be a string list")
    allowed_map_trial_ids = set(raw_allowed_overlap)
    normalized_languages = tuple(language.lower() for language in languages)
    if not normalized_languages or len(set(normalized_languages)) != len(
        normalized_languages
    ):
        raise ValueError("languages must be a non-empty unique sequence")
    if (
        min(
            direction_rows_per_cell,
            trial_rows_per_cell,
            final_rows_per_cell,
        )
        <= 0
    ):
        raise ValueError("all expected row counts must be positive")

    direction_files = [
        LanguageFile(
            language,
            direction,  # type: ignore[arg-type]
            split / f"map_{language}_{direction}_{direction_rows_per_cell}.jsonl",
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
    trial_rows = _interleave_trial_rows_for_workers(
        trial_rows,
        languages=normalized_languages,
        rows_per_cell=trial_rows_per_cell,
    )
    final_rows = _load_directional_final_pool(
        root,
        languages=normalized_languages,
        expected_rows=final_rows_per_cell,
    )
    intentional_overlap_rows = _assert_pool_prompt_overlap_contract(
        {
            "map": direction_rows,
            "trial": trial_rows,
            "final": final_rows,
        },
        allowed_map_trial_ids=allowed_map_trial_ids,
    )

    file_records: dict[str, dict[str, object]] = {}
    for pool, specifications, expected in (
        ("map", direction_files, direction_rows_per_cell),
        ("trial", trial_files, trial_rows_per_cell),
    ):
        for specification in specifications:
            path = Path(specification.path)
            ids = [
                row.canonical_id
                for row in (direction_rows if pool == "map" else trial_rows)
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
    for direction in ("safe", "unsafe"):
        for language in normalized_languages:
            path = root / f"final_{language}_{direction}_{final_rows_per_cell}.jsonl"
            ids = [
                row.base_id
                for row in final_rows
                if row.language == language and row.direction == direction
            ]
            file_records[path.name] = _file_record(
                path,
                relative_to=root,
                rows=final_rows_per_cell,
                pool="final",
                language=language,
                direction=direction,
                ids=ids,
            )

    split_manifest = split / "manifest.json"
    source_manifests = {
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS",
        "languages": list(normalized_languages),
        "counts": {
            "map": len(direction_rows),
            "trial": len(trial_rows),
            "final": len(final_rows),
        },
        "rows_per_cell": {
            "map": direction_rows_per_cell,
            "trial": trial_rows_per_cell,
            "final": final_rows_per_cell,
        },
        "source_manifests": source_manifests,
        "files": dict(sorted(file_records.items())),
        "intentional_map_trial_overlap_rows": intentional_overlap_rows,
        "unexpected_cross_pool_overlap": 0,
    }
    manifest["contract_sha256"] = _canonical_sha256(manifest)
    return MultilingualDatasetBundle(
        direction_rows=tuple(direction_rows),
        trial_rows=tuple(trial_rows),
        final_rows=tuple(final_rows),
        manifest=manifest,
    )
