# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic multilingual prompt panels for trial evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal


TrialLanguageMode = Literal["full", "scheduled"]


def _stable_key(*parts: object) -> bytes:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()


def _base_slots(
    groups: dict[tuple[str, str], list[int]],
    index: list[dict[str, object]],
    *,
    direction: str,
    language_count: int,
    seed: int,
    block: int,
) -> dict[tuple[str, str], int]:
    by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, positions in groups.items():
        if key[0] != direction:
            continue
        category = str(index[positions[0]]["category_id"])
        by_category[category].append(key)

    result: dict[tuple[str, str], int] = {}
    cursor = 0
    for category in sorted(by_category):
        keys = sorted(
            by_category[category],
            key=lambda key: _stable_key(
                seed, "block-slots", block, direction, category, key[1]
            ),
        )
        for key in keys:
            result[key] = cursor % language_count
            cursor += 1
    return result


def _phase_for_trial(
    *,
    direction: str,
    language_count: int,
    seed: int,
    trial_number: int,
) -> int:
    block = trial_number // language_count
    position = trial_number % language_count
    previous_last: int | None = None
    current: list[int] = []
    for block_number in range(block + 1):
        current = sorted(
            range(language_count),
            key=lambda phase: _stable_key(
                seed, "phase", direction, block_number, phase
            ),
        )
        if previous_last is not None and current[0] == previous_last:
            current[0], current[1] = current[1], current[0]
        previous_last = current[-1]
    return current[position]


def trial_language_indices(
    index: list[dict[str, object]],
    *,
    mode: TrialLanguageMode,
    languages: Sequence[str],
    trial_number: int,
    seed: int,
) -> list[int]:
    """Select full translations or one balanced translation per canonical ID."""

    if mode == "full":
        return list(range(len(index)))
    if mode != "scheduled":
        raise ValueError(f"unknown trial language mode: {mode}")
    normalized_languages = tuple(str(language).lower() for language in languages)
    if not normalized_languages or len(set(normalized_languages)) != len(
        normalized_languages
    ):
        raise ValueError("languages must be a non-empty unique sequence")
    if trial_number < 0:
        raise ValueError("trial_number must be non-negative")

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(index):
        key = (str(row["direction_class"]).lower(), str(row["canonical_id"]))
        groups[key].append(position)

    expected_languages = set(normalized_languages)
    for positions in groups.values():
        observed = [
            str(index[position]["language"]).lower() for position in positions
        ]
        if (
            len(observed) != len(normalized_languages)
            or set(observed) != expected_languages
        ):
            raise ValueError("canonical group has missing aligned languages")
        categories = {
            str(index[position]["category_id"]) for position in positions
        }
        if len(categories) != 1:
            raise ValueError("canonical group category mismatch between translations")

    directions = sorted({key[0] for key in groups})
    block = trial_number // len(normalized_languages)
    selected: list[int] = []
    for direction in directions:
        slots = _base_slots(
            groups,
            index,
            direction=direction,
            language_count=len(normalized_languages),
            seed=seed,
            block=block,
        )
        phase = _phase_for_trial(
            direction=direction,
            language_count=len(normalized_languages),
            seed=seed,
            trial_number=trial_number,
        )
        for key, base_slot in slots.items():
            by_language = {
                str(index[position]["language"]).lower(): position
                for position in groups[key]
            }
            language = normalized_languages[
                (base_slot + phase) % len(normalized_languages)
            ]
            selected.append(by_language[language])
    return sorted(selected)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_trial_language_schedule(
    index: list[dict[str, object]],
    *,
    output_dir: str | Path,
    languages: Sequence[str],
    seed: int,
    total_trials: int,
    expected_per_direction: int = 400,
) -> dict[str, object]:
    """Freeze row-ID assignments for every trial without exposing prompt text."""

    if total_trials <= 0 or expected_per_direction <= 0:
        raise ValueError("schedule sizes must be positive")
    normalized_languages = tuple(str(language).lower() for language in languages)
    row_ids = [str(row.get("row_id", "")) for row in index]
    if not all(row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("schedule index row IDs must be non-empty and unique")
    contract = {
        "schema_version": 1,
        "languages": list(normalized_languages),
        "seed": int(seed),
        "trials": int(total_trials),
        "expected_per_direction": int(expected_per_direction),
        "index_sha256": _canonical_sha256(
            [
                {
                    "canonical_id": str(row.get("canonical_id", "")),
                    "row_id": str(row.get("row_id", "")),
                    "language": str(row.get("language", "")).lower(),
                    "direction_class": str(row.get("direction_class", "")).lower(),
                    "category_id": str(row.get("category_id", "")),
                }
                for row in index
            ]
        ),
    }
    contract_sha = _canonical_sha256(contract)
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    schedule_path = output / "schedule.jsonl"
    if output.exists():
        if not manifest_path.is_file() or not schedule_path.is_file():
            raise ValueError("existing schedule package is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "PASS"
            or manifest.get("schedule_contract_sha256") != contract_sha
            or manifest.get("schedule_sha256") != _file_sha256(schedule_path)
        ):
            raise ValueError("existing schedule package contract mismatch")
        return manifest

    records = []
    rows_per_trial = expected_per_direction * 2
    for trial_number in range(total_trials):
        selected = trial_language_indices(
            index,
            mode="scheduled",
            languages=normalized_languages,
            trial_number=trial_number,
            seed=seed,
        )
        if len(selected) != rows_per_trial:
            raise ValueError("scheduled trial row count mismatch")
        records.append(
            {
                "trial_number": trial_number,
                "row_ids": [row_ids[position] for position in selected],
            }
        )
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        payload = "".join(
            json.dumps(record, separators=(",", ":")) + "\n" for record in records
        ).encode("utf-8")
        temporary_schedule = temporary / "schedule.jsonl"
        temporary_schedule.write_bytes(payload)
        manifest: dict[str, object] = {
            **contract,
            "status": "PASS",
            "schedule_contract_sha256": contract_sha,
            "rows_per_trial": rows_per_trial,
            "schedule_sha256": _file_sha256(temporary_schedule),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_trial_language_schedule(
    input_dir: str | Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Verify and load a frozen text-free schedule package."""

    source = Path(input_dir).resolve()
    manifest_path = source / "manifest.json"
    schedule_path = source / "schedule.jsonl"
    if not manifest_path.is_file() or not schedule_path.is_file():
        raise FileNotFoundError("trial language schedule package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("schedule_sha256") != _file_sha256(schedule_path)
    ):
        raise ValueError("trial language schedule hash mismatch")
    contract = {
        key: manifest[key]
        for key in (
            "schema_version",
            "languages",
            "seed",
            "trials",
            "expected_per_direction",
            "index_sha256",
        )
    }
    if manifest.get("schedule_contract_sha256") != _canonical_sha256(contract):
        raise ValueError("trial language schedule contract hash mismatch")
    records = [
        json.loads(line)
        for line in schedule_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_trials = int(manifest["trials"])
    expected_rows = int(manifest["rows_per_trial"])
    if [record.get("trial_number") for record in records] != list(
        range(expected_trials)
    ):
        raise ValueError("trial language schedule number coverage mismatch")
    for record in records:
        row_ids = record.get("row_ids")
        if (
            not isinstance(row_ids, list)
            or len(row_ids) != expected_rows
            or len(set(row_ids)) != expected_rows
            or any(not isinstance(row_id, str) or not row_id for row_id in row_ids)
        ):
            raise ValueError("trial language schedule row coverage mismatch")
    return manifest, records
