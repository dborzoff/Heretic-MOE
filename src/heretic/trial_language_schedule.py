# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic multilingual prompt panels for trial evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict, deque
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


def _balanced_language_panels(
    assigned: dict[str, list[tuple[str, str]]],
    groups: dict[tuple[str, str], list[int]],
    index: list[dict[str, object]],
    *,
    panel_count: int,
    seed: int,
    block: int,
    phase_position: int,
    direction: str,
) -> dict[str, list[list[tuple[str, str]]]]:
    """Split assigned rows with exact language and near-exact category balance."""

    if panel_count == 1:
        return {language: [list(keys)] for language, keys in assigned.items()}
    if panel_count != 2:
        raise ValueError("scheduled language panels currently require two halves")
    panels = {
        language: [[], []] for language in assigned
    }
    oddments: dict[tuple[str, str], tuple[str, str]] = {}
    category_totals: Counter[str] = Counter()
    for language in sorted(assigned):
        by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key in assigned[language]:
            category = str(index[groups[key][0]]["category_id"])
            by_category[category].append(key)
            category_totals[category] += 1
        for category in sorted(by_category):
            ordered = sorted(
                by_category[category],
                key=lambda key: _stable_key(
                    seed,
                    "panel-category",
                    block,
                    phase_position,
                    direction,
                    language,
                    category,
                    key[1],
                ),
            )
            half = len(ordered) // 2
            panels[language][0].extend(ordered[:half])
            panels[language][1].extend(ordered[half : half * 2])
            if len(ordered) % 2:
                oddments[(language, category)] = ordered[-1]

    first_panel_target = sum(len(keys) for keys in assigned.values()) // 2
    category_targets = {
        category: count // 2 for category, count in category_totals.items()
    }
    target_oddments = first_panel_target - sum(category_targets.values())
    odd_categories = sorted(
        (category for category, count in category_totals.items() if count % 2),
        key=lambda category: _stable_key(
            seed,
            "category-oddment",
            block,
            phase_position,
            direction,
            category,
        ),
    )
    for category in odd_categories[:target_oddments]:
        category_targets[category] += 1

    language_deficits = {
        language: len(assigned[language]) // 2 - len(panels[language][0])
        for language in assigned
    }
    first_category_counts = Counter(
        str(index[groups[key][0]]["category_id"])
        for language_panels in panels.values()
        for key in language_panels[0]
    )
    category_deficits = {
        category: category_targets[category] - first_category_counts[category]
        for category in category_targets
    }

    source = ("source", "")
    sink = ("sink", "")
    residual: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    def add_edge(left: tuple[str, str], right: tuple[str, str], capacity: int) -> None:
        residual[(left, right)] = capacity
        residual[(right, left)] = 0
        adjacency[left].append(right)
        adjacency[right].append(left)

    for language in sorted(assigned):
        add_edge(source, ("language", language), language_deficits[language])
    for language, category in sorted(
        oddments,
        key=lambda value: _stable_key(
            seed,
            "oddment-edge",
            block,
            phase_position,
            direction,
            value[0],
            value[1],
        ),
    ):
        add_edge(("language", language), ("category", category), 1)
    for category in sorted(category_targets):
        add_edge(("category", category), sink, category_deficits[category])

    required_flow = sum(language_deficits.values())
    delivered_flow = 0
    while delivered_flow < required_flow:
        previous: dict[tuple[str, str], tuple[str, str] | None] = {source: None}
        pending = deque([source])
        while pending and sink not in previous:
            left = pending.popleft()
            for right in adjacency[left]:
                if right not in previous and residual[(left, right)] > 0:
                    previous[right] = left
                    pending.append(right)
        if sink not in previous:
            raise ValueError("category-balanced language panel flow is infeasible")
        current = sink
        while previous[current] is not None:
            left = previous[current]
            residual[(left, current)] -= 1
            residual[(current, left)] += 1
            current = left
        delivered_flow += 1

    for (language, category), key in oddments.items():
        first = residual[("category", category), ("language", language)] == 1
        panels[language][0 if first else 1].append(key)

    for language, language_panels in panels.items():
        target = len(assigned[language]) // 2
        if any(len(panel) != target for panel in language_panels):
            raise ValueError("scheduled language panel cannot be balanced exactly")
    return panels


def trial_language_indices(
    index: list[dict[str, object]],
    *,
    mode: TrialLanguageMode,
    languages: Sequence[str],
    trial_number: int,
    seed: int,
    expected_per_direction: int | None = None,
) -> list[int]:
    """Select one frozen, category-balanced multilingual trial panel."""

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
    source_counts = {
        direction: sum(key[0] == direction for key in groups)
        for direction in directions
    }
    if len(set(source_counts.values())) != 1:
        raise ValueError("directions have different canonical row counts")
    source_rows_per_direction = next(iter(source_counts.values()))
    trial_rows_per_direction = (
        source_rows_per_direction
        if expected_per_direction is None
        else int(expected_per_direction)
    )
    if (
        trial_rows_per_direction <= 0
        or source_rows_per_direction % trial_rows_per_direction != 0
        or trial_rows_per_direction % len(normalized_languages) != 0
    ):
        raise ValueError("scheduled trial rows cannot form balanced coverage blocks")
    panel_count = source_rows_per_direction // trial_rows_per_direction
    coverage_block_trials = panel_count * len(normalized_languages)
    block = trial_number // coverage_block_trials
    block_position = trial_number % coverage_block_trials
    phase_position = block_position // panel_count
    panel_index = block_position % panel_count
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
            trial_number=block * len(normalized_languages) + phase_position,
        )
        assigned: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key, base_slot in slots.items():
            language = normalized_languages[
                (base_slot + phase) % len(normalized_languages)
            ]
            assigned[language].append(key)
        assigned_panels = _balanced_language_panels(
            assigned,
            groups,
            index,
            panel_count=panel_count,
            seed=seed,
            block=block,
            phase_position=phase_position,
            direction=direction,
        )
        for language in normalized_languages:
            language_keys = assigned[language]
            expected_source_language = source_rows_per_direction // len(
                normalized_languages
            )
            if len(language_keys) != expected_source_language:
                raise ValueError("scheduled source language assignment is unbalanced")
            chosen_keys = assigned_panels[language][panel_index]
            for key in chosen_keys:
                by_language = {
                    str(index[position]["language"]).lower(): position
                    for position in groups[key]
                }
                selected.append(by_language[language])
    return sorted(
        selected,
        key=lambda position: _stable_key(
            seed,
            "trial-row-order",
            trial_number,
            index[position]["row_id"],
        ),
    )


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
    groups_by_direction = Counter(
        (
            str(row.get("direction_class", "")).lower(),
            str(row.get("canonical_id", "")),
        )
        for row in index
    )
    source_counts = Counter(direction for direction, _ in groups_by_direction)
    if set(source_counts) != {"safe", "unsafe"} or len(set(source_counts.values())) != 1:
        raise ValueError("schedule source direction coverage mismatch")
    source_rows_per_direction = next(iter(source_counts.values()))
    if (
        source_rows_per_direction % expected_per_direction != 0
        or expected_per_direction % len(normalized_languages) != 0
    ):
        raise ValueError("schedule sizes cannot form exact multilingual blocks")
    coverage_block_trials = (
        source_rows_per_direction // expected_per_direction
    ) * len(normalized_languages)
    if total_trials % coverage_block_trials != 0:
        raise ValueError("total trials must contain complete coverage blocks")
    contract = {
        "schema_version": 2,
        "languages": list(normalized_languages),
        "seed": int(seed),
        "trials": int(total_trials),
        "source_rows_per_direction": int(source_rows_per_direction),
        "expected_per_direction": int(expected_per_direction),
        "coverage_block_trials": int(coverage_block_trials),
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
            expected_per_direction=expected_per_direction,
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
    contract_keys = [
        "schema_version",
        "languages",
        "seed",
        "trials",
        "expected_per_direction",
        "index_sha256",
    ]
    if int(manifest.get("schema_version", 1)) >= 2:
        contract_keys.extend(
            ("source_rows_per_direction", "coverage_block_trials")
        )
    contract = {key: manifest[key] for key in contract_keys}
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
