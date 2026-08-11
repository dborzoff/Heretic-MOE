# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic multilingual prompt panels for trial evaluation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
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
