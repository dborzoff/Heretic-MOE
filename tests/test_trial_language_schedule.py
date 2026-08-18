import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from heretic.language_map_analysis import virtual_policy_indices
from heretic.trial_language_schedule import (
    load_trial_language_schedule,
    materialize_trial_language_schedule,
    trial_language_indices,
)

LANGUAGES = ("en", "ru", "zh", "es", "fr")
FOUR_LANGUAGES = ("en", "ru", "zh", "ja")


def _aligned_index(
    rows_per_direction: int = 400,
    languages: tuple[str, ...] = LANGUAGES,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for number in range(rows_per_direction):
            canonical_id = f"{prefix}{number + 1:04d}"
            category_id = f"C{number % 14 + 1:02d}"
            for language in languages:
                rows.append(
                    {
                        "canonical_id": canonical_id,
                        "row_id": f"{language.upper()}-{canonical_id}",
                        "language": language,
                        "direction_class": direction,
                        "category_id": category_id,
                    }
                )
    return rows


def test_scheduled_trial_selects_one_balanced_translation_per_id() -> None:
    index = _aligned_index()

    selected = trial_language_indices(
        index,
        mode="scheduled",
        languages=LANGUAGES,
        trial_number=17,
        seed=20260811,
    )

    assert len(selected) == 800
    chosen = [index[position] for position in selected]
    assert len(
        {
            (str(row["direction_class"]), str(row["canonical_id"]))
            for row in chosen
        }
    ) == 800
    assert Counter(
        (str(row["direction_class"]), str(row["language"])) for row in chosen
    ) == Counter(
        {
            (direction, language): 80
            for direction in ("safe", "unsafe")
            for language in LANGUAGES
        }
    )


def test_full_trial_mode_keeps_every_translation() -> None:
    index = _aligned_index()

    selected = trial_language_indices(
        index,
        mode="full",
        languages=LANGUAGES,
        trial_number=417,
        seed=20260811,
    )

    assert selected == list(range(4000))


def test_schedule_randomizes_five_trial_blocks_without_losing_coverage() -> None:
    index = _aligned_index()

    languages_by_trial: list[str] = []
    for trial_number in range(10):
        selected = trial_language_indices(
            index,
            mode="scheduled",
            languages=LANGUAGES,
            trial_number=trial_number,
            seed=20260811,
        )
        target = next(
            index[position]
            for position in selected
            if index[position]["canonical_id"] == "S0001"
        )
        languages_by_trial.append(str(target["language"]))

    assert set(languages_by_trial[:5]) == set(LANGUAGES)
    assert set(languages_by_trial[5:]) == set(LANGUAGES)
    assert languages_by_trial[:5] != languages_by_trial[5:]
    assert all(
        current != following
        for current, following in itertools.pairwise(languages_by_trial)
    )


def test_each_five_trial_block_uses_new_id_level_panels() -> None:
    index = _aligned_index()

    def block_panels(
        first_trial: int, direction: str
    ) -> set[frozenset[tuple[str, str]]]:
        panels = set()
        for trial_number in range(first_trial, first_trial + len(LANGUAGES)):
            selected = trial_language_indices(
                index,
                mode="scheduled",
                languages=LANGUAGES,
                trial_number=trial_number,
                seed=20260811,
            )
            panels.add(
                frozenset(
                    (
                        str(index[position]["canonical_id"]),
                        str(index[position]["language"]),
                    )
                    for position in selected
                    if index[position]["direction_class"] == direction
                )
            )
        return panels

    assert block_panels(0, "safe") != block_panels(5, "safe")
    assert block_panels(0, "unsafe") != block_panels(5, "unsafe")


def test_schedule_rejects_a_canonical_id_missing_a_translation() -> None:
    index = [
        row
        for row in _aligned_index()
        if not (row["canonical_id"] == "S0001" and row["language"] == "zh")
    ]

    with pytest.raises(ValueError, match="missing aligned languages"):
        trial_language_indices(
            index,
            mode="scheduled",
            languages=LANGUAGES,
            trial_number=0,
            seed=20260811,
        )


def test_schedule_rejects_category_drift_between_translations() -> None:
    index = _aligned_index()
    changed = next(
        row
        for row in index
        if row["canonical_id"] == "U0001" and row["language"] == "fr"
    )
    changed["category_id"] = "DRIFT"

    with pytest.raises(ValueError, match="category mismatch"):
        trial_language_indices(
            index,
            mode="scheduled",
            languages=LANGUAGES,
            trial_number=0,
            seed=20260811,
        )


def test_cached_geometry_accepts_the_scheduled_trial_policy() -> None:
    index = _aligned_index()

    selected = virtual_policy_indices(
        index,
        "scheduled_languages:417",
        seed=20260811,
    )

    chosen = [index[position] for position in selected]
    assert len(chosen) == 800
    assert Counter(str(row["language"]) for row in chosen) == Counter(
        {language: 160 for language in LANGUAGES}
    )


def test_schedule_is_materialized_text_free_and_resume_safe(tmp_path: Path) -> None:
    index = _aligned_index(rows_per_direction=10)
    output = tmp_path / "schedule"

    manifest = materialize_trial_language_schedule(
        index,
        output_dir=output,
        languages=LANGUAGES,
        seed=123,
        total_trials=10,
        expected_per_direction=10,
    )

    assert manifest["status"] == "PASS"
    assert manifest["trials"] == 10
    assert manifest["rows_per_trial"] == 20
    lines = [
        json.loads(line)
        for line in (output / "schedule.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["trial_number"] for row in lines] == list(range(10))
    assert all(len(row["row_ids"]) == 20 for row in lines)
    serialized = json.dumps(manifest, sort_keys=True)
    assert "prompt" not in serialized
    loaded_manifest, loaded_rows = load_trial_language_schedule(output)
    assert loaded_manifest == manifest
    assert [row["trial_number"] for row in loaded_rows] == list(range(10))
    assert materialize_trial_language_schedule(
        index,
        output_dir=output,
        languages=LANGUAGES,
        seed=123,
        total_trials=10,
        expected_per_direction=10,
    ) == manifest

    with pytest.raises(ValueError, match="contract"):
        materialize_trial_language_schedule(
            index,
            output_dir=output,
            languages=LANGUAGES,
            seed=124,
            total_trials=10,
            expected_per_direction=10,
        )


def test_four_language_schedule_freezes_exact_eight_trial_coverage(
    tmp_path: Path,
) -> None:
    index = _aligned_index(languages=FOUR_LANGUAGES)
    by_row_id = {str(row["row_id"]): row for row in index}
    output = tmp_path / "schedule-v5"

    manifest = materialize_trial_language_schedule(
        index,
        output_dir=output,
        languages=FOUR_LANGUAGES,
        seed=20260817,
        total_trials=600,
        expected_per_direction=200,
    )
    _loaded_manifest, records = load_trial_language_schedule(output)

    assert manifest["trials"] == 600
    assert manifest["source_rows_per_direction"] == 400
    assert manifest["expected_per_direction"] == 200
    assert manifest["rows_per_trial"] == 400
    assert manifest["coverage_block_trials"] == 8
    assert len(records) == 600

    for record in records:
        chosen = [by_row_id[row_id] for row_id in record["row_ids"]]
        assert len(chosen) == 400
        assert Counter(str(row["direction_class"]) for row in chosen) == Counter(
            {"safe": 200, "unsafe": 200}
        )
        assert Counter(
            (str(row["direction_class"]), str(row["language"]))
            for row in chosen
        ) == Counter(
            {
                (direction, language): 50
                for direction in ("safe", "unsafe")
                for language in FOUR_LANGUAGES
            }
        )
        for direction in ("safe", "unsafe"):
            category_counts = Counter(
                str(row["category_id"])
                for row in chosen
                if row["direction_class"] == direction
            )
            assert len(category_counts) == 14
            assert max(category_counts.values()) - min(category_counts.values()) <= 3
        directions = [str(row["direction_class"]) for row in chosen]
        assert any(
            left != right for left, right in itertools.pairwise(directions)
        )

    for pair_start in range(0, 600, 2):
        pair = records[pair_start : pair_start + 2]
        for direction in ("safe", "unsafe"):
            canonical = [
                str(by_row_id[row_id]["canonical_id"])
                for record in pair
                for row_id in record["row_ids"]
                if by_row_id[row_id]["direction_class"] == direction
            ]
            assert len(canonical) == 400
            assert len(set(canonical)) == 400

    for block_start in range(0, 600, 8):
        block = records[block_start : block_start + 8]
        for direction in ("safe", "unsafe"):
            cells = [
                (
                    str(by_row_id[row_id]["canonical_id"]),
                    str(by_row_id[row_id]["language"]),
                )
                for record in block
                for row_id in record["row_ids"]
                if by_row_id[row_id]["direction_class"] == direction
            ]
            assert len(cells) == 1600
            assert len(set(cells)) == 1600

    assert records[:8] != records[8:16]


def test_hard_soft_schedule_freezes_exact_sixteen_trial_cycle(
    tmp_path: Path,
) -> None:
    index = _aligned_index(rows_per_direction=800, languages=FOUR_LANGUAGES)
    for row in index:
        if row["direction_class"] == "safe":
            row["trial_behavior_class"] = "safe"
        else:
            number = int(str(row["canonical_id"])[1:])
            row["trial_behavior_class"] = "hard" if number <= 400 else "soft"
    by_row_id = {str(row["row_id"]): row for row in index}
    output = tmp_path / "hard-soft-schedule"

    manifest = materialize_trial_language_schedule(
        index,
        output_dir=output,
        languages=FOUR_LANGUAGES,
        seed=20260818,
        total_trials=32,
        expected_per_direction=200,
    )
    _, records = load_trial_language_schedule(output)

    assert manifest["coverage_block_trials"] == 16
    assert manifest["behavior_rows_per_trial"] == {
        "hard": 100,
        "safe": 200,
        "soft": 100,
    }
    for record in records:
        chosen = [by_row_id[row_id] for row_id in record["row_ids"]]
        assert Counter(str(row["trial_behavior_class"]) for row in chosen) == {
            "hard": 100,
            "safe": 200,
            "soft": 100,
        }
        assert Counter(
            (str(row["trial_behavior_class"]), str(row["language"]))
            for row in chosen
        ) == Counter(
            {
                (behavior, language): rows
                for behavior, rows in (("hard", 25), ("soft", 25), ("safe", 50))
                for language in FOUR_LANGUAGES
            }
        )
    for block_start in (0, 16):
        block_ids = [
            row_id
            for record in records[block_start : block_start + 16]
            for row_id in record["row_ids"]
        ]
        assert len(block_ids) == 6400
        assert len(set(block_ids)) == 6400
    assert records[:16] != records[16:32]
