from collections import Counter

import pytest

from heretic.language_map_analysis import virtual_policy_indices
from heretic.trial_language_schedule import trial_language_indices


LANGUAGES = ("en", "ru", "zh", "es", "fr")


def _aligned_index(rows_per_direction: int = 400) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for number in range(rows_per_direction):
            canonical_id = f"{prefix}{number + 1:04d}"
            category_id = f"C{number % 14 + 1:02d}"
            for language in LANGUAGES:
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
        for current, following in zip(languages_by_trial, languages_by_trial[1:])
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
