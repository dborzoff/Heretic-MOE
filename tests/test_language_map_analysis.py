import json
from pathlib import Path

import pytest
import torch

from heretic.language_map_analysis import (
    analyze_geometry,
    virtual_policy_indices,
    write_geometry_reports,
)


def aligned_index() -> list[dict[str, object]]:
    rows = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for language in ("en", "ru"):
            for item in range(2):
                canonical_id = f"{prefix}{item + 1:04d}"
                rows.append(
                    {
                        "index": len(rows),
                        "canonical_id": canonical_id,
                        "row_id": f"{language.upper()}-{canonical_id}",
                        "language": language,
                        "direction_class": direction,
                        "category_id": f"C{item + 1:02d}",
                    }
                )
    return rows


def redundant_residuals() -> torch.Tensor:
    # Two layers, three components. Translation pairs are exactly identical.
    vectors = {
        "S0001": (1.0, 0.0, 0.0),
        "S0002": (0.8, 0.2, 0.0),
        "U0001": (0.0, 1.0, 0.0),
        "U0002": (0.2, 0.8, 0.0),
    }
    rows = []
    for row in aligned_index():
        vector = torch.tensor(vectors[str(row["canonical_id"])])
        rows.append(torch.stack((vector, vector * 2)))
    return torch.stack(rows)


def unique_ru_branch_residuals() -> torch.Tensor:
    residuals = redundant_residuals()
    index = aligned_index()
    for row in index:
        if (
            row["language"] == "ru"
            and row["direction_class"] == "unsafe"
            and row["canonical_id"] == "U0001"
        ):
            residuals[int(row["index"]), :, 2] += 2.0
    return residuals


def test_redundant_language_has_zero_marginal_direction_loss():
    report = analyze_geometry(aligned_index(), redundant_residuals())

    assert report["language_contributions"]["ru"]["direction_loss"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert report["factor_map"]["layers"][0]["language_energy_ratio"] == pytest.approx(
        0.0, abs=1e-7
    )


def test_unique_language_branch_is_retained_as_cold_not_deleted():
    report = analyze_geometry(aligned_index(), unique_ru_branch_residuals())

    assert report["language_contributions"]["ru"]["direction_loss"] > 0
    assert report["temperature"]["cold_observations"] > 0
    assert report["temperature"]["discarded_observations"] == 0


def test_nonduplicated_policy_uses_one_language_per_canonical_id():
    index = aligned_index()
    selected = virtual_policy_indices(index, "cycle_languages", seed=42)
    selected_rows = [index[position] for position in selected]
    keys = [
        (row["direction_class"], row["canonical_id"]) for row in selected_rows
    ]

    assert len(selected_rows) == 4
    assert len(set(keys)) == len(keys)
    assert {row["language"] for row in selected_rows} == {"en", "ru"}


def test_report_compares_every_scheduled_language_panel():
    report = analyze_geometry(aligned_index(), redundant_residuals())
    policies = {entry["policy"] for entry in report["subset_candidates"]}

    assert {"scheduled_languages:0", "scheduled_languages:1"} <= policies


def test_weighted_and_fractional_policies_are_cached_only_and_deterministic():
    index = []
    for direction in ("safe", "unsafe"):
        for canonical in range(100):
            for language in ("en", "ru"):
                index.append(
                    {
                        "canonical_id": f"{direction}-{canonical:04d}",
                        "row_id": f"{direction}-{canonical:04d}-{language}",
                        "language": language,
                        "direction_class": direction,
                        "category_id": "C01",
                    }
                )

    weighted_policy = 'weighted:{"en":0.9,"ru":0.1}'
    weighted = virtual_policy_indices(index, weighted_policy, seed=42)
    repeated = virtual_policy_indices(index, weighted_policy, seed=42)
    reduced = virtual_policy_indices(
        index, "fraction:0.5:cycle_languages", seed=42
    )

    assert weighted == repeated
    assert sum(index[position]["language"] == "en" for position in weighted) > 150
    assert 0 < len(reduced) < 200
    assert len(
        {
            (index[position]["direction_class"], index[position]["canonical_id"])
            for position in reduced
        }
    ) == len(reduced)


def test_report_writer_is_text_free(tmp_path: Path):
    report = analyze_geometry(aligned_index(), redundant_residuals())
    write_geometry_reports(report, tmp_path)

    expected = {
        "component_regions.json",
        "layer_statistics.json",
        "factor_map.json",
        "language_contributions.json",
        "language_distances.json",
        "language_selection.json",
        "language_selection",
        "subset_candidates.json",
        "report.html",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    for path in (value for value in tmp_path.rglob("*") if value.is_file()):
        content = path.read_text(encoding="utf-8").lower()
        assert '"prompt"' not in content
        assert '"response"' not in content
        assert '"answer"' not in content
    assert json.loads((tmp_path / "factor_map.json").read_text(encoding="utf-8"))[
        "category_is_nested_in_direction"
    ] is True
    selection = json.loads(
        (tmp_path / "language_selection.json").read_text(encoding="utf-8")
    )
    assert selection["fixed_language"] == "en"


def test_public_report_uses_neutral_group_names():
    report = analyze_geometry(aligned_index(), redundant_residuals())
    serialized = json.dumps(report).lower()

    assert "group_a_cohesion" in serialized
    assert "group_b_cohesion" in serialized
    assert "contrast_heat" in serialized
    assert "refusal" not in serialized
    assert "unsafe_cohesion" not in serialized
    assert "safe_cohesion" not in serialized


def test_component_regions_separate_shared_and_conditioned_effects():
    index = []
    values = []
    for direction in ("safe", "unsafe"):
        for language in ("en", "ru"):
            for category in ("C01", "C02"):
                for repeat in range(2):
                    index.append(
                        {
                            "canonical_id": f"{direction}-{category}-{repeat}",
                            "row_id": f"{direction}-{language}-{category}-{repeat}",
                            "language": language,
                            "direction_class": direction,
                            "category_id": category,
                        }
                    )
                    vector = torch.zeros(100)
                    if direction == "unsafe":
                        vector[0] += 4.0
                    if language == "ru":
                        vector[1] += 4.0
                    if direction == "unsafe" and language == "ru":
                        vector[2] += 8.0
                    if direction == "unsafe" and category == "C02":
                        vector[3] += 8.0
                    values.append(vector)

    residuals = torch.stack(values).unsqueeze(1)
    report = analyze_geometry(index, residuals)
    regions = report["component_regions"]["layers"][0]["regions"]

    assert regions["shared_contrast"][0]["component"] == 0
    assert regions["language_core"][0]["component"] == 1
    assert regions["language_conditioned_contrast"][0]["component"] == 2
    assert regions["category_conditioned_contrast"][0]["component"] == 3
