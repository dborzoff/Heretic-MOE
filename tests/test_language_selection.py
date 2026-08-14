from __future__ import annotations

import pytest
import torch

from heretic.language_selection import (
    build_language_distance_report,
    combine_language_distance_reports,
    write_language_selection_report,
)


def _geometry() -> tuple[list[dict[str, object]], torch.Tensor]:
    languages = ("en", "es", "ru", "zh", "fr", "ja")
    offsets = {
        "en": torch.tensor((0.0, 0.0, 0.0, 0.0)),
        "es": torch.tensor((0.01, 0.0, 0.0, 0.0)),
        "fr": torch.tensor((0.02, 0.0, 0.0, 0.0)),
        "ru": torch.tensor((0.0, 1.5, 0.0, 0.0)),
        "zh": torch.tensor((0.0, 0.0, 1.8, 0.0)),
        "ja": torch.tensor((0.0, 0.0, 0.0, 1.2)),
    }
    rows: list[dict[str, object]] = []
    values = []
    for direction, prefix, sign in (("safe", "S", -1.0), ("unsafe", "U", 1.0)):
        for language in languages:
            for item in range(4):
                category = "S1" if item < 2 else "S2"
                rows.append(
                    {
                        "canonical_id": f"{prefix}{item:04d}",
                        "row_id": f"{language}-{prefix}{item:04d}",
                        "language": language,
                        "direction_class": direction,
                        "category_id": category,
                        "category_ids": [category, "S9"] if item == 0 else [category],
                    }
                )
                base = torch.tensor((sign * (1 + item / 10), 0.0, 0.0, 0.0))
                # Give each distant language a different direction interaction too.
                interaction = offsets[language] * sign * 0.25
                vector = base + offsets[language] + interaction
                values.append(torch.stack((vector, vector * 1.5)))
    return rows, torch.stack(values)


def test_full_space_language_distances_are_symmetric_and_category_aware() -> None:
    index, residuals = _geometry()

    report = build_language_distance_report(index, residuals)

    assert report["status"] == "PASS"
    assert report["method"] == "full_space_language_geometry_v1"
    assert report["languages"] == ["en", "es", "fr", "ja", "ru", "zh"]
    distances = torch.tensor(report["distances"])
    assert torch.allclose(distances, distances.T)
    assert torch.allclose(torch.diag(distances), torch.zeros(6))
    assert distances[0, 1] < distances[0, 4]
    assert "S9" in report["categories"]


def test_combined_k_medoids_keeps_english_and_prefers_nonredundant_languages() -> None:
    index, residuals = _geometry()
    first = build_language_distance_report(index, residuals)
    second = build_language_distance_report(index, residuals * 2)

    result = combine_language_distance_reports([first, second], min_k=4, max_k=6)

    assert result["status"] == "PASS"
    assert result["models"] == 2
    assert result["recommended_k"] in {4, 5, 6}
    assert "en" in result["recommended_languages"]
    assert len(result["recommended_languages"]) == result["recommended_k"]
    assert set(result["candidates"]) == {"4", "5", "6"}
    assert result["recommended_languages"] == combine_language_distance_reports(
        [first, second], min_k=4, max_k=6
    )["recommended_languages"]


def test_combiner_rejects_misaligned_language_axes() -> None:
    index, residuals = _geometry()
    report = build_language_distance_report(index, residuals)
    changed = dict(report)
    changed["languages"] = list(reversed(report["languages"]))

    with pytest.raises(ValueError, match="language axes"):
        combine_language_distance_reports([report, changed], min_k=4, max_k=5)


def test_selection_writer_creates_text_free_heatmap(tmp_path) -> None:
    index, residuals = _geometry()
    report = combine_language_distance_reports(
        [build_language_distance_report(index, residuals)], min_k=4, max_k=5
    )

    write_language_selection_report(report, tmp_path)

    document = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert (tmp_path / "language_selection.json").is_file()
    assert "Language distance heatmap" in document
    assert "recommended_languages" in document
    assert '"prompt"' not in document.lower()
