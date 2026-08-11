from __future__ import annotations

from pathlib import Path

import pytest
import torch

from heretic.language_map_data import GeometryRow
from heretic.trial_geometry_metrics import evaluate_trial_geometry


def _rows() -> list[GeometryRow]:
    rows = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for language in ("en", "ru"):
            for category in ("C01", "C02"):
                rows.append(
                    GeometryRow(
                        canonical_id=f"{prefix}-{language}-{category}",
                        row_id=f"{language}-{prefix}-{category}",
                        language=language,
                        direction=direction,
                        category_id=category,
                        prompt="private",
                        source_path=Path("private.jsonl"),
                        source_line=1,
                    )
                )
    return rows


def test_uniform_unsafe_projection_reduction_has_gain_without_damage() -> None:
    rows = _rows()
    clean = torch.tensor(
        [[0.0, 0.0] if row.direction == "safe" else [2.0, 4.0] for row in rows]
    )
    candidate = torch.tensor(
        [[0.0, 0.0] if row.direction == "safe" else [1.0, 2.0] for row in rows]
    )

    metrics = evaluate_trial_geometry(
        rows,
        clean,
        candidate,
        layer_reliability=torch.tensor([1.0, 0.5]),
    )

    assert metrics["unsafe_geometry_gain"] > 0.0
    assert metrics["safe_geometry_damage"] == 0.0
    assert metrics["language_instability"] == pytest.approx(0.0)
    assert metrics["category_instability"] == pytest.approx(0.0)


def test_safe_displacement_is_damage_regardless_of_sign() -> None:
    rows = _rows()
    clean = torch.tensor(
        [[0.0, 0.0] if row.direction == "safe" else [2.0, 2.0] for row in rows]
    )
    candidate = clean.clone()
    for index, row in enumerate(rows):
        if row.direction == "safe":
            candidate[index] += 0.5 if row.language == "en" else -0.5

    metrics = evaluate_trial_geometry(
        rows,
        clean,
        candidate,
        layer_reliability=torch.ones(2),
    )

    assert metrics["unsafe_geometry_gain"] == pytest.approx(0.0)
    assert metrics["safe_geometry_damage"] > 0.0
    assert metrics["language_instability"] == pytest.approx(0.0)


def test_asymmetric_unsafe_change_is_reported_as_language_instability() -> None:
    rows = _rows()
    clean = torch.tensor(
        [[0.0, 0.0] if row.direction == "safe" else [2.0, 2.0] for row in rows]
    )
    candidate = clean.clone()
    for index, row in enumerate(rows):
        if row.direction == "unsafe" and row.language == "en":
            candidate[index] -= 1.0

    metrics = evaluate_trial_geometry(
        rows,
        clean,
        candidate,
        layer_reliability=torch.ones(2),
    )

    assert metrics["language_instability"] > 0.0
    assert metrics["category_instability"] == pytest.approx(0.0)


def test_geometry_inputs_must_align() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="aligned"):
        evaluate_trial_geometry(
            rows,
            torch.zeros((len(rows), 2)),
            torch.zeros((len(rows) - 1, 2)),
            layer_reliability=torch.ones(2),
        )
