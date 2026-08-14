from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from heretic.language_map_projection import (
    fit_frozen_projection,
    project_residuals,
    retained_shift_ratio,
    write_projection_package,
)


def _residuals() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
            [[1.0, 2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 3.0]],
            [[3.0, 1.0, 0.0, 1.0], [2.0, 1.0, 1.0, 0.0]],
            [[2.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 1.0]],
        ],
        dtype=torch.float32,
    )


def _index() -> list[dict[str, object]]:
    languages = ("en", "ru", "zh", "es", "fr", "en")
    return [
        {
            "index": row,
            "canonical_id": f"A{row:04d}",
            "row_id": f"{languages[row].upper()}-A{row:04d}",
            "language": languages[row],
            "direction_class": "safe" if row % 2 == 0 else "unsafe",
            "category_id": f"C{row % 3 + 1:02d}",
            "category_ids": [f"C{row % 3 + 1:02d}", "shared"],
            "source_file": f"direction_{languages[row]}.jsonl",
            "source_line": row + 1,
        }
        for row in range(6)
    ]


def test_frozen_projection_is_deterministic_and_does_not_refit_candidates() -> None:
    residuals = _residuals()
    first = fit_frozen_projection(residuals, seed=17)
    second = fit_frozen_projection(residuals, seed=17)

    assert torch.equal(first.centers, second.centers)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.explained_variance, second.explained_variance)
    assert first.components.shape == (2, 3, 4)

    for layer in range(2):
        for component in range(3):
            vector = first.components[layer, component]
            pivot = int(torch.argmax(torch.abs(vector)))
            assert float(vector[pivot]) >= 0.0

    original_points = project_residuals(residuals, first)
    moved = residuals.clone()
    moved[:, :, 3] += 0.25
    moved_points = project_residuals(moved, first)
    assert original_points.shape == moved_points.shape == (6, 2, 3)
    assert torch.equal(first.components, second.components)


def test_retained_shift_ratio_reports_hidden_projection_movement() -> None:
    residuals = _residuals()
    basis = fit_frozen_projection(residuals, seed=11)
    moved = residuals.clone()
    moved[:, :, 3] += 1.0

    ratios = retained_shift_ratio(residuals, moved, basis)

    assert ratios.shape == (6, 2)
    assert bool(torch.isfinite(ratios).all())
    assert bool(((ratios >= 0.0) & (ratios <= 1.000001)).all())
    assert bool((ratios < 1.0).any())


def test_projection_package_is_atomic_hashed_and_text_free(tmp_path: Path) -> None:
    output = tmp_path / "geometry_3d"
    manifest = write_projection_package(
        index=_index(),
        residuals=_residuals(),
        output_dir=output,
        seed=23,
    )

    assert manifest["status"] == "PASS"
    assert manifest["shape"] == [6, 2, 3]
    assert not list(output.glob("*.tmp"))
    for name in ("projection_basis.safetensors", "base_points.f32", "base_index.json"):
        path = output / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name]["sha256"]

    public_index = json.loads((output / "base_index.json").read_text(encoding="utf-8"))
    assert len(public_index) == 6
    assert public_index[0]["direction_class"] == "safe"
    assert public_index[0]["category_ids"] == ["C01", "shared"]
    assert all("prompt" not in row and "response" not in row and "answer" not in row for row in public_index)
    coordinates = np.fromfile(output / "base_points.f32", dtype=np.float32)
    assert coordinates.size == 6 * 2 * 3

    with pytest.raises(FileExistsError):
        write_projection_package(
            index=_index(),
            residuals=_residuals(),
            output_dir=output,
            seed=23,
        )
