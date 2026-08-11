# SPDX-License-Identifier: AGPL-3.0-or-later

"""Relative multilingual geometry metrics for one edited trial."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor

from .language_map_data import GeometryRow


def _macro(values: dict[tuple[str, ...], list[float]]) -> float:
    if not values:
        return 0.0
    return sum(sum(group) / len(group) for group in values.values()) / len(values)


def _dispersion(values: dict[str, list[float]]) -> float:
    if len(values) <= 1:
        return 0.0
    means = torch.tensor(
        [sum(group) / len(group) for _, group in sorted(values.items())],
        dtype=torch.float64,
    )
    return float(torch.std(means, correction=0))


def evaluate_trial_geometry(
    rows: Sequence[GeometryRow],
    clean_projection: Tensor,
    candidate_projection: Tensor,
    *,
    layer_reliability: Tensor,
) -> dict[str, object]:
    """Compare candidate projections to the exact clean rows in normalized units."""

    ordered = tuple(rows)
    if (
        not ordered
        or clean_projection.ndim != 2
        or candidate_projection.shape != clean_projection.shape
        or clean_projection.shape[0] != len(ordered)
    ):
        raise ValueError("rows and projection tensors must be aligned")
    reliability = layer_reliability.detach().to(torch.float64).flatten()
    if reliability.shape != (clean_projection.shape[1],):
        raise ValueError("layer reliability is not aligned with projections")
    if not bool(
        torch.isfinite(clean_projection).all()
        and torch.isfinite(candidate_projection).all()
        and torch.isfinite(reliability).all()
    ):
        raise ValueError("geometry metrics contain non-finite values")
    if bool(torch.any(reliability < 0)) or float(reliability.sum()) <= 0:
        raise ValueError("layer reliability must be nonnegative with positive mass")

    clean = clean_projection.detach().to(torch.float64)
    candidate = candidate_projection.detach().to(torch.float64)
    center = torch.quantile(clean, 0.5, dim=0)
    mad = 1.4826 * torch.quantile(torch.abs(clean - center), 0.5, dim=0)
    positive = mad[mad > 0]
    floor = max(
        float(torch.quantile(positive, 0.5)) * 0.25 if len(positive) else 0.0,
        1e-4,
    )
    scale = torch.clamp(mad, min=floor)
    standardized = torch.clamp((clean - candidate) / scale, min=-5.0, max=5.0)
    weights = reliability / reliability.sum()
    row_gain = standardized @ weights
    row_damage = torch.abs(standardized) @ weights

    unsafe_cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    safe_cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    unsafe_by_language: dict[str, list[float]] = defaultdict(list)
    safe_by_language: dict[str, list[float]] = defaultdict(list)
    unsafe_by_category: dict[str, list[float]] = defaultdict(list)
    safe_by_category: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(ordered):
        if row.direction == "unsafe":
            value = float(row_gain[index])
            unsafe_cells[(row.language, row.category_id)].append(value)
            unsafe_by_language[row.language].append(value)
            unsafe_by_category[row.category_id].append(value)
        else:
            value = float(row_damage[index])
            safe_cells[(row.language, row.category_id)].append(value)
            safe_by_language[row.language].append(value)
            safe_by_category[row.category_id].append(value)

    unsafe_macro = _macro(unsafe_cells)
    safe_damage = _macro(safe_cells)
    language_instability = 0.5 * (
        _dispersion(unsafe_by_language) + _dispersion(safe_by_language)
    )
    category_instability = 0.5 * (
        _dispersion(unsafe_by_category) + _dispersion(safe_by_category)
    )
    return {
        "unsafe_geometry_gain": math.tanh(unsafe_macro),
        "safe_geometry_damage": safe_damage,
        "language_instability": language_instability,
        "category_instability": category_instability,
        "standardized_unsafe_macro": unsafe_macro,
        "scale_floor": floor,
        "rows": len(ordered),
        "safe_rows": sum(row.direction == "safe" for row in ordered),
        "unsafe_rows": sum(row.direction == "unsafe" for row in ordered),
    }
