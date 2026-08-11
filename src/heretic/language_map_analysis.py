# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure cached mathematics for multilingual Heretic geometry maps."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


_SENSITIVE_KEYS = {"prompt", "response", "answer", "text"}


def _metadata(index: list[dict[str, object]], key: str) -> list[str]:
    return [str(row[key]).lower() for row in index]


def _validate(index: list[dict[str, object]], residuals: Tensor) -> None:
    if residuals.ndim != 3:
        raise ValueError("residuals must have shape [rows, layers, hidden]")
    if len(index) != residuals.shape[0] or not index:
        raise ValueError("index/residual row count mismatch")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals contain non-finite values")
    required = {
        "canonical_id",
        "row_id",
        "language",
        "direction_class",
        "category_id",
    }
    for row in index:
        if not required.issubset(row):
            raise ValueError("row index is missing required metadata")


def _stable_unit_interval(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def virtual_policy_indices(
    index: list[dict[str, object]], policy: str, seed: int = 42
) -> list[int]:
    """Select cached rows for a virtual corpus without translated duplicates."""

    languages = sorted(set(_metadata(index, "language")))
    if policy == "full":
        return list(range(len(index)))
    if policy == "en_only":
        return [i for i, row in enumerate(index) if str(row["language"]).lower() == "en"]
    if policy.startswith("leave_out:"):
        excluded = policy.split(":", 1)[1].lower()
        if excluded not in languages:
            raise ValueError(f"unknown language in policy: {excluded}")
        return [
            i
            for i, row in enumerate(index)
            if str(row["language"]).lower() != excluded
        ]
    if policy.startswith("fraction:"):
        _, raw_fraction, base_policy = policy.split(":", 2)
        fraction = float(raw_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("fraction policy must be in (0, 1]")
        base = virtual_policy_indices(index, base_policy, seed=seed)
        selected = [
            position
            for position in base
            if _stable_unit_interval(
                seed,
                "fraction:"
                + str(index[position]["direction_class"])
                + ":"
                + str(index[position]["canonical_id"]),
            )
            <= fraction
        ]
        required_directions = {
            str(row["direction_class"]).lower() for row in index
        }
        selected_directions = {
            str(index[position]["direction_class"]).lower()
            for position in selected
        }
        for missing in sorted(required_directions - selected_directions):
            candidates = [
                position
                for position in base
                if str(index[position]["direction_class"]).lower() == missing
            ]
            selected.append(
                min(
                    candidates,
                    key=lambda position: _stable_unit_interval(
                        seed,
                        "fraction:"
                        + str(index[position]["direction_class"])
                        + ":"
                        + str(index[position]["canonical_id"]),
                    ),
                )
            )
        return sorted(selected)

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(index):
        groups[(str(row["direction_class"]), str(row["canonical_id"]))].append(
            position
        )
    ordered_groups = list(groups.items())

    if policy == "cycle_languages":
        selected = []
        for group_number, (_, positions) in enumerate(ordered_groups):
            positions = sorted(
                positions, key=lambda value: str(index[value]["language"])
            )
            selected.append(positions[(group_number + seed) % len(positions)])
        return sorted(selected)

    if policy.startswith("weighted:"):
        weights = json.loads(policy.split(":", 1)[1])
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weighted policy must contain a language-weight object")
        normalized = {str(key).lower(): float(value) for key, value in weights.items()}
        if set(normalized) - set(languages) or any(value < 0 for value in normalized.values()):
            raise ValueError("weighted policy contains invalid language weights")
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("weighted policy weights must sum above zero")
        cumulative: list[tuple[str, float]] = []
        running = 0.0
        for language in languages:
            running += normalized.get(language, 0.0) / total
            cumulative.append((language, running))
        selected = []
        for key, positions in ordered_groups:
            draw = _stable_unit_interval(seed, ":".join(key))
            language = next(
                candidate for candidate, boundary in cumulative if draw <= boundary
            )
            by_language = {
                str(index[position]["language"]).lower(): position
                for position in positions
            }
            if language not in by_language:
                raise ValueError(f"canonical group is missing weighted language {language}")
            selected.append(by_language[language])
        return sorted(selected)
    raise ValueError(f"unknown virtual corpus policy: {policy}")


def _mean(x: Tensor, positions: list[int]) -> Tensor:
    if not positions:
        raise ValueError("virtual corpus removed every row in a required group")
    return x[positions].mean(dim=0, dtype=torch.float64)


def _direction(x: Tensor, index: list[dict[str, object]], positions: list[int]) -> Tensor:
    safe = [i for i in positions if str(index[i]["direction_class"]).lower() == "safe"]
    unsafe = [
        i for i in positions if str(index[i]["direction_class"]).lower() == "unsafe"
    ]
    return _mean(x, unsafe) - _mean(x, safe)


def _cosine(left: Tensor, right: Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(F.cosine_similarity(left, right, dim=0).clamp(-1, 1))


def _group_energy(
    x: Tensor,
    keys: list[tuple[str, ...]],
    reference_means: dict[tuple[str, ...], Tensor] | None = None,
    reference_key: callable | None = None,
) -> tuple[float, dict[tuple[str, ...], Tensor]]:
    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        positions[key].append(index)
    grand = x.mean(dim=0, dtype=torch.float64)
    means = {key: _mean(x, values) for key, values in positions.items()}
    energy = 0.0
    for key, values in positions.items():
        if reference_means is None:
            reference = grand
        else:
            assert reference_key is not None
            reference = reference_means[reference_key(key)]
        delta = means[key] - reference
        energy += len(values) * float(torch.sum(delta * delta))
    return energy, means


def _layer_factor_statistics(
    x: Tensor,
    directions: list[str],
    languages: list[str],
    categories: list[str],
) -> dict[str, float]:
    grand = x.mean(dim=0, dtype=torch.float64)
    total = float(torch.sum((x.to(torch.float64) - grand) ** 2))
    denominator = max(total, torch.finfo(torch.float64).eps)

    direction_keys = [(value,) for value in directions]
    language_keys = [(value,) for value in languages]
    category_keys = list(zip(directions, categories))
    direction_language_keys = list(zip(directions, languages))
    category_language_keys = list(zip(directions, categories, languages))

    direction_energy, direction_means = _group_energy(x, direction_keys)
    language_energy, _ = _group_energy(x, language_keys)
    category_energy, category_means = _group_energy(
        x,
        category_keys,
        reference_means=direction_means,
        reference_key=lambda key: (key[0],),
    )

    dl_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, key in enumerate(direction_language_keys):
        dl_positions[key].append(position)
    language_positions: dict[str, list[int]] = defaultdict(list)
    for position, language in enumerate(languages):
        language_positions[language].append(position)
    language_means = {
        language: _mean(x, values) for language, values in language_positions.items()
    }
    dl_energy = 0.0
    for key, values in dl_positions.items():
        cell = _mean(x, values)
        delta = (
            cell
            - direction_means[(key[0],)]
            - language_means[key[1]]
            + grand
        )
        dl_energy += len(values) * float(torch.sum(delta * delta))

    dcl_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for position, key in enumerate(category_language_keys):
        dcl_positions[key].append(position)
    dl_means = {key: _mean(x, values) for key, values in dl_positions.items()}
    dcl_energy = 0.0
    for key, values in dcl_positions.items():
        cell = _mean(x, values)
        delta = (
            cell
            - category_means[(key[0], key[1])]
            - dl_means[(key[0], key[2])]
            + direction_means[(key[0],)]
        )
        dcl_energy += len(values) * float(torch.sum(delta * delta))

    return {
        "total_energy": total,
        "direction_energy_ratio": direction_energy / denominator,
        "language_energy_ratio": language_energy / denominator,
        "category_nested_energy_ratio": category_energy / denominator,
        "direction_language_energy_ratio": dl_energy / denominator,
        "language_category_nested_energy_ratio": dcl_energy / denominator,
    }


def _cohesion(x: Tensor, positions: list[int]) -> float:
    center = _mean(x, positions)
    values = []
    for position in positions:
        values.append(_cosine(x[position].to(torch.float64), center))
    return sum(values) / len(values)


def _temperature(
    index: list[dict[str, object]], residuals: Tensor
) -> tuple[dict[str, object], list[dict[str, float]]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(index):
        groups[(str(row["direction_class"]), str(row["canonical_id"]))].append(
            position
        )
    cold_total = 0
    observations = 0
    layer_reports = []
    eps = torch.finfo(torch.float64).eps
    for layer in range(residuals.shape[1]):
        x = residuals[:, layer, :].to(torch.float64)
        distances = []
        for positions in groups.values():
            values = x[positions]
            center = torch.quantile(values, 0.5, dim=0)
            distances.extend(
                float(value)
                for value in torch.linalg.vector_norm(values - center, dim=1)
                / math.sqrt(values.shape[1])
            )
        distance_tensor = torch.tensor(distances, dtype=torch.float64)
        median = float(torch.quantile(distance_tensor, 0.5))
        mad = float(torch.quantile(torch.abs(distance_tensor - median), 0.5))
        positive = distance_tensor[distance_tensor > eps]
        positive_floor = (
            float(torch.quantile(positive, 0.25)) * 0.25 if len(positive) else 0.0
        )
        bandwidth = max(median + 1.4826 * mad, positive_floor, 1e-12)
        heat = torch.exp(-(distance_tensor**2) / (2 * bandwidth**2))
        cold = int(torch.sum(heat < 0.25))
        cold_total += cold
        observations += len(distances)
        layer_reports.append(
            {
                "layer": layer,
                "bandwidth": bandwidth,
                "mean_heat": float(torch.mean(heat)),
                "cold_rate": cold / len(distances),
            }
        )
    return (
        {
            "observations": observations,
            "cold_observations": cold_total,
            "discarded_observations": 0,
            "cold_threshold": 0.25,
            "kernel": "rbf_robust_bandwidth_v1",
        },
        layer_reports,
    )


def _policy_comparison(
    index: list[dict[str, object]], residuals: Tensor, policy: str, seed: int
) -> dict[str, object]:
    full_positions = list(range(len(index)))
    selected = virtual_policy_indices(index, policy, seed=seed)
    losses = []
    norm_errors = []
    layer_cosines = []
    for layer in range(residuals.shape[1]):
        x = residuals[:, layer, :]
        full_direction = _direction(x, index, full_positions)
        selected_direction = _direction(x, index, selected)
        cosine = _cosine(full_direction, selected_direction)
        full_norm = float(torch.linalg.vector_norm(full_direction))
        selected_norm = float(torch.linalg.vector_norm(selected_direction))
        norm_error = (
            abs(selected_norm - full_norm) / full_norm
            if full_norm > 0
            else float(selected_norm > 0)
        )
        losses.append(max(0.0, 1.0 - cosine))
        norm_errors.append(norm_error)
        layer_cosines.append(cosine)
    return {
        "policy": policy,
        "rows": len(selected),
        "direction_loss": sum(losses) / len(losses),
        "direction_norm_error": sum(norm_errors) / len(norm_errors),
        "layer_cosine_min": min(layer_cosines),
        "translated_duplicates": len(selected)
        - len(
            {
                (
                    str(index[position]["direction_class"]),
                    str(index[position]["canonical_id"]),
                )
                for position in selected
            }
        ),
    }


def _robust_heat(values: Tensor) -> Tensor:
    values = torch.abs(values.to(torch.float64))
    positive = values[values > 0]
    if not len(positive):
        return torch.zeros_like(values)
    scale = torch.quantile(positive, 0.75).clamp_min(
        torch.finfo(torch.float64).eps
    )
    return torch.clamp(values / scale, min=0.0, max=1.0)


def _geometric_membership(*values: Tensor) -> Tensor:
    stacked = torch.stack([value.clamp(0.0, 1.0) for value in values])
    return torch.prod(stacked, dim=0).pow(1.0 / len(values))


def _top_components(scores: Tensor, limit: int) -> list[dict[str, float | int]]:
    positive = torch.nonzero(scores > 0, as_tuple=False).flatten()
    if not len(positive):
        return []
    count = min(limit, len(positive))
    values, offsets = torch.topk(scores[positive], count)
    components = positive[offsets]
    return [
        {"component": int(component), "score": float(score)}
        for component, score in zip(components, values)
    ]


def _component_regions(
    index: list[dict[str, object]], residuals: Tensor
) -> dict[str, object]:
    """Build neutral component-level fuzzy regions and their intersections."""

    directions = _metadata(index, "direction_class")
    languages = _metadata(index, "language")
    categories = _metadata(index, "category_id")
    unique_languages = sorted(set(languages))
    direction_positions = {
        direction: [i for i, value in enumerate(directions) if value == direction]
        for direction in ("safe", "unsafe")
    }
    language_direction_positions = {
        (direction, language): [
            i
            for i, (row_direction, row_language) in enumerate(
                zip(directions, languages)
            )
            if row_direction == direction and row_language == language
        ]
        for direction in ("safe", "unsafe")
        for language in unique_languages
    }
    category_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, (direction, category) in enumerate(zip(directions, categories)):
        category_positions[(direction, category)].append(position)

    top_k = max(8, min(64, math.ceil(residuals.shape[2] * 0.01)))
    layer_reports = []
    for layer in range(residuals.shape[1]):
        x = residuals[:, layer, :].to(torch.float64)
        direction_means = {
            direction: _mean(x, positions)
            for direction, positions in direction_positions.items()
        }
        contrast = direction_means["unsafe"] - direction_means["safe"]
        cell_means = {
            key: _mean(x, positions)
            for key, positions in language_direction_positions.items()
        }
        language_effects = torch.stack(
            [
                torch.stack(
                    [
                        cell_means[(direction, language)]
                        - direction_means[direction]
                        for direction in ("safe", "unsafe")
                    ]
                ).mean(dim=0)
                for language in unique_languages
            ]
        )
        language_contrasts = torch.stack(
            [
                cell_means[("unsafe", language)]
                - cell_means[("safe", language)]
                for language in unique_languages
            ]
        )
        interaction_effects = language_contrasts - contrast
        category_effects = torch.stack(
            [
                _mean(x, positions) - direction_means[key[0]]
                for key, positions in sorted(category_positions.items())
            ]
        )

        contrast_heat = _robust_heat(contrast)
        language_heat = _robust_heat(
            torch.sqrt(torch.mean(language_effects**2, dim=0))
        )
        interaction_heat = _robust_heat(
            torch.sqrt(torch.mean(interaction_effects**2, dim=0))
        )
        category_heat = _robust_heat(
            torch.sqrt(torch.mean(category_effects**2, dim=0))
        )
        global_sign = torch.sign(contrast)
        cross_language_alignment = torch.mean(
            (torch.sign(language_contrasts) == global_sign).to(torch.float64),
            dim=0,
        ) * (global_sign != 0).to(torch.float64)

        scores = {
            "shared_contrast": _geometric_membership(
                contrast_heat,
                cross_language_alignment,
                1.0 - language_heat,
                1.0 - interaction_heat,
                1.0 - category_heat,
            ),
            "language_core": _geometric_membership(
                language_heat,
                1.0 - interaction_heat,
                1.0 - contrast_heat,
            ),
            "language_conditioned_contrast": _geometric_membership(
                contrast_heat, interaction_heat
            ),
            "category_conditioned_contrast": _geometric_membership(
                contrast_heat, category_heat
            ),
            "ambiguous_overlap": _geometric_membership(
                contrast_heat,
                torch.maximum(
                    language_heat, torch.maximum(interaction_heat, category_heat)
                ),
            ),
        }
        regions = {
            name: _top_components(score, top_k) for name, score in scores.items()
        }
        component_sets = {
            name: {int(value["component"]) for value in values}
            for name, values in regions.items()
        }
        overlaps = []
        region_names = list(regions)
        for left_index, left in enumerate(region_names):
            for right in region_names[left_index + 1 :]:
                union = component_sets[left] | component_sets[right]
                intersection = component_sets[left] & component_sets[right]
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "intersection": len(intersection),
                        "union": len(union),
                        "jaccard": len(intersection) / len(union) if union else 0.0,
                    }
                )
        layer_reports.append(
            {
                "layer": layer,
                "top_k": top_k,
                "regions": regions,
                "overlaps": overlaps,
            }
        )
    return {
        "schema_version": 1,
        "method": "robust_fuzzy_component_regions_v1",
        "associative_not_causal": True,
        "layers": layer_reports,
    }


def analyze_geometry(
    index: list[dict[str, object]], residuals: Tensor, seed: int = 42
) -> dict[str, Any]:
    """Calculate text-free maps and virtual-corpus comparisons from one cache."""

    _validate(index, residuals)
    residuals = residuals.detach().to(device="cpu", dtype=torch.float32)
    directions = _metadata(index, "direction_class")
    languages = _metadata(index, "language")
    categories = _metadata(index, "category_id")
    unique_languages = sorted(set(languages))

    temperature, layer_temperatures = _temperature(index, residuals)
    layer_statistics = []
    factor_layers = []
    safe_positions = [i for i, value in enumerate(directions) if value == "safe"]
    unsafe_positions = [i for i, value in enumerate(directions) if value == "unsafe"]
    for layer in range(residuals.shape[1]):
        x = residuals[:, layer, :]
        safe_mean = _mean(x, safe_positions)
        unsafe_mean = _mean(x, unsafe_positions)
        direction_separation = max(0.0, 1.0 - _cosine(safe_mean, unsafe_mean))
        factors = _layer_factor_statistics(x, directions, languages, categories)
        group_a_cohesion = _cohesion(x, safe_positions)
        group_b_cohesion = _cohesion(x, unsafe_positions)
        contrast_heat = (
            direction_separation
            * max(0.0, group_b_cohesion)
            * max(0.0, 1.0 - min(1.0, factors["language_energy_ratio"]))
        )
        layer_statistics.append(
            {
                "layer": layer,
                "group_a_cohesion": group_a_cohesion,
                "group_b_cohesion": group_b_cohesion,
                "direction_separation": direction_separation,
                "contrast_heat": contrast_heat,
                **layer_temperatures[layer],
            }
        )
        factor_layers.append({"layer": layer, **factors})

    language_contributions = {}
    for language in unique_languages:
        comparison = _policy_comparison(
            index, residuals, f"leave_out:{language}", seed
        )
        language_contributions[language] = {
            "direction_loss": comparison["direction_loss"],
            "direction_norm_error": comparison["direction_norm_error"],
            "marginal_rows": sum(value == language for value in languages),
        }

    policy_names = ["full", "cycle_languages"]
    if "en" in unique_languages:
        policy_names.append("en_only")
    policy_names.extend(f"leave_out:{language}" for language in unique_languages)
    if len(unique_languages) > 1:
        equal_weights = {
            language: 1.0 / len(unique_languages) for language in unique_languages
        }
        policy_names.append(
            "weighted:"
            + json.dumps(equal_weights, sort_keys=True, separators=(",", ":"))
        )
        remaining_weight = 0.3 / (len(unique_languages) - 1)
        for dominant in unique_languages:
            weights = {
                language: 0.7 if language == dominant else remaining_weight
                for language in unique_languages
            }
            policy_names.append(
                "weighted:"
                + json.dumps(weights, sort_keys=True, separators=(",", ":"))
            )
    policy_names.extend(
        f"fraction:{fraction}:cycle_languages"
        for fraction in (0.75, 0.5, 0.25)
    )
    subset_candidates = [
        _policy_comparison(index, residuals, policy, seed) for policy in policy_names
    ]
    component_regions = _component_regions(index, residuals)

    return {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(index),
        "layers": int(residuals.shape[1]),
        "hidden_size": int(residuals.shape[2]),
        "languages": unique_languages,
        "measurement_reused_only": True,
        "temperature": temperature,
        "layer_statistics": layer_statistics,
        "factor_map": {
            "category_is_nested_in_direction": True,
            "layers": factor_layers,
        },
        "component_regions": component_regions,
        "language_contributions": language_contributions,
        "subset_candidates": subset_candidates,
    }


def _assert_text_free(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _SENSITIVE_KEYS:
                raise ValueError(f"sensitive key in report: {key}")
            _assert_text_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_text_free(child)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_geometry_reports(report: dict[str, Any], output_dir: Path) -> None:
    """Write aggregate machine and human reports without corpus text."""

    _assert_text_free(report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "component_regions.json": report["component_regions"],
        "layer_statistics.json": {
            "schema_version": report["schema_version"],
            "layers": report["layer_statistics"],
            "temperature": report["temperature"],
        },
        "factor_map.json": report["factor_map"],
        "language_contributions.json": report["language_contributions"],
        "subset_candidates.json": report["subset_candidates"],
    }
    for name, value in documents.items():
        _write_json(output_dir / name, value)

    summary = {
        "status": report["status"],
        "rows": report["rows"],
        "layers": report["layers"],
        "hidden_size": report["hidden_size"],
        "languages": report["languages"],
        "temperature": report["temperature"],
        "language_contributions": report["language_contributions"],
        "subset_candidates": report["subset_candidates"],
    }
    embedded = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / "report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Heretic-MOE language geometry map</title></head><body>"
        "<h1>Heretic-MOE language geometry map</h1><pre>"
        + embedded
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
