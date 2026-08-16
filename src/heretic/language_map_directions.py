# SPDX-License-Identifier: AGPL-3.0-or-later

"""Frozen multilingual direction tensors derived from one residual map."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor


@dataclass(frozen=True)
class DirectionMapProfile:
    consensus_refusal_direction: Tensor
    per_layer_direction: Tensor
    language_subspace: Tensor
    language_ranks: Tensor
    language_explained_variance: Tensor
    category_branch_directions: Tensor
    category_ids: tuple[str, ...]
    layer_reliability: Tensor
    recommended_layer_bounds: tuple[int, int]
    diagnostics: dict[str, Any]


def _canonical_sign(vector: Tensor) -> Tensor:
    result = vector.clone()
    if bool(torch.any(result != 0)):
        pivot = int(torch.argmax(torch.abs(result)))
        if float(result[pivot]) < 0:
            result.neg_()
    return result


def _unit(vector: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= torch.finfo(torch.float64).eps:
        return torch.zeros_like(vector)
    return vector / norm


def _cosine(left: Tensor, right: Tensor) -> float:
    if float(torch.linalg.vector_norm(left)) == 0.0 or float(
        torch.linalg.vector_norm(right)
    ) == 0.0:
        return 0.0
    return float(F.cosine_similarity(left, right, dim=0).clamp(-1.0, 1.0))


def _median(values: Sequence[Tensor]) -> Tensor:
    if not values:
        raise ValueError("cannot calculate a robust center from an empty group")
    return torch.median(torch.stack(tuple(values)), dim=0).values


def _validate(
    index: list[dict[str, object]],
    residuals: Tensor,
    languages: tuple[str, ...],
) -> None:
    if residuals.ndim != 3 or len(index) != residuals.shape[0] or not index:
        raise ValueError("index/residuals must have aligned [rows,layers,hidden] data")
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("residuals contain non-finite values")
    required = {
        "canonical_id",
        "row_id",
        "language",
        "direction_class",
        "category_id",
    }
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in index:
        if not required.issubset(row):
            raise ValueError("row index is missing required metadata")
        direction = str(row["direction_class"]).lower()
        if direction not in {"safe", "unsafe"}:
            raise ValueError("direction_class must be safe or unsafe")
        groups[(direction, str(row["canonical_id"]))].append(row)
    expected = set(languages)
    for rows in groups.values():
        observed = [str(row["language"]).lower() for row in rows]
        if len(observed) != len(languages) or set(observed) != expected:
            raise ValueError("canonical group has missing aligned languages")
        if len({str(row["category_id"]) for row in rows}) != 1:
            raise ValueError("canonical group category mismatch")


def _language_basis(
    language_offsets: dict[str, list[Tensor]],
    languages: tuple[str, ...],
    explained_variance_target: float,
) -> tuple[Tensor, Tensor, int]:
    centroids = torch.stack(
        [_median(language_offsets[language]) for language in languages]
    )
    centroids = centroids - centroids.mean(dim=0)
    _, singular, vh = torch.linalg.svd(centroids, full_matrices=False)
    variance = singular.square()
    total = float(variance.sum())
    if total <= torch.finfo(torch.float64).eps:
        return (
            torch.zeros((0, centroids.shape[1]), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            0,
        )
    fractions = variance / variance.sum()
    cumulative = torch.cumsum(fractions, dim=0)
    rank = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(explained_variance_target, dtype=cumulative.dtype),
        )
    ) + 1
    rank = min(rank, len(languages) - 1, int(vh.shape[0]))
    basis = torch.stack([_canonical_sign(row) for row in vh[:rank]])
    return basis, fractions[:rank], rank


def _project(vector: Tensor, basis: Tensor) -> Tensor:
    if basis.shape[0] == 0:
        return torch.zeros_like(vector)
    return basis.T @ (basis @ vector)


def _macro_unsafe_center(
    centers: dict[tuple[str, str], Tensor],
    categories: dict[tuple[str, str], str],
) -> tuple[Tensor, dict[str, Tensor]]:
    by_category: dict[str, list[Tensor]] = defaultdict(list)
    for key, center in centers.items():
        if key[0] == "unsafe":
            by_category[categories[key]].append(center)
    category_centers = {
        category: _median(values) for category, values in sorted(by_category.items())
    }
    return torch.stack(tuple(category_centers.values())).mean(dim=0), category_centers


def _clean_direction(
    raw: Tensor,
    basis: Tensor,
    language_contrasts: list[Tensor],
) -> tuple[Tensor, float, float]:
    language_part = _project(raw, basis)
    raw_norm = float(torch.linalg.vector_norm(raw))
    raw_alignment = sum(
        max(0.0, _cosine(raw, contrast)) for contrast in language_contrasts
    ) / max(len(language_contrasts), 1)
    for removal_fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
        candidate = raw - removal_fraction * language_part
        norm_ratio = float(torch.linalg.vector_norm(candidate)) / max(raw_norm, 1e-12)
        alignment = sum(
            max(0.0, _cosine(candidate, contrast))
            for contrast in language_contrasts
        ) / max(len(language_contrasts), 1)
        if norm_ratio >= 0.20 and alignment + 0.05 >= raw_alignment:
            return candidate, removal_fraction, min(
                1.0,
                float(torch.linalg.vector_norm(language_part))
                / max(raw_norm, 1e-12),
            )
    return raw, 0.0, 0.0


def _longest_reliable_bounds(reliability: Tensor) -> tuple[int, int]:
    if reliability.numel() == 0:
        raise ValueError("layer reliability is empty")
    peak = float(reliability.max())
    if peak <= 0:
        strongest = int(torch.argmax(reliability))
        return strongest, strongest
    mask = reliability >= max(peak * 0.25, 1e-8)
    best_start = best_end = int(torch.argmax(reliability))
    start: int | None = None
    for index, active in enumerate(mask.tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index - 1
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = None
    return best_start, best_end


def build_direction_map_profile(
    index: list[dict[str, object]],
    residuals: Tensor,
    *,
    languages: Sequence[str] = ("en", "ru", "zh", "ja"),
    explained_variance_target: float = 0.90,
) -> DirectionMapProfile:
    """Build language-cleaned directions and deterministic search bounds."""

    normalized_languages = tuple(str(language).lower() for language in languages)
    if not 0.5 <= explained_variance_target < 1.0:
        raise ValueError("explained_variance_target must be in [0.5, 1.0)")
    _validate(index, residuals, normalized_languages)
    x = residuals.detach().to(device="cpu", dtype=torch.float64)
    rows_by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    categories: dict[tuple[str, str], str] = {}
    language_positions: dict[tuple[str, str, str], int] = {}
    for position, row in enumerate(index):
        direction = str(row["direction_class"]).lower()
        canonical_id = str(row["canonical_id"])
        key = (direction, canonical_id)
        rows_by_group[key].append(position)
        categories[key] = str(row["category_id"])
        language_positions[(direction, canonical_id, str(row["language"]).lower())] = position

    unsafe_category_ids = tuple(
        sorted({category for key, category in categories.items() if key[0] == "unsafe"})
    )
    layers, hidden = map(int, x.shape[1:])
    max_language_rank = len(normalized_languages) - 1
    consensus = torch.zeros((layers, hidden), dtype=torch.float64)
    language_subspace = torch.zeros(
        (layers, max_language_rank, hidden), dtype=torch.float64
    )
    explained = torch.zeros((layers, max_language_rank), dtype=torch.float64)
    ranks = torch.zeros((layers,), dtype=torch.int64)
    category_branches = torch.zeros(
        (layers, len(unsafe_category_ids), hidden), dtype=torch.float64
    )
    reliability = torch.zeros((layers,), dtype=torch.float64)
    layer_diagnostics: list[dict[str, float | int]] = []

    for layer in range(layers):
        centers = {
            key: _median([x[position, layer] for position in positions])
            for key, positions in rows_by_group.items()
        }
        language_offsets: dict[str, list[Tensor]] = defaultdict(list)
        for key, center in centers.items():
            for language in normalized_languages:
                position = language_positions[(key[0], key[1], language)]
                language_offsets[language].append(x[position, layer] - center)
        basis, layer_explained, rank = _language_basis(
            language_offsets,
            normalized_languages,
            explained_variance_target,
        )
        if rank:
            language_subspace[layer, :rank] = basis
            explained[layer, :rank] = layer_explained
        ranks[layer] = rank

        safe_centers = [center for key, center in centers.items() if key[0] == "safe"]
        safe_core = _median(safe_centers)
        unsafe_core, unsafe_category_centers = _macro_unsafe_center(
            centers, categories
        )
        raw = unsafe_core - safe_core
        language_contrasts = []
        for language in normalized_languages:
            safe_language = _median(
                [
                    x[language_positions[(key[0], key[1], language)], layer]
                    for key in centers
                    if key[0] == "safe"
                ]
            )
            unsafe_by_category: dict[str, list[Tensor]] = defaultdict(list)
            for key in centers:
                if key[0] == "unsafe":
                    unsafe_by_category[categories[key]].append(
                        x[language_positions[(key[0], key[1], language)], layer]
                    )
            unsafe_language = torch.stack(
                [_median(values) for _, values in sorted(unsafe_by_category.items())]
            ).mean(dim=0)
            language_contrasts.append(unsafe_language - safe_language)
        cleaned, removal_fraction, language_leakage = _clean_direction(
            raw, basis, language_contrasts
        )
        direction = _unit(cleaned)
        if float(torch.dot(direction, raw)) < 0:
            direction.neg_()
        consensus[layer] = direction

        cleaned_categories: list[Tensor] = []
        for category_index, category in enumerate(unsafe_category_ids):
            category_direction = unsafe_category_centers[category] - safe_core
            category_clean = category_direction - removal_fraction * _project(
                category_direction, basis
            )
            cleaned_categories.append(category_clean)
            common = torch.dot(category_clean, direction) * direction
            category_branches[layer, category_index] = _unit(
                category_clean - common
            )

        safe_scores = torch.tensor(
            [float(torch.dot(center, direction)) for center in safe_centers],
            dtype=torch.float64,
        )
        unsafe_scores = torch.tensor(
            [
                float(torch.dot(center, direction))
                for key, center in centers.items()
                if key[0] == "unsafe"
            ],
            dtype=torch.float64,
        )
        separation_delta = float(torch.median(unsafe_scores) - torch.median(safe_scores))
        separation_strength = max(0.0, 1.0 - math.exp(-abs(separation_delta)))
        threshold = float((torch.median(safe_scores) + torch.median(unsafe_scores)) / 2)
        overlap_error = 0.5 * (
            float(torch.mean((safe_scores >= threshold).to(torch.float64)))
            + float(torch.mean((unsafe_scores < threshold).to(torch.float64)))
        )
        cross_language_stability = sum(
            max(0.0, _cosine(direction, value)) for value in language_contrasts
        ) / len(language_contrasts)
        category_support = sum(
            max(0.0, _cosine(direction, value)) for value in cleaned_categories
        ) / max(len(cleaned_categories), 1)

        ordered_safe = sorted(key for key in centers if key[0] == "safe")
        ordered_unsafe = sorted(key for key in centers if key[0] == "unsafe")
        split_contrasts = []
        for parity in (0, 1):
            safe_half = [centers[key] for offset, key in enumerate(ordered_safe) if offset % 2 == parity]
            unsafe_half = [centers[key] for offset, key in enumerate(ordered_unsafe) if offset % 2 == parity]
            if safe_half and unsafe_half:
                split_contrasts.append(_median(unsafe_half) - _median(safe_half))
        bootstrap_stability = sum(
            max(0.0, _cosine(direction, value)) for value in split_contrasts
        ) / max(len(split_contrasts), 1)
        layer_reliability = (
            separation_strength
            * cross_language_stability
            * category_support
            * (1.0 - min(1.0, overlap_error))
            * (1.0 - min(1.0, language_leakage))
            * bootstrap_stability
        )
        reliability[layer] = min(1.0, max(0.0, layer_reliability))
        layer_diagnostics.append(
            {
                "layer": layer,
                "language_rank": rank,
                "language_removal_fraction": removal_fraction,
                "separation_strength": separation_strength,
                "cross_language_stability": cross_language_stability,
                "cross_category_support": category_support,
                "safe_overlap_penalty": overlap_error,
                "language_leakage": language_leakage,
                "bootstrap_stability": bootstrap_stability,
                "layer_reliability": float(reliability[layer]),
            }
        )

    bounds = _longest_reliable_bounds(reliability)
    return DirectionMapProfile(
        consensus_refusal_direction=consensus.float().contiguous(),
        per_layer_direction=(consensus * reliability[:, None]).float().contiguous(),
        language_subspace=language_subspace.float().contiguous(),
        language_ranks=ranks.contiguous(),
        language_explained_variance=explained.float().contiguous(),
        category_branch_directions=category_branches.float().contiguous(),
        category_ids=unsafe_category_ids,
        layer_reliability=reliability.float().contiguous(),
        recommended_layer_bounds=bounds,
        diagnostics={
            "schema_version": 1,
            "explained_variance_target": explained_variance_target,
            "layers": layer_diagnostics,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_direction_map_package(
    profile: DirectionMapProfile,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically write the frozen search-direction package."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        tensor_path = temporary / "directions.safetensors"
        save_file(
            {
                "consensus_refusal_direction": profile.consensus_refusal_direction,
                "per_layer_direction": profile.per_layer_direction,
                "language_subspace": profile.language_subspace,
                "language_ranks": profile.language_ranks,
                "language_explained_variance": profile.language_explained_variance,
                "category_branch_directions": profile.category_branch_directions,
                "layer_reliability": profile.layer_reliability,
            },
            str(tensor_path),
            metadata={"schema_version": "1"},
        )
        tensor_sha = _sha256(tensor_path)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "PASS",
            "layers": int(profile.consensus_refusal_direction.shape[0]),
            "hidden_size": int(profile.consensus_refusal_direction.shape[1]),
            "category_ids": list(profile.category_ids),
            "recommended_layer_bounds": list(profile.recommended_layer_bounds),
            "diagnostics": profile.diagnostics,
            "files": {
                "directions.safetensors": {
                    "bytes": tensor_path.stat().st_size,
                    "sha256": tensor_sha,
                }
            },
            "package_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "tensor_sha256": tensor_sha,
                        "category_ids": profile.category_ids,
                        "recommended_layer_bounds": profile.recommended_layer_bounds,
                        "diagnostics": profile.diagnostics,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
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


def load_direction_map_package(
    input_dir: str | Path,
) -> tuple[DirectionMapProfile, dict[str, Any]]:
    """Load a frozen direction package after verifying all public hashes."""

    source = Path(input_dir).resolve()
    manifest_path = source / "manifest.json"
    tensor_path = source / "directions.safetensors"
    if not manifest_path.is_file() or not tensor_path.is_file():
        raise FileNotFoundError("direction package is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        raise ValueError("direction package manifest is not PASS schema version 1")
    tensor_record = (manifest.get("files") or {}).get("directions.safetensors")
    if not isinstance(tensor_record, dict) or tensor_record.get("sha256") != _sha256(
        tensor_path
    ):
        raise ValueError("direction package tensor SHA-256 mismatch")
    package_payload = {
        "tensor_sha256": tensor_record["sha256"],
        "category_ids": tuple(manifest.get("category_ids") or ()),
        "recommended_layer_bounds": tuple(
            manifest.get("recommended_layer_bounds") or ()
        ),
        "diagnostics": manifest.get("diagnostics"),
    }
    package_sha = hashlib.sha256(
        json.dumps(
            package_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("package_sha256") != package_sha:
        raise ValueError("direction package manifest SHA-256 mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    required = {
        "consensus_refusal_direction",
        "per_layer_direction",
        "language_subspace",
        "language_ranks",
        "language_explained_variance",
        "category_branch_directions",
        "layer_reliability",
    }
    if set(tensors) != required:
        raise ValueError("direction package tensor set mismatch")
    consensus = tensors["consensus_refusal_direction"]
    if (
        consensus.ndim != 2
        or consensus.shape
        != (int(manifest["layers"]), int(manifest["hidden_size"]))
        or tensors["layer_reliability"].shape != (consensus.shape[0],)
        or not all(bool(torch.isfinite(value).all()) for value in tensors.values())
    ):
        raise ValueError("direction package tensor shape or finiteness mismatch")
    bounds = tuple(int(value) for value in manifest["recommended_layer_bounds"])
    if len(bounds) != 2:
        raise ValueError("direction package layer bounds are invalid")
    profile = DirectionMapProfile(
        consensus_refusal_direction=consensus,
        per_layer_direction=tensors["per_layer_direction"],
        language_subspace=tensors["language_subspace"],
        language_ranks=tensors["language_ranks"],
        language_explained_variance=tensors["language_explained_variance"],
        category_branch_directions=tensors["category_branch_directions"],
        category_ids=tuple(str(value) for value in manifest["category_ids"]),
        layer_reliability=tensors["layer_reliability"],
        recommended_layer_bounds=(bounds[0], bounds[1]),
        diagnostics=dict(manifest["diagnostics"]),
    )
    return profile, manifest
