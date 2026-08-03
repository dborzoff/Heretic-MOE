#!/usr/bin/env python3
"""Algebraic dry-run for localized ablation/inversion mechanics.

This uses deterministic synthetic tensors, not prompts or model outputs. It
checks identities that must hold for any real output-projection matrix.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def project_output(weight: torch.Tensor, direction: torch.Tensor, strength: float) -> torch.Tensor:
    direction = F.normalize(direction, dim=0)
    return weight - strength * torch.outer(direction, direction @ weight)


def full_row_normalized_update(
    weight: torch.Tensor,
    direction: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Exact pre-SVD version of Heretic's RowNormalization.FULL path."""
    direction = F.normalize(direction, dim=0)
    row_norms = torch.linalg.vector_norm(weight, dim=1, keepdim=True)
    normalized = F.normalize(weight, dim=1)
    adjusted = project_output(normalized, direction, strength)
    return F.normalize(adjusted, dim=1) * row_norms


def rms_norm(value: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denominator = torch.sqrt(value.square().mean() + eps)
    return scale * value / denominator


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def smooth_window(
    layer: float,
    center: float,
    inner_radius: float,
    outer_radius: float,
    amplitude: float,
) -> float:
    """Flat center with compact cubic-Bezier/Hermite edges."""
    distance = abs(layer - center)
    if distance <= inner_radius:
        return amplitude
    if distance >= outer_radius:
        return 0.0
    progress = (outer_radius - distance) / (outer_radius - inner_radius)
    return amplitude * smoothstep(progress)


def main() -> None:
    torch.manual_seed(1729)
    dtype = torch.float64
    out_features = 17
    in_features = 23

    weight = torch.randn(out_features, in_features, dtype=dtype)
    direction = F.normalize(torch.randn(out_features, dtype=dtype), dim=0)
    inputs = torch.randn(in_features, dtype=dtype)

    removed = project_output(weight, direction, 1.0)
    inverted = project_output(weight, direction, 2.0)
    full_inverted = full_row_normalized_update(weight, direction, 2.0)

    original_projection = direction @ (weight @ inputs)
    removed_projection = direction @ (removed @ inputs)
    inverted_projection = direction @ (inverted @ inputs)
    full_inverted_projection = direction @ (full_inverted @ inputs)

    scale = torch.exp(torch.linspace(-1.0, 1.0, out_features, dtype=dtype))
    norm_aware_direction = F.normalize(scale * direction, dim=0)
    norm_aware_removed = project_output(weight, norm_aware_direction, 1.0)
    norm_aware_inverted = project_output(weight, norm_aware_direction, 2.0)
    naive_removed = project_output(weight, direction, 1.0)

    normalized_original_projection = direction @ rms_norm(weight @ inputs, scale)
    normalized_removed_projection = direction @ rms_norm(norm_aware_removed @ inputs, scale)
    normalized_inverted_projection = direction @ rms_norm(norm_aware_inverted @ inputs, scale)
    naive_normalized_projection = direction @ rms_norm(naive_removed @ inputs, scale)

    schedule = [
        smooth_window(
            layer=layer,
            center=12.0,
            inner_radius=2.0,
            outer_radius=6.0,
            amplitude=2.0,
        )
        for layer in range(25)
    ]

    result = {
        "raw_projector": {
            "removal_absolute_error": abs(removed_projection).item(),
            "inversion_absolute_error": abs(inverted_projection + original_projection).item(),
            "inversion_frobenius_norm_error": abs(
                torch.linalg.matrix_norm(inverted) - torch.linalg.matrix_norm(weight)
            ).item(),
        },
        "current_full_row_normalization": {
            "lambda_2_sign_flip_error_before_svd": abs(
                full_inverted_projection + original_projection
            ).item(),
            "meaning": "lambda=2 is no longer an exact reflection even before rank truncation",
        },
        "post_rms_norm": {
            "norm_aware_removal_absolute_error": abs(normalized_removed_projection).item(),
            "norm_aware_inversion_absolute_error": abs(
                normalized_inverted_projection + normalized_original_projection
            ).item(),
            "naive_removal_residual": abs(naive_normalized_projection).item(),
            "required_pre_norm_direction": "normalize(rms_scale * residual_direction)",
        },
        "smooth_compact_window": {
            "parameters": {
                "center": 12.0,
                "inner_radius": 2.0,
                "outer_radius": 6.0,
                "amplitude": 2.0,
            },
            "weights_by_layer": schedule,
            "outside_support_is_zero": all(
                value == 0.0
                for layer, value in enumerate(schedule)
                if abs(layer - 12.0) >= 6.0
            ),
            "plateau_is_exact": all(
                value == 2.0
                for layer, value in enumerate(schedule)
                if abs(layer - 12.0) <= 2.0
            ),
        },
    }
    print(json.dumps(result, indent=2))

    assert result["raw_projector"]["removal_absolute_error"] < 1e-12
    assert result["raw_projector"]["inversion_absolute_error"] < 1e-12
    assert result["raw_projector"]["inversion_frobenius_norm_error"] < 1e-12
    assert result["post_rms_norm"]["norm_aware_removal_absolute_error"] < 1e-12
    assert result["post_rms_norm"]["norm_aware_inversion_absolute_error"] < 1e-12
    assert result["current_full_row_normalization"]["lambda_2_sign_flip_error_before_svd"] > 1e-6
    assert result["post_rms_norm"]["naive_removal_residual"] > 1e-6
    assert result["smooth_compact_window"]["outside_support_is_zero"]
    assert result["smooth_compact_window"]["plateau_is_exact"]


if __name__ == "__main__":
    main()
