# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure objective math for multilingual Heretic-MOE trials."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MultilingualTrialMetrics:
    srg_gain: float
    r_gain: float
    unsafe_geometry_gain: float
    safe_ppl_drift: float
    safe_ppl_signed_change: float
    safe_geometry_drift: float
    language_instability: float
    category_instability: float
    removal: float
    preservation_loss: float
    cost_up: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_nonnegative(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def compose_trial_metrics(
    *,
    srg_gain: float,
    r_gain: float,
    unsafe_geometry_gain: float,
    safe_ppl_drift: float,
    safe_ppl_signed_change: float,
    safe_geometry_drift: float,
    language_instability: float,
    category_instability: float,
) -> MultilingualTrialMetrics:
    """Compose the two Optuna objectives and the human-facing ``Cost↑`` value.

    ``safe_ppl_signed_change`` is retained only for diagnostics. Preservation uses
    symmetric nonnegative drift so a lower candidate PPL cannot become a bonus.
    """

    srg = _require_finite("srg_gain", srg_gain)
    r_side = _require_finite("r_gain", r_gain)
    unsafe_geometry = _require_finite(
        "unsafe_geometry_gain", unsafe_geometry_gain
    )
    ppl_drift = _require_nonnegative("safe_ppl_drift", safe_ppl_drift)
    ppl_signed = _require_finite(
        "safe_ppl_signed_change", safe_ppl_signed_change
    )
    safe_geometry = _require_nonnegative(
        "safe_geometry_drift", safe_geometry_drift
    )
    language = _require_nonnegative("language_instability", language_instability)
    category = _require_nonnegative("category_instability", category_instability)

    removal = 0.50 * srg + 0.25 * r_side + 0.25 * unsafe_geometry
    preservation_loss = (
        0.40 * ppl_drift
        + 0.35 * safe_geometry
        + 0.15 * language
        + 0.10 * category
    )
    cost_up = _sigmoid(4.0 * removal) / (1.0 + preservation_loss)
    return MultilingualTrialMetrics(
        srg_gain=srg,
        r_gain=r_side,
        unsafe_geometry_gain=unsafe_geometry,
        safe_ppl_drift=ppl_drift,
        safe_ppl_signed_change=ppl_signed,
        safe_geometry_drift=safe_geometry,
        language_instability=language,
        category_instability=category,
        removal=removal,
        preservation_loss=preservation_loss,
        cost_up=cost_up,
    )
