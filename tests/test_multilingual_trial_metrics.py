from __future__ import annotations

import math

import pytest

from heretic.multilingual_trial_metrics import compose_trial_metrics


def test_clean_baseline_has_zero_removal_and_loss_and_neutral_cost() -> None:
    metrics = compose_trial_metrics(
        srg_gain=0.0,
        r_gain=0.0,
        unsafe_geometry_gain=0.0,
        safe_ppl_drift=0.0,
        safe_ppl_signed_change=0.0,
        safe_geometry_drift=0.0,
        language_instability=0.0,
        category_instability=0.0,
    )

    assert metrics.removal == 0.0
    assert metrics.preservation_loss == 0.0
    assert metrics.cost_up == pytest.approx(0.5)


def test_cost_increases_with_removal_and_decreases_with_preservation_loss() -> None:
    baseline = compose_trial_metrics(
        srg_gain=0.3,
        r_gain=0.2,
        unsafe_geometry_gain=0.1,
        safe_ppl_drift=0.02,
        safe_ppl_signed_change=0.02,
        safe_geometry_drift=0.03,
        language_instability=0.04,
        category_instability=0.05,
    )
    more_removal = compose_trial_metrics(
        srg_gain=0.5,
        r_gain=0.2,
        unsafe_geometry_gain=0.1,
        safe_ppl_drift=0.02,
        safe_ppl_signed_change=0.02,
        safe_geometry_drift=0.03,
        language_instability=0.04,
        category_instability=0.05,
    )
    more_damage = compose_trial_metrics(
        srg_gain=0.3,
        r_gain=0.2,
        unsafe_geometry_gain=0.1,
        safe_ppl_drift=0.20,
        safe_ppl_signed_change=0.20,
        safe_geometry_drift=0.30,
        language_instability=0.40,
        category_instability=0.50,
    )

    assert more_removal.removal > baseline.removal
    assert more_removal.cost_up > baseline.cost_up
    assert more_damage.preservation_loss > baseline.preservation_loss
    assert more_damage.cost_up < baseline.cost_up


def test_negative_signed_ppl_change_never_becomes_a_preservation_bonus() -> None:
    positive = compose_trial_metrics(
        srg_gain=0.2,
        r_gain=0.1,
        unsafe_geometry_gain=0.1,
        safe_ppl_drift=0.07,
        safe_ppl_signed_change=0.07,
        safe_geometry_drift=0.0,
        language_instability=0.0,
        category_instability=0.0,
    )
    negative = compose_trial_metrics(
        srg_gain=0.2,
        r_gain=0.1,
        unsafe_geometry_gain=0.1,
        safe_ppl_drift=0.07,
        safe_ppl_signed_change=-0.07,
        safe_geometry_drift=0.0,
        language_instability=0.0,
        category_instability=0.0,
    )

    assert negative.preservation_loss == positive.preservation_loss
    assert negative.cost_up == positive.cost_up
    assert negative.safe_ppl_signed_change == -0.07


@pytest.mark.parametrize(
    "field,value",
    [
        ("safe_ppl_drift", -0.01),
        ("safe_geometry_drift", -0.01),
        ("language_instability", -0.01),
        ("category_instability", -0.01),
        ("srg_gain", math.inf),
    ],
)
def test_invalid_metric_inputs_are_rejected(field: str, value: float) -> None:
    kwargs = {
        "srg_gain": 0.0,
        "r_gain": 0.0,
        "unsafe_geometry_gain": 0.0,
        "safe_ppl_drift": 0.0,
        "safe_ppl_signed_change": 0.0,
        "safe_geometry_drift": 0.0,
        "language_instability": 0.0,
        "category_instability": 0.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        compose_trial_metrics(**kwargs)
