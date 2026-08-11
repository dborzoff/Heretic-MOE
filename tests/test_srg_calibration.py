from __future__ import annotations

import math

import pytest

from heretic.srg_calibration import build_profile, relative_score


def _result(index: int, margins: list[float]) -> dict[str, object]:
    return {
        "status": "PASS",
        "model_index": index,
        "model_id": f"model-{index}",
        "rows": len(margins),
        "prototype_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "diagnostics": {"margins": margins},
    }


def test_profile_uses_robust_per_row_scale_and_sign_consensus() -> None:
    profile = build_profile(
        [
            _result(0, [1.0, -1.0, 1.0]),
            _result(1, [2.0, -2.0, -1.0]),
            _result(2, [3.0, -1.0, 1.0]),
        ]
    )

    assert profile["status"] == "PASS"
    assert profile["model_count"] == 3
    assert profile["rows"] == 3
    assert profile["baseline_median"] == [2.0, -1.0, 1.0]
    assert profile["scale"][0] == pytest.approx(1.4826)
    assert profile["scale"][1:] == pytest.approx([0.37065, 0.37065])
    assert profile["sign_consensus"] == pytest.approx([1.0, 1.0, 2 / 3])
    assert sum(profile["weight"]) == pytest.approx(3.0)


def test_relative_score_is_zero_at_baseline_and_rewards_r_to_d_transition() -> None:
    profile = build_profile(
        [
            _result(0, [1.0, -1.0, 1.0]),
            _result(1, [2.0, -2.0, -1.0]),
            _result(2, [3.0, -1.0, 1.0]),
        ]
    )
    baseline = [1.0, -1.0, 0.5]

    unchanged = relative_score(baseline, baseline, profile)
    improved = relative_score(baseline, [-1.0, -1.0, 0.5], profile)

    assert unchanged["continuous_gain"] == 0.0
    assert unchanged["side_gain"] == 0.0
    assert unchanged["srg_gain"] == 0.0
    assert unchanged["r_gain"] == 0.0
    assert unchanged["unified_gain"] == 0.0
    assert improved["continuous_gain"] > 0.0
    assert improved["r_to_d_rate"] > 0.0
    assert improved["d_to_r_rate"] == 0.0
    assert improved["srg_gain"] == pytest.approx(math.tanh(improved["continuous_gain"]))
    assert improved["r_gain"] == improved["side_gain"]
    assert improved["unified_gain"] > 0.0


def test_profile_rejects_results_from_different_prompt_sets() -> None:
    left = _result(0, [0.0, 1.0])
    right = _result(1, [0.0, 1.0])
    right["prompt_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="prompt SHA-256"):
        build_profile([left, right])
