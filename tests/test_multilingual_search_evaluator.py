from __future__ import annotations

import pytest
from optuna.study import StudyDirection

from heretic.multilingual_search_evaluator import (
    MultilingualConstraintContract,
    MultilingualSearchEvaluator,
)
from heretic.multilingual_trial_evaluator import TrialMeasurement
from heretic.multilingual_trial_metrics import compose_trial_metrics


class _FrozenRuntime:
    def __init__(self, measurement: TrialMeasurement) -> None:
        self.measurement = measurement
        self.calls: list[tuple[int, int | None]] = []

    def evaluate(
        self,
        trial_number: int,
        *,
        artifact_trial_number: int | None = None,
        residual_capture=None,
    ) -> TrialMeasurement:
        del residual_capture
        self.calls.append((trial_number, artifact_trial_number))
        return self.measurement


def _measurement() -> TrialMeasurement:
    metrics = compose_trial_metrics(
        srg_gain=0.4,
        r_gain=0.2,
        unsafe_geometry_gain=0.3,
        safe_ppl_drift=0.01,
        safe_ppl_signed_change=-0.005,
        safe_geometry_drift=0.2,
        language_instability=0.1,
        category_instability=0.05,
    )
    return TrialMeasurement(
        trial_number=17,
        rows=800,
        safe_rows=400,
        unsafe_rows=400,
        metrics=metrics,
        diagnostics={
            "coverage": {"rows": 800},
            "hard_gates": {
                "empty_response_rate": 0.0,
                "truncated_response_rate": 0.01,
                "safe_d_to_r_rate": 0.02,
            },
        },
        private_records_sha256="a" * 64,
    )


def test_adapter_exposes_two_objectives_cost_and_frozen_constraints() -> None:
    runtime = _FrozenRuntime(_measurement())
    evaluator = MultilingualSearchEvaluator(
        runtime,
        constraints=MultilingualConstraintContract(
            max_safe_ppl_drift=0.02,
            max_safe_geometry_damage=0.25,
            max_language_instability=0.15,
            max_category_instability=0.10,
            max_empty_response_rate=0.01,
            max_truncated_response_rate=0.05,
            max_safe_d_to_r_rate=0.03,
        ),
    )

    scores = evaluator.get_scores(response_archive_id=17)

    assert runtime.calls == [(17, 17)]
    assert [name for name, _ in scores] == [
        "Removal",
        "Preservation loss",
        "Cost↑",
    ]
    assert evaluator.get_objective_names() == ["Removal", "Preservation loss"]
    assert evaluator.get_objective_directions() == [
        StudyDirection.MAXIMIZE,
        StudyDirection.MINIMIZE,
    ]
    assert evaluator.get_objective_values(scores) == (
        _measurement().metrics.removal,
        _measurement().metrics.preservation_loss,
    )
    assert evaluator.get_constraint_names() == [
        "Safe PPL drift <= 0.02",
        "Safe geometry damage <= 0.25",
        "Language instability <= 0.15",
        "Category instability <= 0.1",
        "Empty response rate <= 0.01",
        "Truncated response rate <= 0.05",
        "SAFE D->R rate <= 0.03",
    ]
    assert evaluator.get_constraint_values(scores) == pytest.approx(
        (-0.01, -0.05, -0.05, -0.05, -0.01, -0.04, -0.01)
    )
    records = evaluator.get_paired_score_records(scores)
    assert records[2]["score"]["value"] == _measurement().metrics.cost_up
    assert records[0]["score"]["diagnostics"]["private_records_sha256"] == "a" * 64


def test_adapter_requires_an_integer_global_trial_number() -> None:
    evaluator = MultilingualSearchEvaluator(
        _FrozenRuntime(_measurement()),
        constraints=MultilingualConstraintContract(),
    )

    try:
        evaluator.get_scores(response_archive_id="baseline")
    except ValueError as error:
        assert "global trial number" in str(error)
    else:
        raise AssertionError("non-integer archive IDs must be rejected")


def test_adapter_separates_schedule_and_artifact_trial_numbers() -> None:
    runtime = _FrozenRuntime(_measurement())
    evaluator = MultilingualSearchEvaluator(
        runtime,
        constraints=MultilingualConstraintContract(),
    )

    evaluator.get_scores(response_archive_id=604, schedule_trial_number=598)

    assert runtime.calls == [(598, 604)]


@pytest.mark.parametrize("invalid", [float("inf"), float("nan")])
def test_constraint_contract_rejects_non_finite_limits(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MultilingualConstraintContract(max_safe_ppl_drift=invalid)
