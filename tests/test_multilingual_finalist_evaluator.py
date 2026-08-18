from __future__ import annotations

from heretic.multilingual_finalist_evaluator import MultilingualFinalistEvaluator


def test_finalist_evaluator_reuses_one_fixed_panel_for_every_candidate() -> None:
    calls = []

    class FrozenRuntime:
        expected_per_direction = 200
        expected_languages = ("en", "ru", "zh", "ja")

        def evaluate(
            self,
            schedule_trial_number,
            *,
            artifact_trial_number=None,
            residual_capture=None,
        ):
            calls.append((schedule_trial_number, artifact_trial_number, residual_capture))
            return {"artifact_trial_number": artifact_trial_number}

    runtime = FrozenRuntime()
    evaluator = MultilingualFinalistEvaluator(
        runtime=runtime,
        fixed_schedule_trial_number=0,
    )

    first = evaluator.evaluate(7, artifact_trial_number=42)
    second = evaluator.evaluate(8, artifact_trial_number=43)

    assert first == {"artifact_trial_number": 42}
    assert second == {"artifact_trial_number": 43}
    assert calls == [(0, 42, None), (0, 43, None)]
    assert evaluator.expected_per_direction == 200
    assert evaluator.expected_languages == ("en", "ru", "zh", "ja")
