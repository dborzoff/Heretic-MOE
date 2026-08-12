from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from heretic.language_map_data import GeometryRow
from heretic.multilingual_contract import CalibrationRow
from heretic.multilingual_finalist_evaluator import MultilingualFinalistEvaluator


def _trial_rows(tmp_path: Path) -> list[GeometryRow]:
    rows = []
    for language in ("en", "ru"):
        for direction in ("safe", "unsafe"):
            for index in range(2):
                rows.append(GeometryRow(
                    canonical_id=f"{direction}-{index}", row_id=f"{language}-{direction}-{index}",
                    language=language, direction=direction, category_id=f"C0{index+1}",
                    prompt=f"private-trial-{language}-{direction}-{index}",
                    source_path=tmp_path / "private.jsonl", source_line=index + 1,
                ))
    return rows


def _final_rows(tmp_path: Path) -> list[CalibrationRow]:
    return [CalibrationRow(
        base_id=f"R{index+1}", row_id=f"{language}-R{index+1}", language=language,
        category_id=f"C0{index+1}", prompt=f"private-final-{language}-{index}",
        source_path=tmp_path / "private.jsonl", source_line=index + 1,
    ) for language in ("en", "ru") for index in range(2)]


class _Model:
    def __init__(self, trial_rows: list[GeometryRow]) -> None:
        self.trial_rows = trial_rows
        self.calls = 0
        self.nll_calls = 0

    def get_response_artifacts_with_prefill_residuals_batched(self, prompts, **kwargs):
        self.calls += 1
        count = len(prompts)
        return ([f"candidate-{i}" for i in range(count)], [[i + 1] for i in range(count)], torch.full((count, 2, 2), 0.5))

    def get_conditional_nll(self, prompts, targets):
        self.nll_calls += 1
        return [1.1] * len(prompts)


class _Scorer:
    def score_responses(self, prompts, responses):
        return SimpleNamespace(diagnostics={"margins": [-1.0] * len(prompts)})


def _profile() -> dict[str, object]:
    return {"rows": 660, "scale": [1.0] * 660, "weight": [1.0] * 660, "global_scale": 1.0, "group_scale": {}, "group_weight": {}}


def test_finalist_evaluator_runs_full_pool_and_independent_holdout(tmp_path: Path) -> None:
    trial_rows = _trial_rows(tmp_path)
    final_rows = _final_rows(tmp_path)
    model = _Model(trial_rows)
    clean_trial = [{
        "row_id": row.row_id, "clean_response": f"clean-{i}",
        "clean_response_token_ids": [i + 1],
        "clean_prompt_residual_projection": [1.0, 1.0],
        "clean_conditional_nll": 1.0 if row.direction == "safe" else None,
    } for i, row in enumerate(trial_rows)]
    clean_final = [{
        "row_id": row.row_id, "clean_response": f"clean-final-{i}",
        "clean_margin": 1.0,
        "clean_prompt_residual_projection": [1.0, 1.0],
    } for i, row in enumerate(final_rows)]
    evaluator = MultilingualFinalistEvaluator(
        model=model, trial_rows=trial_rows, clean_trial_records=clean_trial,
        final_rows=final_rows, clean_final_records=clean_final,
        refusal_direction=torch.ones((2, 2)), layer_reliability=torch.ones(2),
        srg_scorer=_Scorer(), srg_profile=_profile(),
        private_output_dir=tmp_path / "private", expected_per_direction=4,
        expected_languages=("en", "ru"), final_max_new_tokens=1024,
    )

    result = evaluator.evaluate(7)

    assert model.calls == 2
    assert model.nll_calls == 1
    assert result.rows == 8
    assert result.to_public_dict()["diagnostics"]["final_holdout"]["rows"] == 4
    assert result.to_public_dict()["diagnostics"]["final_holdout"]["removal"] > 0.0
