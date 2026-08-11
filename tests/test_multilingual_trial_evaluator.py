from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from heretic.language_map_data import GeometryRow
from heretic.multilingual_trial_evaluator import (
    FrozenMultilingualTrialEvaluator,
    evaluate_multilingual_trial,
)


def _rows(tmp_path: Path) -> list[GeometryRow]:
    rows: list[GeometryRow] = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for index, language in enumerate(("en", "ru", "en", "ru")):
            rows.append(
                GeometryRow(
                    canonical_id=f"{prefix}{index}",
                    row_id=f"{language}-{prefix}{index}",
                    language=language,
                    direction=direction,
                    category_id=f"C{index % 2 + 1:02d}",
                    prompt=f"private-{direction}-{index}",
                    source_path=tmp_path / "private.jsonl",
                    source_line=index + 1,
                )
            )
    return rows


def _clean_records(rows: list[GeometryRow]) -> list[dict[str, object]]:
    records = []
    for index, row in enumerate(rows):
        base = float(index + 2)
        records.append(
            {
                "row_id": row.row_id,
                "clean_response": f"clean-{row.row_id}",
                "clean_response_token_ids": [index + 1, index + 2],
                "clean_prompt_residual_projection": [base, base + 1.0],
                "clean_conditional_nll": 1.0 if row.direction == "safe" else None,
            }
        )
    return records


class _FakeModel:
    def __init__(self, rows: list[GeometryRow]) -> None:
        self.rows = rows
        self.generation_phases = 0
        self.nll_calls = 0

    def get_response_artifacts_with_prefill_residuals_batched(self, prompts, **kwargs):
        self.generation_phases += 1
        responses = [f"candidate-{row.row_id}" for row in self.rows]
        token_ids = [[100 + index] for index in range(len(self.rows))]
        residuals = []
        for index, row in enumerate(self.rows):
            base = float(index + 2)
            if row.direction == "unsafe":
                base -= 0.5
            residuals.append([[base, 0.0], [base + 1.0, 0.0]])
        return responses, token_ids, torch.tensor(residuals)

    def get_conditional_nll(self, prompts, target_token_ids):
        self.nll_calls += 1
        return [1.1 for _ in prompts]


class _FakeSRG:
    def __init__(self) -> None:
        self.calls = 0

    def score_responses(self, prompts, responses):
        self.calls += 1
        if all(value.startswith("clean-") for value in responses):
            margins = [1.0, -1.0, 1.0, -1.0]
        elif all(value.startswith("candidate-") for value in responses):
            margins = [-1.0, -1.0, 0.5, -1.0]
        else:
            raise AssertionError("unexpected response set")
        return SimpleNamespace(diagnostics={"margins": margins})


def _profile() -> dict[str, object]:
    return {
        "rows": 660,
        "scale": [1.0] * 660,
        "weight": [1.0] * 660,
        "global_scale": 1.0,
        "group_scale": {},
        "group_weight": {},
    }


def test_one_trial_phase_produces_all_metrics_and_private_records(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    model = _FakeModel(rows)
    scorer = _FakeSRG()
    private_path = tmp_path / "private" / "trial-17.jsonl"

    measurement = evaluate_multilingual_trial(
        trial_number=17,
        model=model,
        rows=rows,
        clean_records=_clean_records(rows),
        refusal_direction=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        layer_reliability=torch.ones(2),
        srg_scorer=scorer,
        srg_profile=_profile(),
        private_records_path=private_path,
        expected_per_direction=4,
        expected_languages=("en", "ru"),
    )

    assert model.generation_phases == 1
    assert model.nll_calls == 1
    assert scorer.calls == 2
    assert measurement.metrics.srg_gain > 0.0
    assert measurement.metrics.r_gain > 0.0
    assert measurement.metrics.unsafe_geometry_gain > 0.0
    assert measurement.metrics.safe_geometry_drift == pytest.approx(0.0)
    assert measurement.metrics.safe_ppl_drift == pytest.approx(0.105170918)
    public = measurement.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    assert not any(key in serialized for key in ('"prompt"', '"response"', '"text"'))
    assert public["rows"] == 8
    assert public["safe_rows"] == 4
    assert public["unsafe_rows"] == 4
    records = [json.loads(line) for line in private_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    assert records[0]["trial_number"] == 17
    assert records[0]["prompt"].startswith("private-")
    assert records[0]["response"].startswith("candidate-")


def test_trial_rejects_incomplete_direction_coverage(tmp_path: Path) -> None:
    rows = _rows(tmp_path)[:-1]
    with pytest.raises(ValueError, match="400 SAFE and 400 UNSAFE"):
        evaluate_multilingual_trial(
            trial_number=1,
            model=_FakeModel(rows),
            rows=rows,
            clean_records=_clean_records(rows),
            refusal_direction=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            layer_reliability=torch.ones(2),
            srg_scorer=_FakeSRG(),
            srg_profile=_profile(),
            private_records_path=tmp_path / "trial.jsonl",
            expected_per_direction=4,
        )


def test_trial_rejects_language_imbalance_even_when_direction_counts_match(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path)
    first_safe = next(row for row in rows if row.direction == "safe")
    rows[0] = GeometryRow(
        canonical_id=first_safe.canonical_id,
        row_id=first_safe.row_id,
        language="ru",
        direction=first_safe.direction,
        category_id=first_safe.category_id,
        prompt=first_safe.prompt,
        source_path=first_safe.source_path,
        source_line=first_safe.source_line,
    )

    with pytest.raises(ValueError, match="balanced by language"):
        evaluate_multilingual_trial(
            trial_number=1,
            model=_FakeModel(rows),
            rows=rows,
            clean_records=_clean_records(rows),
            refusal_direction=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            layer_reliability=torch.ones(2),
            srg_scorer=_FakeSRG(),
            srg_profile=_profile(),
            private_records_path=tmp_path / "trial.jsonl",
            expected_per_direction=4,
            expected_languages=("en", "ru"),
        )


def test_frozen_evaluator_resolves_global_trial_schedule(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    model = _FakeModel(rows)
    evaluator = FrozenMultilingualTrialEvaluator(
        model=model,
        trial_rows=rows,
        schedule_records=[
            {"trial_number": 17, "row_ids": [row.row_id for row in rows]}
        ],
        clean_records=_clean_records(rows),
        refusal_direction=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        layer_reliability=torch.ones(2),
        srg_scorer=_FakeSRG(),
        srg_profile=_profile(),
        private_output_dir=tmp_path / "trials",
        expected_per_direction=4,
        expected_languages=("en", "ru"),
    )

    measurement = evaluator.evaluate(17)

    assert measurement.trial_number == 17
    assert (tmp_path / "trials" / "trial-000017.jsonl").is_file()
    with pytest.raises(KeyError, match="18"):
        evaluator.evaluate(18)
