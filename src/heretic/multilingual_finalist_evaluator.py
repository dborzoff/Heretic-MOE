# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full trial-pool plus independent R-holdout finalist evaluator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import Tensor

from .language_map_data import GeometryRow
from .multilingual_contract import CalibrationRow
from .multilingual_trial_evaluator import (
    TrialMeasurement,
    _clean_by_row_id,
    evaluate_multilingual_trial,
)
from .utils import Prompt


class MultilingualFinalistEvaluator:
    """Remeasure one exact parameter set on the independent final pool."""

    def __init__(
        self,
        *,
        model: Any,
        final_rows: Sequence[CalibrationRow],
        clean_final_records: Sequence[Mapping[str, object]],
        refusal_direction: Tensor,
        layer_reliability: Tensor,
        srg_scorer: Any,
        srg_profile: Mapping[str, object],
        private_output_dir: str | Path,
        expected_per_direction: int,
        expected_languages: tuple[str, ...],
        final_max_new_tokens: int,
    ) -> None:
        self.model = model
        self.final_rows = tuple(
            GeometryRow(
                canonical_id=row.base_id,
                row_id=row.row_id,
                language=row.language.lower(),
                direction=row.direction,  # type: ignore[arg-type]
                category_id=row.category_id,
                prompt=row.prompt,
                source_path=row.source_path,
                source_line=row.source_line,
            )
            for row in final_rows
        )
        self.clean_final_records = tuple(clean_final_records)
        self.refusal_direction = refusal_direction
        self.layer_reliability = layer_reliability
        self.srg_scorer = srg_scorer
        self.srg_profile = dict(srg_profile)
        clean_by_id = _clean_by_row_id(self.final_rows, self.clean_final_records)
        self.clean_final_srg_margins = {
            row.row_id: float(clean_by_id[row.row_id]["clean_margin"])
            for row in self.final_rows
        }
        self.private_output_dir = Path(private_output_dir).resolve()
        self.expected_per_direction = expected_per_direction
        self.expected_languages = expected_languages
        self.final_max_new_tokens = final_max_new_tokens

    def evaluate(
        self,
        trial_number: int,
        *,
        artifact_trial_number: int | None = None,
        residual_capture: Callable[[list[Prompt], Tensor], None] | None = None,
    ) -> TrialMeasurement:
        artifact_number = (
            trial_number if artifact_trial_number is None else artifact_trial_number
        )
        if artifact_number < 0:
            raise ValueError("artifact trial number must be non-negative")
        return evaluate_multilingual_trial(
            trial_number=artifact_number,
            schedule_trial_number=trial_number,
            model=self.model,
            rows=self.final_rows,
            clean_records=self.clean_final_records,
            refusal_direction=self.refusal_direction,
            layer_reliability=self.layer_reliability,
            srg_scorer=self.srg_scorer,
            srg_profile=self.srg_profile,
            clean_srg_margins=self.clean_final_srg_margins,
            private_records_path=(
                self.private_output_dir
                / "final_pool"
                / f"trial-{artifact_number:06d}.jsonl"
            ),
            expected_per_direction=self.expected_per_direction,
            expected_languages=self.expected_languages,
            max_response_length=self.final_max_new_tokens,
            residual_capture=residual_capture,
        )
