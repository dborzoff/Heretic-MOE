# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full trial-pool plus independent R-holdout finalist evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from torch import Tensor

from .language_map_data import GeometryRow
from .multilingual_contract import CalibrationRow
from .multilingual_final_holdout import evaluate_final_holdout
from .multilingual_trial_evaluator import TrialMeasurement, evaluate_multilingual_trial


class MultilingualFinalistEvaluator:
    """Remeasure one exact parameter set on all 4,000+660 frozen rows."""

    def __init__(
        self,
        *,
        model: Any,
        trial_rows: Sequence[GeometryRow],
        clean_trial_records: Sequence[Mapping[str, object]],
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
        self.trial_rows = tuple(trial_rows)
        self.clean_trial_records = tuple(clean_trial_records)
        self.final_rows = tuple(final_rows)
        self.clean_final_records = tuple(clean_final_records)
        self.refusal_direction = refusal_direction
        self.layer_reliability = layer_reliability
        self.srg_scorer = srg_scorer
        self.srg_profile = dict(srg_profile)
        self.private_output_dir = Path(private_output_dir).resolve()
        self.expected_per_direction = expected_per_direction
        self.expected_languages = expected_languages
        self.final_max_new_tokens = final_max_new_tokens

    def evaluate(self, trial_number: int, *, residual_capture=None) -> TrialMeasurement:
        full = evaluate_multilingual_trial(
            trial_number=trial_number,
            model=self.model,
            rows=self.trial_rows,
            clean_records=self.clean_trial_records,
            refusal_direction=self.refusal_direction,
            layer_reliability=self.layer_reliability,
            srg_scorer=self.srg_scorer,
            srg_profile=self.srg_profile,
            private_records_path=(
                self.private_output_dir
                / "trial_pool"
                / f"trial-{trial_number:06d}.jsonl"
            ),
            expected_per_direction=self.expected_per_direction,
            expected_languages=self.expected_languages,
            max_response_length=self.final_max_new_tokens,
            residual_capture=residual_capture,
        )
        final = evaluate_final_holdout(
            trial_number=trial_number,
            model=self.model,
            rows=self.final_rows,
            clean_records=self.clean_final_records,
            refusal_direction=self.refusal_direction,
            layer_reliability=self.layer_reliability,
            srg_scorer=self.srg_scorer,
            srg_profile=self.srg_profile,
            private_records_path=(
                self.private_output_dir
                / "final_holdout"
                / f"trial-{trial_number:06d}.jsonl"
            ),
            max_response_length=self.final_max_new_tokens,
        )
        diagnostics = dict(full.diagnostics)
        diagnostics["final_holdout"] = final
        return replace(full, diagnostics=diagnostics)
