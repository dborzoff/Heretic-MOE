# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixed-panel evaluator for repeatable multilingual finalist rechecks."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .multilingual_trial_evaluator import (
    FrozenMultilingualTrialEvaluator,
    TrialMeasurement,
)
from .utils import Prompt


class MultilingualFinalistEvaluator:
    """Evaluate every finalist against the exact same frozen 400-row panel."""

    def __init__(
        self,
        *,
        runtime: FrozenMultilingualTrialEvaluator,
        fixed_schedule_trial_number: int,
    ) -> None:
        if fixed_schedule_trial_number < 0:
            raise ValueError("fixed schedule trial number must be non-negative")
        self.runtime = runtime
        self.fixed_schedule_trial_number = fixed_schedule_trial_number
        self.expected_per_direction = runtime.expected_per_direction
        self.expected_languages = runtime.expected_languages

    def evaluate(
        self,
        trial_number: int,
        *,
        artifact_trial_number: int | None = None,
        residual_capture: Callable[[list[Prompt], Tensor], None] | None = None,
    ) -> TrialMeasurement:
        del trial_number
        return self.runtime.evaluate(
            self.fixed_schedule_trial_number,
            artifact_trial_number=artifact_trial_number,
            residual_capture=residual_capture,
        )
