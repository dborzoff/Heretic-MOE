# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluator adapter for the frozen multilingual search runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from optuna.study import StudyDirection
from torch import Tensor

from .multilingual_trial_evaluator import FrozenMultilingualTrialEvaluator
from .scorer import Score
from .utils import Prompt


@dataclass(frozen=True)
class MultilingualConstraintContract:
    """Immutable upper bounds expressed in normalized clean-relative units."""

    max_safe_ppl_drift: float = 0.005
    max_safe_geometry_damage: float = 1.0
    max_language_instability: float = 1.0
    max_category_instability: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not 0.0 <= float(value):
                raise ValueError(f"{name} must be finite and nonnegative")


class MultilingualSearchEvaluator:
    """Expose multilingual measurements through the legacy search interface."""

    def __init__(
        self,
        runtime: FrozenMultilingualTrialEvaluator,
        *,
        constraints: MultilingualConstraintContract,
    ) -> None:
        self.runtime = runtime
        self.constraints = constraints
        self.baseline_scores = self._baseline_scores()

    @staticmethod
    def _baseline_scores() -> list[tuple[str, Score]]:
        return [
            ("Removal", Score(0.0, "+0.00000", "0.00000")),
            ("Preservation loss", Score(0.0, "0.00000", "0.00000")),
            ("Cost↑", Score(0.5, "0.500", "0.500")),
        ]

    def get_scores(
        self,
        response_archive_id: str | int | None = None,
        residual_capture: Callable[[list[Prompt], Tensor], None] | None = None,
    ) -> list[tuple[str, Score]]:
        if not isinstance(response_archive_id, int) or response_archive_id < 0:
            raise ValueError("multilingual evaluation requires a global trial number")
        measurement = (
            self.runtime.evaluate(response_archive_id)
            if residual_capture is None
            else self.runtime.evaluate(
                response_archive_id,
                residual_capture=residual_capture,
            )
        )
        metrics = measurement.metrics
        public = measurement.to_public_dict()
        diagnostics: dict[str, Any] = {
            **public,
            "private_records_sha256": measurement.private_records_sha256,
        }
        return [
            (
                "Removal",
                Score(
                    metrics.removal,
                    f"{metrics.removal:+.5f}",
                    f"{metrics.removal:+.5f}",
                    diagnostics=diagnostics,
                ),
            ),
            (
                "Preservation loss",
                Score(
                    metrics.preservation_loss,
                    f"{metrics.preservation_loss:.5f}",
                    f"{metrics.preservation_loss:.5f}",
                    diagnostics=diagnostics,
                ),
            ),
            (
                "Cost↑",
                Score(
                    metrics.cost_up,
                    f"{metrics.cost_up:.3f}",
                    f"{metrics.cost_up:.3f}",
                    diagnostics=diagnostics,
                ),
            ),
        ]

    @staticmethod
    def get_objective_names() -> list[str]:
        return ["Removal", "Preservation loss"]

    @staticmethod
    def get_objective_directions() -> list[StudyDirection]:
        return [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]

    @staticmethod
    def get_objective_values(scores: list[tuple[str, Score]]) -> tuple[float, ...]:
        by_name = {name: score for name, score in scores}
        return (by_name["Removal"].value, by_name["Preservation loss"].value)

    def get_constraint_names(self) -> list[str]:
        return [
            f"Safe PPL drift <= {self.constraints.max_safe_ppl_drift}",
            (
                "Safe geometry damage <= "
                f"{self.constraints.max_safe_geometry_damage}"
            ),
            (
                "Language instability <= "
                f"{self.constraints.max_language_instability}"
            ),
            (
                "Category instability <= "
                f"{self.constraints.max_category_instability}"
            ),
        ]

    def get_constraint_values(
        self, scores: list[tuple[str, Score]]
    ) -> tuple[float, ...]:
        removal = dict(scores)["Removal"]
        diagnostics = removal.diagnostics or {}
        metrics = diagnostics.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("multilingual score is missing metric diagnostics")
        return (
            float(metrics["safe_ppl_drift"])
            - self.constraints.max_safe_ppl_drift,
            float(metrics["safe_geometry_drift"])
            - self.constraints.max_safe_geometry_damage,
            float(metrics["language_instability"])
            - self.constraints.max_language_instability,
            float(metrics["category_instability"])
            - self.constraints.max_category_instability,
        )

    def get_paired_score_records(
        self, scores: list[tuple[str, Score]]
    ) -> list[dict[str, Any]]:
        baselines = dict(self.baseline_scores)
        return [
            {
                "name": name,
                "score": dict(score.__dict__),
                "baseline": dict(baselines[name].__dict__),
            }
            for name, score in scores
        ]

    @staticmethod
    def get_dataset_specifications() -> list[object]:
        return []

    @staticmethod
    def all_scorers_reproducible() -> bool:
        return True

    @staticmethod
    def all_scorers_builtin() -> bool:
        return True
