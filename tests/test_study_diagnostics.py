# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import tempfile
import unittest
from pathlib import Path

import optuna

from heretic.study_diagnostics import (
    ParameterImportanceReporter,
    build_parameter_importance_report,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class StudyDiagnosticsTests(unittest.TestCase):
    def _study(self) -> optuna.study.Study:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.RandomSampler(seed=7),
        )

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            x = trial.suggest_float("x", -1.0, 1.0)
            y = trial.suggest_float("y", -1.0, 1.0)
            return x * x + 0.01 * y * y, y * y + 0.01 * x * x

        study.optimize(objective, n_trials=24)
        return study

    def test_builds_one_importance_map_per_objective(self) -> None:
        report = build_parameter_importance_report(
            self._study(),
            ["refusal", "perplexity"],
            seed=11,
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["completed_trials"], 24)
        self.assertEqual(len(report["objectives"]), 2)
        for objective in report["objectives"]:
            self.assertEqual(set(objective["importances"]), {"x", "y"})
            self.assertAlmostEqual(sum(objective["importances"].values()), 1.0)

    def test_periodic_report_is_written_only_at_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "importance.json"
            callback = ParameterImportanceReporter(
                interval=8,
                output_path=output_path,
                objective_names=["refusal", "perplexity"],
                seed=11,
            )
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                sampler=optuna.samplers.RandomSampler(seed=7),
            )

            def objective(trial: optuna.Trial) -> tuple[float, float]:
                x = trial.suggest_float("x", -1.0, 1.0)
                y = trial.suggest_float("y", -1.0, 1.0)
                return x * x, y * y

            study.optimize(objective, n_trials=7, callbacks=[callback])
            self.assertFalse(output_path.exists())

            study.optimize(objective, n_trials=1, callbacks=[callback])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["completed_trials"], 8)


if __name__ == "__main__":
    unittest.main()
