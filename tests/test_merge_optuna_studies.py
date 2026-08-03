# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import unittest
from pathlib import Path

import optuna


def load_merge_module():
    path = Path(__file__).parents[1] / "research" / "scripts" / "merge_optuna_studies.py"
    spec = importlib.util.spec_from_file_location("merge_optuna_studies", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_merge_module()


class MergeContractTests(unittest.TestCase):
    def test_stage_controls_do_not_change_contract(self) -> None:
        random_settings = {
            "model": "example/model",
            "scorers": [{"plugin": "KeywordRate", "optimization": "minimize"}],
            "n_trials": 120,
            "n_startup_trials": 60,
            "startup_design": "random",
            "device_map": "cuda:0",
        }
        sobol_settings = dict(
            random_settings,
            startup_design="sobol",
            device_map="cuda:1",
        )

        self.assertEqual(
            merge.study_contract(random_settings),
            merge.study_contract(sobol_settings),
        )

    def test_scorer_change_changes_contract_hash(self) -> None:
        left = {
            "model": "example/model",
            "scorers": [{"plugin": "KeywordRate", "optimization": "minimize"}],
        }
        right = {
            "model": "example/model",
            "scorers": [{"plugin": "Perplexity", "optimization": "minimize"}],
        }

        self.assertNotEqual(
            merge.canonical_hash(merge.study_contract(left)),
            merge.canonical_hash(merge.study_contract(right)),
        )

    def test_distribution_map_supports_conditional_parameters(self) -> None:
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.Trial) -> float:
            enabled = trial.suggest_categorical("component.enabled", [True, False])
            if enabled:
                return trial.suggest_float("component.x", 0.0, 1.0)
            return 0.0

        study.optimize(objective, n_trials=8)
        distributions = merge.distribution_map(study)

        self.assertIn("component.enabled", distributions)
        self.assertTrue(set(distributions).issubset({"component.enabled", "component.x"}))


if __name__ == "__main__":
    unittest.main()
