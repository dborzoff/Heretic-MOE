# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


def load_merge_module():
    path = Path(__file__).parents[1] / "research" / "scripts" / "merge_optuna_studies.py"
    spec = importlib.util.spec_from_file_location("merge_optuna_studies", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_merge_module()


class MergeContractTests(unittest.TestCase):
    def test_round_robin_merge_assigns_random_even_and_sobol_odd(self) -> None:
        random = [
            ("random", SimpleNamespace(number=number)) for number in range(3)
        ]
        sobol = [("sobol", SimpleNamespace(number=number)) for number in range(3)]

        ordered = merge.order_source_trials([random, sobol], "round-robin")

        self.assertEqual(
            [(source, trial.number) for source, trial in ordered],
            [
                ("random", 0),
                ("sobol", 0),
                ("random", 1),
                ("sobol", 1),
                ("random", 2),
                ("sobol", 2),
            ],
        )

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

    def test_round_robin_merge_writes_one_resumable_journal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name, design in (("random", "random"), ("sobol", "sobol")):
                journal = root / f"{name}.jsonl"
                storage = JournalStorage(
                    JournalFileBackend(
                        str(journal),
                        lock_obj=JournalFileOpenLock(str(journal)),
                    )
                )
                study = optuna.create_study(
                    storage=storage,
                    study_name="heretic",
                    direction="minimize",
                )
                study.set_user_attr(
                    "settings",
                    json.dumps(
                        {
                            "model": "example/model",
                            "scorers": [
                                {
                                    "plugin": "KeywordRate",
                                    "optimization": "minimize",
                                }
                            ],
                            "n_trials": 2,
                            "n_startup_trials": 2,
                            "startup_design": design,
                            "seed": 7,
                            "device_map": "cuda:0",
                        }
                    ),
                )
                study.set_user_attr("constraint_names", [])
                study.optimize(
                    lambda trial: trial.suggest_float("x", 0.0, 1.0),
                    n_trials=2,
                )
                sources.append((name, journal))

            output = root / "merged.jsonl"
            argv = [
                "merge_optuna_studies.py",
                "--source",
                f"random={sources[0][1]}",
                "--source",
                f"sobol={sources[1][1]}",
                "--output",
                str(output),
                "--target-trials",
                "6",
                "--order",
                "round-robin",
            ]
            with patch.object(sys, "argv", argv):
                merge.main()

            merged = merge.load_study(output)
            self.assertEqual(
                [trial.user_attrs["merged_source"] for trial in merged.trials],
                ["random", "sobol", "random", "sobol"],
            )
            manifest = json.loads(
                output.with_suffix(".jsonl.merge.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["merge_order"], "round-robin")
            self.assertEqual(manifest["merged_prefix_trials"], 4)


if __name__ == "__main__":
    unittest.main()
