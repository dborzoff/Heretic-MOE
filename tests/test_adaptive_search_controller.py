# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def load_controller_module():
    path = Path(__file__).parents[1] / "research" / "scripts" / "run_adaptive_search.py"
    name = "run_adaptive_search"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load_controller_module()


class AdaptiveSearchControllerTests(unittest.TestCase):
    def test_sanitized_model_name_matches_heretic_checkpoint_name(self) -> None:
        self.assertEqual(
            controller.sanitized_model_name("F:/models/example"),
            "F----models--example",
        )

    def test_managed_config_allows_only_explicit_target_update(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.toml"
            original = {"model": "example/model", "n_trials": 600}
            controller.write_managed_config(path, original, dry_run=False)

            extended = dict(original, n_trials=1000)
            controller.write_managed_config(
                path,
                extended,
                dry_run=False,
                allowed_updates=frozenset({"n_trials"}),
            )
            self.assertEqual(controller.read_config(path)["n_trials"], 1000)

            with self.assertRaisesRegex(FileExistsError, "Changed keys"):
                controller.write_managed_config(
                    path,
                    dict(extended, model="different/model"),
                    dry_run=False,
                    allowed_updates=frozenset({"n_trials"}),
                )

    def test_total_exploration_is_split_evenly_between_branches(self) -> None:
        self.assertEqual(controller.split_worker_budget(120, 2), [60, 60])

    def test_remaining_tpe_budget_is_split_between_two_gpus(self) -> None:
        self.assertEqual(controller.split_worker_budget(600 - 120, 2), [240, 240])

    def test_stage_config_maps_branch_trials_to_even_numbers(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = controller.stage_config(
                {"model": "example/model", "save_trial_responses": True},
                checkpoint_dir=root / "checkpoints",
                n_trials=60,
                n_startup_trials=60,
                startup_design="random",
                response_archive=root / "trial-responses.sqlite3",
                response_number_offset=0,
                response_number_stride=2,
                parallel_workers=1,
            )

        self.assertEqual(config["n_trials"], 60)
        self.assertEqual(config["n_startup_trials"], 60)
        self.assertEqual(config["trial_response_number_offset"], 0)
        self.assertEqual(config["trial_response_number_stride"], 2)


if __name__ == "__main__":
    unittest.main()
