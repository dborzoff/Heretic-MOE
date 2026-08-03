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


if __name__ == "__main__":
    unittest.main()
