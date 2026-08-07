# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import io
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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
    @staticmethod
    def parse_args(*extra: str) -> Namespace:
        argv = [
            "run_adaptive_search.py",
            "--base-config",
            "config.toml",
            "--run-root",
            "run",
            *extra,
        ]
        with patch.object(sys, "argv", argv):
            return controller.parse_args()

    def test_default_post_search_mode_exports_after_top_six_recheck(self) -> None:
        args = self.parse_args()

        self.assertEqual(args.post_search_mode, "export")
        self.assertTrue(args.finalize)
        self.assertFalse(args.recheck_only)
        self.assertEqual(args.finalist_top_n, 6)

    def test_recheck_only_is_distinct_from_search_only_and_export(self) -> None:
        args = self.parse_args("--recheck-only")

        self.assertEqual(args.post_search_mode, "recheck")
        self.assertFalse(args.finalize)
        self.assertTrue(args.recheck_only)
        self.assertEqual(
            controller.post_search_completion_status(args.post_search_mode),
            "recheck_complete",
        )

    def test_legacy_search_only_aliases_select_no_post_search_work(self) -> None:
        for flag in ("--search-only", "--no-finalize"):
            with self.subTest(flag=flag):
                args = self.parse_args(flag)
                self.assertEqual(args.post_search_mode, "none")
                self.assertFalse(args.finalize)
                self.assertFalse(args.recheck_only)

    def test_post_search_modes_are_mutually_exclusive(self) -> None:
        for conflicting in ("--finalize", "--search-only", "--no-finalize"):
            with self.subTest(conflicting=conflicting):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.parse_args("--recheck-only", conflicting)

    def test_recheck_only_returns_before_creating_model_exports(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            journal = root / "journal.log"
            journal.write_text("immutable journal\n", encoding="utf-8")
            base_config = root / "config.toml"
            base_config.write_text('model = "example/model"\n', encoding="utf-8")
            finalist_dir = root / "finalists-v1"
            finalist_dir.mkdir()
            (finalist_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (finalist_dir / "winners.json").write_text("{}\n", encoding="utf-8")
            export_root = root / "exports-v1"
            workflow_path = root / "workflow.json"
            stage = controller.Stage(
                "shared_tpe",
                root,
                base_config,
                journal,
                None,
            )
            args = Namespace(dry_run=False, export_strategy="merge")
            winners = {
                "Balanced": {
                    "trial_number": 11,
                    "source_trial_index": 7,
                    "params_sha256": "same",
                },
                "Max": {
                    "trial_number": 11,
                    "source_trial_index": 7,
                    "params_sha256": "same",
                },
            }
            selected = (1, finalist_dir, export_root, workflow_path, {"v": 1})

            with (
                patch.object(controller, "assigned_devices", return_value=["0"]),
                patch.object(
                    controller,
                    "select_finalization_paths",
                    return_value=selected,
                ),
                patch.object(
                    controller,
                    "finalization_manifest_matches",
                    return_value=True,
                ),
                patch.object(
                    controller,
                    "prepared_finalization_artifacts_match",
                    return_value=True,
                ),
                patch.object(
                    controller,
                    "load_valid_winners_report",
                    return_value={"status": "PASS", "winners": winners},
                ),
                patch.object(controller.subprocess, "Popen") as popen,
            ):
                controller.finalize_and_export(
                    args,
                    root=root,
                    base_config=base_config,
                    shared_stage=stage,
                    executable=Path("hereticMOE.exe"),
                    export_models=False,
                )

            self.assertFalse(export_root.exists())
            popen.assert_not_called()

    def test_recheck_dry_run_reads_the_completed_shared_journal(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "journal.log"
            journal.write_text("immutable journal\n", encoding="utf-8")
            with patch.object(
                controller,
                "journal_trial_counts",
                return_value=(600, 0),
            ) as trial_counts:
                result = controller.controller_trial_counts(
                    journal,
                    dry_run=True,
                    continue_shared_only=True,
                    dynamic_worker_queue=True,
                    exploration_trials=120,
                )

        self.assertEqual(result, (600, 0))
        trial_counts.assert_called_once_with(journal)

    def test_completed_recheck_does_not_require_tpe_constraint_backfill(self) -> None:
        self.assertFalse(
            controller.should_require_constraint_metadata(
                dry_run=False,
                journal_exists=True,
                remaining_trials=0,
            )
        )
        self.assertTrue(
            controller.should_require_constraint_metadata(
                dry_run=False,
                journal_exists=True,
                remaining_trials=1,
            )
        )

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
