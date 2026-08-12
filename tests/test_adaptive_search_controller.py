# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import io
import json
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from heretic.work_queue import TrialWorkQueue


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
        self.assertEqual(args.keyword_near_gate_extra, 1)

    def test_multilingual_v3_config_does_not_require_legacy_cost_scorers(self) -> None:
        controller.validate_adaptive_cost_contract(
            {
                "model": "example/model",
                "selection_policy": "feasible_diverse",
                "multilingual_search": {
                    "enabled": True,
                    "dataset_root": "F:/dataset",
                },
            },
            source=Path("multilingual-v3.toml"),
        )

    def test_multilingual_data_root_points_at_frozen_dataset_layout(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "operative_split_1000_400_v1"
            split.mkdir()
            (root / "manifest.json").write_text("{}\n", encoding="utf-8")
            (split / "manifest.json").write_text("{}\n", encoding="utf-8")
            config = controller.apply_data_root(
                {
                    "model": "example/model",
                    "multilingual_search": {
                        "enabled": True,
                        "dataset_root": "old",
                    },
                },
                root,
            )

        self.assertEqual(
            config["multilingual_search"]["dataset_root"], root.as_posix()
        )
        self.assertEqual(
            config["multilingual_search"]["split_root"], split.as_posix()
        )

    def test_multilingual_preparation_commands_use_all_devices_and_frozen_pools(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "dataset" / "operative_split_1000_400_v1"
            split.mkdir(parents=True)
            config = {
                "model": "F:/models/qwen",
                "batch_size": 8,
                "dtypes": ["bfloat16"],
                "seed": 19,
                "multilingual_search": {
                    "enabled": True,
                    "dataset_root": (root / "dataset").as_posix(),
                    "split_root": split.as_posix(),
                    "languages": ["en", "ru", "zh", "es", "fr"],
                    "direction_rows_per_cell": 1000,
                },
            }
            geometry = controller.multilingual_geometry_command(
                config,
                executable=Path("hereticMOE.exe"),
                run_root=root / "run",
                devices=["0", "1", "3"],
            )
            prepare = controller.multilingual_runtime_prepare_command(
                config,
                executable=Path("hereticMOE.exe"),
                base_config=Path("config.toml"),
                run_root=root / "run",
                devices=["0", "1", "3"],
                srg_source=Path("srg"),
            )

        self.assertIn("0,1,3", geometry)
        self.assertEqual(geometry.count("--group-a"), 5)
        self.assertEqual(geometry.count("--group-b"), 5)
        self.assertIn("prepare-multilingual", prepare)
        self.assertEqual(prepare[prepare.index("--devices") + 1], "0,1,3")
        self.assertEqual(prepare[prepare.index("--batch-size") + 1], "8")
        self.assertEqual(
            Path(prepare[prepare.index("--runtime-root") + 1]),
            root / "run" / "runtime",
        )

    def test_multilingual_runtime_preparation_dry_run_prints_both_stages(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "dataset" / "operative_split_1000_400_v1"
            split.mkdir(parents=True)
            config_path = root / "config.toml"
            config_path.write_text('model = "F:/models/qwen"\n', encoding="utf-8")
            config = {
                "model": "F:/models/qwen",
                "batch_size": 8,
                "dtypes": ["bfloat16"],
                "seed": 19,
                "multilingual_search": {
                    "enabled": True,
                    "dataset_root": (root / "dataset").as_posix(),
                    "split_root": split.as_posix(),
                    "languages": ["en", "ru", "zh", "es", "fr"],
                    "direction_rows_per_cell": 1000,
                },
            }
            output = io.StringIO()
            with redirect_stdout(output), patch.object(
                controller.subprocess, "run"
            ) as run:
                result = controller.prepare_multilingual_run_runtime(
                    config,
                    executable=Path("hereticMOE.exe"),
                    base_config=config_path,
                    run_root=root / "run",
                    devices=["0", "1"],
                    srg_source=root / "srg",
                    dry_run=True,
                )

        self.assertEqual(result["status"], "DRY_RUN")
        self.assertIn("multilingual_geometry_prepare", output.getvalue())
        self.assertIn("multilingual_runtime_prepare", output.getvalue())
        run.assert_not_called()

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
                return_value=controller.JournalTrialCounts(
                    total=600,
                    complete=600,
                    waiting=0,
                ),
            ) as trial_counts:
                result = controller.controller_trial_counts(
                    journal,
                    dry_run=True,
                    continue_shared_only=True,
                    dynamic_worker_queue=True,
                    exploration_trials=120,
                )

        self.assertEqual(
            result,
            controller.JournalTrialCounts(total=600, complete=600, waiting=0),
        )
        trial_counts.assert_called_once_with(journal)

    def test_empty_journal_is_treated_as_an_uninitialized_study(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "journal.log"
            journal.touch()

            self.assertEqual(
                controller.journal_trial_counts(journal),
                controller.JournalTrialCounts(total=0, complete=0, waiting=0),
            )

    def test_empty_journal_has_no_trials_for_queue_recovery(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "journal.log"
            journal.touch()

            self.assertEqual(controller.load_journal_trials(journal), [])

    def test_empty_journal_has_no_orphaned_worker_trials(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "journal.log"
            journal.touch()

            self.assertEqual(
                controller.fail_running_trials_for_worker(journal, "gpu-0"),
                [],
            )

    def test_failed_trials_do_not_satisfy_the_completed_trial_target(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            journal = Path(temporary_directory) / "journal.log"
            storage = JournalStorage(
                JournalFileBackend(
                    str(journal),
                    lock_obj=JournalFileOpenLock(str(journal)),
                )
            )
            study = optuna.create_study(storage=storage, study_name="heretic")
            study.optimize(lambda _: 1.0, n_trials=1)

            def fail(_: optuna.Trial) -> float:
                raise RuntimeError("synthetic failure")

            study.optimize(fail, n_trials=1, catch=(RuntimeError,))
            counts = controller.journal_trial_counts(journal)

        self.assertEqual(counts.total, 2)
        self.assertEqual(counts.complete, 1)
        self.assertEqual(counts.waiting, 0)
        self.assertEqual(controller.remaining_complete_trial_budget(2, counts), 1)

    def test_queue_contract_separates_base_trial_records_from_completed_work(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            queue = TrialWorkQueue(Path(temporary_directory) / "queue.sqlite3")
            queue.initialize(
                first_task_id=599,
                task_count=1,
                target_trial_count=600,
                tpe_concurrency=2,
                journal_base_trial_count=602,
                journal_base_complete_count=599,
                journal_base_size_bytes=0,
                journal_base_sha256=sha256().hexdigest(),
            )

            contract = queue.contract()

        self.assertEqual(contract.first_task_id, 599)
        self.assertEqual(contract.journal_base_trial_count, 602)
        self.assertEqual(contract.journal_base_complete_count, 599)

    def test_completed_recheck_does_not_require_tpe_constraint_backfill(self) -> None:
        self.assertFalse(
            controller.should_require_constraint_metadata(
                dry_run=False,
                journal_has_trials=True,
                remaining_trials=0,
            )
        )
        self.assertTrue(
            controller.should_require_constraint_metadata(
                dry_run=False,
                journal_has_trials=True,
                remaining_trials=1,
            )
        )

    def test_uninitialized_journal_does_not_require_constraint_backfill(self) -> None:
        self.assertFalse(
            controller.should_require_constraint_metadata(
                dry_run=False,
                journal_has_trials=False,
                remaining_trials=600,
            )
        )

    def test_constraint_guard_ignores_exploration_trials(self) -> None:
        trials = [
            SimpleNamespace(
                number=0,
                user_attrs={"queue_task_kind": "random"},
                system_attrs={},
            ),
            SimpleNamespace(
                number=1,
                user_attrs={"queue_task_kind": "sobol"},
                system_attrs={},
            ),
            SimpleNamespace(
                number=2,
                user_attrs={"queue_task_kind": "tpe"},
                system_attrs={"constraints": [0.0]},
            ),
        ]

        self.assertEqual(controller.missing_constraint_metadata(trials), [])

    def test_constraint_guard_rejects_tpe_trial_without_metadata(self) -> None:
        trials = [
            SimpleNamespace(
                number=120,
                user_attrs={"queue_task_kind": "tpe"},
                system_attrs={},
            )
        ]

        self.assertEqual(controller.missing_constraint_metadata(trials), [120])

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

    def test_worker_processes_use_device_specific_compiler_caches(self) -> None:
        with patch.dict(
            controller.os.environ,
            {
                "TRITON_CACHE_DIR": "F:/cache/triton",
                "TORCHINDUCTOR_CACHE_DIR": "F:/cache/inductor",
            },
            clear=False,
        ):
            environment = controller.process_environment("1")

        self.assertEqual(
            Path(environment["TRITON_CACHE_DIR"]),
            Path("F:/cache/triton/gpu-1"),
        )
        self.assertEqual(
            Path(environment["TORCHINDUCTOR_CACHE_DIR"]),
            Path("F:/cache/inductor/gpu-1"),
        )

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

    def test_stage_config_pins_multilingual_runtime_below_run_root(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = controller.stage_config(
                {
                    "model": "example/model",
                    "multilingual_search": {
                        "enabled": True,
                        "dataset_root": "F:/dataset",
                    },
                },
                checkpoint_dir=root / "checkpoints",
                n_trials=600,
                n_startup_trials=0,
                startup_design="random",
                response_archive=root / "trial-responses.sqlite3",
                response_number_offset=0,
                response_number_stride=1,
                parallel_workers=2,
                runtime_root=root / "runtime",
            )

        self.assertEqual(
            config["multilingual_search"]["runtime_root"],
            (root / "runtime").as_posix(),
        )

    def test_load_valid_winners_report_accepts_multilingual_v3(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "winners.json"
            report = {
                "status": "PASS",
                "contract": "multilingual_v3_full_recheck",
                "winners": {
                    "Balanced": {
                        "trial_number": 2,
                        "source_trial_index": 403,
                        "source_trial_number": 402,
                    },
                    "Max": {
                        "trial_number": 4,
                        "source_trial_index": 542,
                        "source_trial_number": 541,
                    },
                },
                "winners_distinct": True,
                "measured": [{"source_trial_number": 402}, {"source_trial_number": 541}],
            }
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = controller.load_valid_winners_report(path)

        self.assertEqual(loaded, report)

    def test_console_safe_text_replaces_glyphs_missing_from_cp1251(self) -> None:
        rendered = controller.console_safe_text("GPU 0 | 25% ▏", "cp1251")

        self.assertEqual(rendered, "GPU 0 | 25% ?")


if __name__ == "__main__":
    unittest.main()
