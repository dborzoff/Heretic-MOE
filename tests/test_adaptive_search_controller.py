# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import io
import json
import os
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

    def test_search_worker_batch_event_is_forwarded_as_human_status(self) -> None:
        line = json.dumps(
            {
                "event": "batch_validation",
                "mode": "generation",
                "batch_size": 40,
                "max_new_tokens": 100,
            }
        )

        self.assertEqual(
            controller.format_stage_worker_line("1", line),
            "GPU 1 | Validating batch 40 with 100 generated tokens...",
        )

    def test_unrelated_worker_line_is_not_forwarded_as_status(self) -> None:
        self.assertIsNone(controller.format_stage_worker_line("0", "ordinary output"))

    def test_git_provenance_counts_untracked_files_as_dirty(self) -> None:
        calls: list[list[str]] = []

        class Result:
            def __init__(self, stdout: str) -> None:
                self.stdout = stdout

        def run(command, **kwargs):
            del kwargs
            calls.append(list(command))
            if command[:2] == ["git", "rev-parse"]:
                return Result("abc123\n")
            return Result("?? new-file.txt\n")

        with patch.object(controller.subprocess, "run", side_effect=run):
            revision, dirty = controller.git_provenance(Path("repo"))

        self.assertEqual(revision, "abc123")
        self.assertTrue(dirty)
        self.assertIn("--untracked-files=all", calls[1])

    def test_direct_controller_lock_rejects_second_writer_and_allows_supervised_child(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run"
            with controller.controller_run_lock(
                root, supervised=False, dry_run=False
            ):
                with (
                    self.assertRaises(controller.ControllerRunLockError),
                    controller.controller_run_lock(
                        root, supervised=False, dry_run=False
                    ),
                ):
                    self.fail("a second direct controller acquired the run lock")
                with controller.controller_run_lock(root, supervised=True, dry_run=False):
                    pass
                with controller.controller_run_lock(root, supervised=False, dry_run=True):
                    pass

    def test_lock_contention_never_marks_the_active_run_failed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run"
            args = SimpleNamespace(run_root=root, dry_run=False)
            with controller.controller_run_lock(
                root, supervised=False, dry_run=False
            ):
                with (
                    self.assertRaises(controller.ControllerRunLockError),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(controller, "mark_existing_run_failed") as mark_failed,
                ):
                    controller.run_controller(
                        args,
                        arguments=["--run-root", str(root)],
                    )
                mark_failed.assert_not_called()

    def test_controller_failure_after_lock_acquisition_is_recorded(self) -> None:
        error = RuntimeError("worker failed")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run"
            args = SimpleNamespace(run_root=root, dry_run=False)
            arguments = ["--run-root", str(root)]
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(controller, "_main_with_args", side_effect=error),
                patch.object(controller, "mark_existing_run_failed") as mark_failed,
                self.assertRaisesRegex(RuntimeError, "worker failed"),
            ):
                controller.run_controller(args, arguments=arguments)
            mark_failed.assert_called_once_with(arguments, error)

    def test_dry_run_failure_never_marks_an_existing_run_failed(self) -> None:
        error = RuntimeError("dry-run validation failed")
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "run"
            args = SimpleNamespace(run_root=root, dry_run=True)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(controller, "_main_with_args", side_effect=error),
                patch.object(controller, "mark_existing_run_failed") as mark_failed,
                self.assertRaisesRegex(RuntimeError, "dry-run validation failed"),
            ):
                controller.run_controller(args, arguments=["--run-root", str(root)])
            mark_failed.assert_not_called()

    def test_multilingual_v3_finalization_contract_excludes_legacy_recheck_gates(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            args = SimpleNamespace(
                finalist_top_n=6,
                finalist_selection_policy="feasible_cost",
                recheck_ppl_chunks=64,
                recheck_ppl_window=1024,
                max_ppl_drift=0.005,
                max_keywords=2,
                keyword_total=136,
                keyword_near_gate_extra=1,
                balanced_srg_gate=None,
                balanced_removal_fraction=0.8,
                export_root=None,
                export_strategy="merge",
                heretic_path="hereticMOE.exe",
                heretic_sha256="a" * 64,
                multilingual_v3_enabled=True,
            )
            contract = controller.build_finalization_contract(
                args,
                root=root,
                source_journal_sha256="b" * 64,
                base_config_sha256="c" * 64,
            )
            manifest = {
                "version": 1,
                "status": "prepared",
                "contract": "multilingual_v3_full_recheck",
                "source_journal_sha256": "b" * 64,
                "base_config_sha256": "c" * 64,
                "top_n": 6,
                "selection_policy": "feasible_cost",
                "gates": {"balanced_removal_fraction": 0.8},
                "finalization_overrides": None,
            }

            self.assertIsNone(contract["ppl"])
            self.assertEqual(contract["gates"], {"balanced_removal_fraction": 0.8})
            self.assertTrue(controller.finalization_manifest_matches(manifest, contract))
            wrong_contract_manifest = {**manifest, "contract": "legacy_recheck"}
            self.assertFalse(
                controller.finalization_manifest_matches(
                    wrong_contract_manifest,
                    contract,
                )
            )
            legacy_manifest = {
                **manifest,
                "gates": {
                    "balanced_removal_fraction": 0.8,
                    "max_ppl_drift": 0.005,
                },
            }
            self.assertFalse(
                controller.finalization_manifest_matches(legacy_manifest, contract)
            )

    def test_legacy_finalization_manifest_still_accepts_baseline_override(self) -> None:
        contract = {
            "source_journal_sha256": "b" * 64,
            "base_config_sha256": "c" * 64,
            "top_n": 6,
            "selection_policy": "feasible_cost",
            "ppl": {"chunks": 64, "window": 1024},
            "gates": {
                "max_ppl_drift": 0.005,
                "max_keyword_rate": 2 / 136,
                "max_keywords": 2,
                "keyword_total": 136,
                "keyword_near_gate_extra": 1,
                "balanced_srg_gate": None,
                "balanced_removal_fraction": 0.8,
                "baseline_srg_override": 0.5,
            },
            "overrides": None,
        }
        manifest = {
            "version": 1,
            "status": "prepared",
            "source_journal_sha256": "b" * 64,
            "base_config_sha256": "c" * 64,
            "top_n": 6,
            "selection_policy": "feasible_cost",
            "ppl": {"chunks": 64, "window": 1024},
            "gates": {
                "max_ppl_drift": 0.005,
                "max_keyword_rate": 2 / 136,
                "max_keywords": 2,
                "keyword_total": 136,
                "keyword_near_gate_extra": 1,
                "balanced_srg_gate": None,
                "balanced_removal_fraction": 0.8,
                "baseline_srg": 0.5,
            },
            "finalization_overrides": None,
        }
        self.assertTrue(controller.finalization_manifest_matches(manifest, contract))
        manifest["gates"]["baseline_srg"] = 0.4
        self.assertFalse(controller.finalization_manifest_matches(manifest, contract))

    def test_multilingual_v3_finalist_prepare_command_excludes_legacy_flags(self) -> None:
        args = SimpleNamespace(
            finalist_top_n=6,
            finalist_selection_policy="feasible_cost",
            recheck_ppl_chunks=64,
            recheck_ppl_window=1024,
            max_ppl_drift=0.005,
            max_keywords=2,
            keyword_total=136,
            keyword_near_gate_extra=1,
            balanced_srg_gate=0.1,
            balanced_removal_fraction=0.8,
            multilingual_v3_enabled=True,
        )
        command = controller.build_finalist_prepare_command(
            args,
            finalist_script=Path("finalist_recheck.py"),
            source_journal=Path("journal.jsonl"),
            base_config=Path("config.toml"),
            finalist_dir=Path("finalist_recheck"),
            devices=["0", "2"],
        )

        for legacy_flag in (
            "--ppl-chunks",
            "--ppl-window",
            "--max-ppl-drift",
            "--max-keywords",
            "--keyword-total",
            "--keyword-near-gate-extra",
            "--balanced-srg-gate",
        ):
            self.assertNotIn(legacy_flag, command)
        self.assertIn("--balanced-removal-fraction", command)
        self.assertEqual(
            command[command.index("--devices") + 1 : command.index("--devices") + 3],
            ["0", "2"],
        )

        args.multilingual_v3_enabled = False
        legacy = controller.build_finalist_prepare_command(
            args,
            finalist_script=Path("finalist_recheck.py"),
            source_journal=Path("journal.jsonl"),
            base_config=Path("config.toml"),
            finalist_dir=Path("finalist_recheck"),
            devices=["0", "2"],
        )
        self.assertIn("--ppl-chunks", legacy)
        self.assertIn("--balanced-srg-gate", legacy)

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

        self.assertEqual(config["multilingual_search"]["dataset_root"], root.as_posix())
        self.assertEqual(config["multilingual_search"]["split_root"], split.as_posix())

    def test_multilingual_preparation_commands_use_all_devices_and_frozen_pools(
        self,
    ) -> None:
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
            with (
                redirect_stdout(output),
                patch.object(controller.subprocess, "run") as run,
            ):
                result = controller.prepare_multilingual_run_runtime(
                    config,
                    executable=Path("hereticMOE.exe"),
                    base_config=config_path,
                    run_root=root / "run",
                    devices=["0", "1"],
                    dry_run=True,
                )

        self.assertEqual(result["status"], "DRY_RUN")
        self.assertIn("multilingual_geometry_prepare", output.getvalue())
        self.assertIn("multilingual_runtime_prepare", output.getvalue())
        run.assert_not_called()

    def test_frozen_run_contract_links_every_prepared_component(self) -> None:
        runtime = {
            "dataset_contract_sha256": "a" * 64,
            "model_fingerprint": "d" * 64,
            "static_runtime_sha256": "e" * 64,
            "worker_runtime_contract_sha256": "f" * 64,
        }
        static = {
            "direction_package_sha256": "b" * 64,
            "srg_profile_sha256": "c" * 64,
            "schedule_contract_sha256": "9" * 64,
        }
        config = {
            "model": "example/model",
            "model_commit": "revision-a",
            "generation_backend": "compiled_static",
            "generation_prompt_bucket_multiple": 64,
            "generation_compile_mode": "default",
            "multilingual_search": {
                "enabled": True,
                "dataset_root": "F:/dataset",
                "schedule_seed": 42,
                "schedule_version": 2,
                "ordinary_max_new_tokens": 100,
                "final_max_new_tokens": 100,
                "max_safe_ppl_drift": 0.005,
                "max_safe_geometry_damage": 1.0,
                "max_language_instability": 1.0,
                "max_category_instability": 1.0,
            },
        }

        contract = controller.build_prepared_frozen_run_contract(
            config,
            runtime_manifest=runtime,
            static_manifest=static,
        )

        self.assertEqual(contract["dataset_contract_sha256"], "a" * 64)
        self.assertEqual(contract["model_fingerprint_sha256"], "d" * 64)
        self.assertEqual(contract["map_sha256"], "b" * 64)
        self.assertEqual(contract["srg_profile_sha256"], "c" * 64)
        self.assertEqual(
            contract["metric_contract"]["worker_runtime_contract_sha256"],
            "f" * 64,
        )
        self.assertEqual(contract["metric_contract"]["static_runtime_sha256"], "e" * 64)
        self.assertEqual(
            contract["metric_contract"]["schedule_contract_sha256"], "9" * 64
        )

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
            with (
                self.subTest(conflicting=conflicting),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
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

    def test_queue_contract_separates_base_trial_records_from_completed_work(
        self,
    ) -> None:
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

    def test_queue_verifier_rejects_two_complete_attempts_for_one_permit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            journal = root / "journal.log"
            queue = TrialWorkQueue(root / "queue.sqlite3")
            queue.initialize(
                first_task_id=0,
                task_count=1,
                exploration_task_count=1,
                target_trial_count=1,
                tpe_concurrency=1,
                journal_base_trial_count=0,
                journal_base_complete_count=0,
                journal_base_size_bytes=0,
                journal_base_sha256=sha256().hexdigest(),
            )
            storage = JournalStorage(
                JournalFileBackend(
                    str(journal),
                    lock_obj=JournalFileOpenLock(str(journal)),
                )
            )
            study = optuna.create_study(storage=storage, study_name="heretic")

            first_item = queue.claim("gpu-0")
            self.assertIsNotNone(first_item)
            first = study.ask()
            first.set_user_attr("queue_task_id", 0)
            first.set_user_attr("queue_attempt", 1)
            first.set_user_attr("queue_task_kind", "random")
            first.set_user_attr("queue_worker_id", "gpu-0")
            study.tell(first, 1.0)
            queue.fail(first_item, error_type="legacy_callback_failure", retry=True)

            second_item = queue.claim("gpu-1")
            self.assertIsNotNone(second_item)
            second = study.ask()
            second.set_user_attr("queue_task_id", 0)
            second.set_user_attr("queue_attempt", 2)
            second.set_user_attr("queue_task_kind", "random")
            second.set_user_attr("queue_worker_id", "gpu-1")
            study.tell(second, 2.0)
            queue.finish(
                second_item, trial_number=second.number, trial_state="COMPLETE"
            )

            valid, reason = controller.verify_queue_against_journal(
                queue,
                journal,
                target_trial_count=1,
                tpe_concurrency=1,
            )

        self.assertFalse(valid)
        self.assertEqual(reason, "duplicate_complete:0")

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

    def test_worker_completion_counts_support_arbitrary_gpu_count(self) -> None:
        records = [
            SimpleNamespace(worker_id="gpu-0", state="complete"),
            SimpleNamespace(worker_id="gpu-2", state="complete"),
            SimpleNamespace(worker_id="gpu-2", state="claimed"),
            SimpleNamespace(worker_id="gpu-7", state="complete"),
        ]

        self.assertEqual(
            controller.worker_completion_counts(records),
            {"gpu-0": 1, "gpu-2": 1, "gpu-7": 1},
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
            from heretic.multilingual_finalists import select_multilingual_winners

            measured = [
                {
                    "trial_number": 2,
                    "source_trial_index": 403,
                    "source_trial_number": 402,
                    "feasible": True,
                    "removal": 0.8,
                    "preservation_loss": 0.2,
                    "safe_ppl_drift": 0.01,
                    "safe_geometry_damage": 0.02,
                    "worst_language": 0.4,
                    "worst_category": 0.3,
                    "final_holdout_removal": 0.7,
                },
                {
                    "trial_number": 4,
                    "source_trial_index": 542,
                    "source_trial_number": 541,
                    "feasible": True,
                    "removal": 1.0,
                    "preservation_loss": 0.5,
                    "safe_ppl_drift": 0.03,
                    "safe_geometry_damage": 0.04,
                    "worst_language": 0.5,
                    "worst_category": 0.4,
                    "final_holdout_removal": 0.9,
                },
            ]
            report = select_multilingual_winners(
                measured,
                balanced_removal_fraction=0.8,
            )
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = controller.load_valid_winners_report(path)

        self.assertEqual(loaded, report)

    def test_load_valid_winners_report_rejects_tampered_multilingual_winner(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "winners.json"
            from heretic.multilingual_finalists import select_multilingual_winners

            measured = [
                {
                    "trial_number": number,
                    "source_trial_index": number + 1,
                    "source_trial_number": number,
                    "feasible": True,
                    "removal": removal,
                    "preservation_loss": preservation,
                    "safe_ppl_drift": 0.01,
                    "safe_geometry_damage": 0.02,
                    "worst_language": 0.4,
                    "worst_category": 0.3,
                    "final_holdout_removal": removal,
                }
                for number, removal, preservation in (
                    (10, 0.8, 0.1),
                    (11, 1.0, 0.5),
                    (12, 0.7, 0.2),
                )
            ]
            report = select_multilingual_winners(
                measured,
                balanced_removal_fraction=0.8,
            )
            report["winners"]["Max"] = measured[2]
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = controller.load_valid_winners_report(path)

        self.assertIsNone(loaded)

    def test_console_safe_text_replaces_glyphs_missing_from_cp1251(self) -> None:
        rendered = controller.console_safe_text("GPU 0 | 25% ▏", "cp1251")

        self.assertEqual(rendered, "GPU 0 | 25% ?")


if __name__ == "__main__":
    unittest.main()
