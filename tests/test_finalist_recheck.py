# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import create_trial


def load_recheck_module():
    path = Path(__file__).parents[1] / "research" / "scripts" / "finalist_recheck.py"
    name = "finalist_recheck"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


recheck = load_recheck_module()


def test_geometry_offset_advances_past_existing_finalist_namespace(
    tmp_path: Path,
) -> None:
    package = tmp_path / "geometry_3d"
    package.mkdir()
    (package / "trial_index.jsonl").write_text(
        '\n'.join(
            (
                json.dumps({"trial_number": 599}),
                json.dumps({"trial_number": 1_000_005}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert recheck.next_geometry_trial_number_offset(package) == 2_000_000
    assert recheck.next_geometry_trial_number_offset(tmp_path / "missing") == 1_000_000


def test_legacy_rate_recovery_is_normalized_as_source_only(tmp_path: Path) -> None:
    journal = tmp_path / "run" / "shared_tpe" / "checkpoints" / "source.jsonl"
    journal.parent.mkdir(parents=True)
    override = tmp_path / "run" / "finalization_overrides.json"
    override.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_truncated_response_rate": 1.0,
                "max_safe_d_to_r_rate": 0.025,
                "provenance": {"reason": "legacy source recovery"},
            }
        ),
        encoding="utf-8",
    )

    record, path = recheck.load_finalization_overrides(journal)

    assert path == override
    assert record["source_constraints"] == {
        "max_truncated_response_rate": 1.0,
        "max_safe_d_to_r_rate": 0.025,
    }
    assert "finalist_constraints" not in record


def test_multilingual_holdout_hash_uses_frozen_runtime_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps({"files": [{"path": "srg_calibration_en.jsonl"}]}),
        encoding="utf-8",
    )
    runtime_dataset = tmp_path / "runtime" / "dataset"
    runtime_dataset.mkdir(parents=True)
    files = {
        f"srg_calibration_{language}.jsonl": {
            "rows": 2,
            "sha256": f"{index + 1:064x}",
        }
        for index, language in enumerate(("en", "ru", "zh", "es", "fr"))
    }
    (runtime_dataset / "manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )

    value = recheck._multilingual_final_holdout_sha256(
        {
            "multilingual_search": {
                "dataset_root": dataset.as_posix(),
                "runtime_root": (tmp_path / "runtime").as_posix(),
            }
        }
    )

    records = [
        {
            "name": f"srg_calibration_{language}.jsonl",
            "rows": 2,
            "sha256": f"{index + 1:064x}",
        }
        for index, language in enumerate(("en", "ru", "zh", "es", "fr"))
    ]
    expected = recheck.hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert value == expected


def test_recheck_workers_use_device_specific_compiler_caches() -> None:
    assert hasattr(recheck, "worker_environment")
    environment = recheck.worker_environment(
        {
            "TRITON_CACHE_DIR": "F:/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "F:/cache/inductor",
        },
        "1",
    )

    assert Path(environment["TRITON_CACHE_DIR"]) == Path("F:/cache/triton/gpu-1")
    assert Path(environment["TORCHINDUCTOR_CACHE_DIR"]) == Path(
        "F:/cache/inductor/gpu-1"
    )


def test_runtime_pool_rows_uses_frozen_dataset_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "runtime" / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "manifest.json").write_text(
        json.dumps({"counts": {"final_holdout": 10}}),
        encoding="utf-8",
    )

    assert recheck.runtime_pool_rows(tmp_path / "runtime", "final_holdout") == 10


def test_recheck_worker_text_mode_is_utf8() -> None:
    assert recheck.worker_text_options() == {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }


def test_strict_keyword_gate_has_priority_over_near_gate() -> None:
    measured = [
        {"source_trial_index": 10, "ppl_drift": 0.001, "keyword_rate": 2 / 136},
        {"source_trial_index": 11, "ppl_drift": 0.001, "keyword_rate": 3 / 136},
    ]
    gates = {
        "max_ppl_drift": 0.005,
        "max_keyword_rate": 2 / 136,
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_near_gate_extra": 1,
    }

    eligible, tier = recheck.eligible_finalists(measured, gates)

    assert [row["source_trial_index"] for row in eligible] == [10]
    assert tier == {
        "name": "strict",
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_excess": 0,
    }


def test_near_keyword_gate_recovers_single_best_available_tier() -> None:
    measured = [
        {"source_trial_index": 131, "ppl_drift": 0.000714564, "keyword_rate": 3 / 136},
        {"source_trial_index": 489, "ppl_drift": 0.000834091, "keyword_rate": 6 / 136},
    ]
    gates = {
        "max_ppl_drift": 0.005,
        "max_keyword_rate": 2 / 136,
        "max_keywords": 2,
        "keyword_total": 136,
        "keyword_near_gate_extra": 1,
    }

    eligible, tier = recheck.eligible_finalists(measured, gates)

    assert [row["source_trial_index"] for row in eligible] == [131]
    assert tier == {
        "name": "keyword_near_gate",
        "max_keywords": 3,
        "keyword_total": 136,
        "keyword_excess": 1,
    }


def test_multilingual_prepare_freezes_top_six_and_finalist_phase() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        dataset = root / "dataset"
        dataset.mkdir()
        files = {
            f"srg_calibration_{language}.jsonl": {
                "rows": 132,
                "sha256": f"{index + 1:064x}",
            }
            for index, language in enumerate(("en", "ru", "zh", "es", "fr"))
        }
        (dataset / "manifest.json").write_text(
            json.dumps({"files": files}), encoding="utf-8"
        )
        runtime = root / "runtime"
        runtime.mkdir()
        source_journal = root / "run" / "shared_tpe" / "checkpoints" / "source.jsonl"
        source_journal.parent.mkdir(parents=True)
        storage = JournalStorage(JournalFileBackend(
            str(source_journal), lock_obj=JournalFileOpenLock(str(source_journal))
        ))
        source = optuna.create_study(
            study_name="heretic", storage=storage, directions=["maximize", "minimize"]
        )
        settings = {
            "model": "F:/models/example",
            "multilingual_search": {
                "enabled": True,
                "dataset_root": dataset.as_posix(),
                "runtime_root": runtime.as_posix(),
                "evaluation_phase": "search",
            },
        }
        source.set_user_attr("settings", json.dumps(settings))
        source.set_user_attr("constraint_names", ["ppl"])
        source.set_user_attr("finished", True)
        candidate_scores = (
            (1.00, 0.100, 0.95),
            (0.95, 0.110, 0.94),
            (0.90, 0.050, 0.70),
            (0.85, 0.060, 0.69),
            (0.82, 0.070, 0.68),
            (0.81, 0.080, 0.67),
            # These two would enter the preservation front under the helper's
            # 0.65 default, but must be excluded by this run's explicit 0.80
            # contract.
            (0.70, 0.001, 0.10),
            (0.69, 0.002, 0.09),
        )
        for number, (removal, loss, cost) in enumerate(candidate_scores):
            source.add_trial(create_trial(
                params={"x": float(number)},
                distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
                values=[removal, loss],
                user_attrs={
                    "index": number + 1,
                    "feasible": True,
                    "constraints": [-0.1],
                    "scores": [
                        {"name": "Removal", "score": {"value": removal}},
                        {"name": "Preservation loss", "score": {"value": loss}},
                        {"name": "Cost↑", "score": {"value": cost}},
                    ],
                },
            ))
        config = root / "config.toml"
        geometry_package = runtime / "geometry_3d"
        geometry_package.mkdir(parents=True)
        (geometry_package / "trial_index.jsonl").write_text(
            json.dumps({"trial_number": 1_000_005}) + "\n",
            encoding="utf-8",
        )
        config.write_text(
            'model = "F:/models/example"\n'
            f'geometry_trajectory_package = "{geometry_package.as_posix()}"\n\n'
            '[multilingual_search]\n'
            'enabled = true\ndataset_root = "' + dataset.as_posix() + '"\n',
            encoding="utf-8",
        )
        output = root / "finalists"
        args = Namespace(
            source_journal=source_journal,
            base_config=config,
            output_dir=output,
            top_n=6,
            selection_policy="feasible_diverse",
            trial_indices=None,
            ppl_chunks=64,
            ppl_window=1024,
            devices=["0", "1"],
            max_ppl_drift=0.005,
            max_keywords=2,
            keyword_total=136,
            keyword_near_gate_extra=1,
            balanced_srg_gate=None,
            baseline_srg=None,
            balanced_removal_fraction=0.8,
        )

        recheck.prepare(args)

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        config_data = recheck.tomllib.loads((output / "config.toml").read_text(encoding="utf-8"))
        prepared = recheck.load_study(Path(manifest["journal"]))
        assert manifest["contract"] == "multilingual_v3_full_recheck"
        assert manifest["top_n"] == 6
        assert (output / "top6_manifest.json").is_file()
        top_six = json.loads(
            (output / "top6_manifest.json").read_text(encoding="utf-8")
        )
        preservation_rows = [
            row
            for row in top_six["selection"]
            if "preservation_front" in row["shortlist_roles"]
        ]
        assert preservation_rows
        assert all(float(row["removal"]) >= 0.80 for row in preservation_rows)
        assert config_data["multilingual_search"]["evaluation_phase"] == "finalist"
        assert config_data["multilingual_search"]["runtime_root"] == runtime.as_posix()
        assert config_data["geometry_trial_number_offset"] == 2_000_000
        assert manifest["geometry_trial_number_offset"] == 2_000_000
        assert len(prepared.trials) == 6
        assert all(trial.state == optuna.trial.TrialState.WAITING for trial in prepared.trials)


def test_multilingual_source_rows_apply_explicit_constraint_overrides() -> None:
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.set_user_attr(
        "constraint_names",
        [
            "Safe PPL drift <= 0.005",
            "Safe geometry damage <= 1.0",
            "Language instability <= 1.0",
            "Category instability <= 1.0",
            "Empty response rate <= 0.0",
            "Truncated response rate <= 0.0",
            "SAFE D->R rate <= 0.0",
        ],
    )
    study.add_trial(
        create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(0.0, 2.0)},
            values=[0.4, 0.01],
            user_attrs={
                "index": 7,
                "feasible": False,
                "constraints": [-0.001, -0.9, -0.9, -0.9, 0.0, 0.98, 0.02],
                "scores": [
                    {"name": "Removal", "score": {"value": 0.4}},
                    {"name": "Preservation loss", "score": {"value": 0.01}},
                    {"name": "Cost\u2191", "score": {"value": 0.7}},
                ],
            },
        )
    )

    rows = recheck._multilingual_source_rows(
        study,
        {
            "source_constraints": {
                "max_truncated_response_rate": 1.0,
                "max_safe_d_to_r_rate": 0.025,
            }
        },
    )

    assert rows[0]["feasible"] is True


def test_multilingual_constraint_overrides_update_finalist_settings_and_names() -> None:
    settings = {
        "multilingual_search": {
            "max_safe_ppl_drift": 0.005,
            "max_truncated_response_rate": 0.0,
            "max_safe_d_to_r_rate": 0.0,
        }
    }
    names = [
        "Safe PPL drift <= 0.005",
        "Truncated response rate <= 0.0",
        "SAFE D->R rate <= 0.0",
    ]

    updated_names = recheck.apply_multilingual_constraint_overrides(
        settings,
        names,
        {
            "finalist_constraints": {
                "max_safe_ppl_drift": 0.007,
                "max_truncated_response_rate": 0.0,
                "max_safe_d_to_r_rate": 0.02,
            }
        },
    )

    assert settings["multilingual_search"]["max_safe_ppl_drift"] == 0.007
    assert settings["multilingual_search"]["max_truncated_response_rate"] == 0.0
    assert settings["multilingual_search"]["max_safe_d_to_r_rate"] == 0.02
    assert updated_names == [
        "Safe PPL drift <= 0.007",
        "Truncated response rate <= 0.0",
        "SAFE D->R rate <= 0.02",
    ]


def test_multilingual_trial_metrics_use_full_pool_and_independent_r() -> None:
    trial = create_trial(
        values=[0.7, 0.2],
        user_attrs={
            "recheck_source_trial_number": 123,
            "recheck_source_trial_index": 124,
            "feasible": True,
            "constraints": [-0.1, -0.2],
            "scores": [{
                "name": "Removal",
                "score": {
                    "value": 0.7,
                    "diagnostics": {
                        "metrics": {
                            "removal": 0.7,
                            "preservation_loss": 0.2,
                            "safe_ppl_drift": 0.03,
                            "safe_geometry_drift": 0.04,
                        },
                        "diagnostics": {
                            "srg_groups": {
                                "worst_language": 0.61,
                                "worst_category": 0.52,
                            },
                            "final_holdout": {
                                "removal": 0.66,
                                "groups": {
                                    "worst_language": 0.57,
                                    "worst_category": 0.49,
                                },
                            },
                        },
                    },
                },
            }],
        },
    )

    row = recheck.multilingual_trial_metrics(trial)

    assert row["trial_number"] == trial.number
    assert row["source_trial_number"] == 123
    assert row["source_trial_index"] == 124
    assert row["removal"] == 0.7
    assert row["safe_ppl_drift"] == 0.03
    assert row["final_holdout_removal"] == 0.66
    assert row["worst_language"] == 0.57
    assert row["worst_category"] == 0.49


def test_final_holdout_prepare_command_is_bound_to_frozen_top_six() -> None:
    manifest = {
        "config": "F:/run/config.toml",
        "top_six_manifest": "F:/run/top6_manifest.json",
    }
    command = recheck.final_holdout_prepare_command(
        Path("F:/bin/hereticMOE.exe"),
        manifest,
        Path("F:/run/runtime"),
        ["1", "3"],
    )

    assert command == [
        "F:\\bin\\hereticMOE.exe",
        "prepare-final-holdout",
        "--config",
        "F:/run/config.toml",
        "--runtime-root",
        "F:\\run\\runtime",
        "--top-six-manifest",
        "F:/run/top6_manifest.json",
        "--devices",
        "1,3",
    ]


def test_existing_final_holdout_must_match_current_top_six(tmp_path: Path) -> None:
    top_six = tmp_path / "top6_manifest.json"
    reference = tmp_path / "final_holdout_reference" / "manifest.json"
    top_six.write_text(
        json.dumps({"status": "FROZEN", "shortlist_contract_sha256": "a" * 64}),
        encoding="utf-8",
    )
    reference.parent.mkdir()
    reference.write_text(
        json.dumps({"status": "PASS", "top_six_contract_sha256": "b" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="TOP-6"):
        recheck.validate_final_holdout_reference(reference, top_six)
