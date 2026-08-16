from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from heretic.launch_config import (
    LaunchOverrides,
    build_internal_settings,
    effective_resume_contract_sha256,
    load_effective_launch_config,
    resolve_run_root,
    write_effective_config_bundle,
)


def _write_config(path: Path) -> None:
    payload = {
        "version": 1,
        "model": {"path": "example/model"},
        "run": {
            "root": "runs/example",
            "target_trials": 600,
            "exploration_trials": 120,
            "seed": 42,
            "post_search": "export",
            "incompatible_contract": "archive",
        },
        "devices": {
            "mode": "auto",
            "include": [],
            "max_workers": 2,
            "min_free_fraction": 0.7,
            "min_free_gib": 4.0,
        },
        "data": {
            "dataset_root": "datasets/heretic_moe_4lang_v4",
            "split_root": "datasets/heretic_moe_4lang_v4",
            "languages": ["en", "ru", "zh", "ja"],
            "direction_rows_per_cell": 1000,
            "trial_rows_per_cell": 400,
            "final_rows_per_cell": 200,
        },
        "generation": {
            "ordinary_max_new_tokens": 100,
            "final_max_new_tokens": 100,
            "batch_size": "auto",
            "conditional_nll_batch_size": "auto",
            "max_batch_size": 4096,
            "backend": "compiled_static",
            "vram_headroom_fraction": 0.10,
        },
        "search": {
            "schedule_capacity": 1000,
            "schedule_seed": 20260811,
            "schedule_version": 2,
            "direction_modes": ["global", "per_layer"],
            "save_responses": True,
        },
        "metrics": {
            "contract_version": "multilingual_v4",
            "max_safe_ppl_drift": 0.005,
            "max_safe_geometry_damage": 1.0,
            "max_language_instability": 1.0,
            "max_category_instability": 1.0,
            "max_empty_rate": 0.0,
            "max_truncated_rate": 0.0,
            "max_safe_d_to_r_rate": 0.0,
        },
        "finalists": {
            "top_n": 6,
            "balanced_removal_fraction": 0.8,
            "export_roles": ["Balanced", "Max"],
        },
        "geometry": {
            "trajectory": True,
            "capture_trials": True,
            "render_html": True,
        },
        "recovery": {
            "worker_heartbeat_seconds": 15,
            "worker_timeout_seconds": 180,
            "max_restarts_per_gpu": 2,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_repository_example_contains_the_complete_public_schema() -> None:
    source = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_effective_launch_config(source, LaunchOverrides())

    assert config.version == 1
    assert config.run.target_trials == 600
    assert config.generation.ordinary_max_new_tokens == 100
    assert config.data.languages == ["en", "ru", "zh", "ja"]
    assert config.data.direction_rows_per_cell == 1000
    assert config.data.trial_rows_per_cell == 400
    assert config.data.final_rows_per_cell == 200
    assert config.metrics.contract_version == "multilingual_v4"
    assert config.finalists.top_n == 6
    assert config.geometry.render_html is True


def test_loads_public_yaml_and_applies_cli_overrides_last(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)

    config = load_effective_launch_config(
        source,
        LaunchOverrides(
            model="override/model",
            run_root=tmp_path / "override-run",
            devices="1,3",
            target_trials=800,
            exploration_trials=160,
            post_search="recheck",
            incompatible_contract="new_run",
        ),
    )

    assert config.model.path == "override/model"
    assert config.run.root == tmp_path / "override-run"
    assert config.devices.mode == "include"
    assert config.devices.include == ["1", "3"]
    assert config.devices.selection_specification() == "1,3"
    assert config.run.target_trials == 800
    assert config.run.exploration_trials == 160
    assert config.run.post_search == "recheck"
    assert config.run.incompatible_contract == "new_run"
    assert config.run.seed == 42


def test_public_yaml_rejects_legacy_settings_block(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["settings"] = {"model": "legacy/model"}
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_effective_launch_config(source, LaunchOverrides())


def test_public_yaml_requires_dataset_root(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["data"].pop("dataset_root")
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="dataset_root"):
        load_effective_launch_config(source, LaunchOverrides())


def test_unknown_public_yaml_key_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["mystery"] = True
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_effective_launch_config(source, LaunchOverrides())


def test_public_yaml_rejects_removed_per_model_srg_calibration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["data"]["srg_calibration_source"] = "legacy/search_unsafe"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="srg_calibration_source"):
        load_effective_launch_config(source, LaunchOverrides())


def test_exploration_cannot_exceed_target(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)

    with pytest.raises(ValidationError):
        load_effective_launch_config(
            source,
            LaunchOverrides(target_trials=100, exploration_trials=120),
        )


def test_exploration_cannot_equal_target(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)

    with pytest.raises(ValidationError):
        load_effective_launch_config(
            source,
            LaunchOverrides(target_trials=120, exploration_trials=120),
        )


def test_public_yaml_maps_to_internal_settings_without_exposing_settings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())

    internal = build_internal_settings(config)

    assert internal.model == "example/model"
    assert internal.seed == 42
    assert internal.n_trials == 600
    assert internal.n_startup_trials == 120
    assert internal.batch_size == 0
    assert internal.conditional_nll_batch_size == 0
    assert internal.max_batch_size == 4096
    assert internal.max_response_length == 100
    assert internal.generation_backend.value == "compiled_static"
    assert internal.generation_batch_target_headroom_fraction == 0.10
    assert internal.batch_size_vram_headroom_fraction == 0.10
    assert internal.save_trial_responses is True
    assert internal.geometry_capture_evaluation is True
    assert internal.geometry_trajectory_package
    assert internal.geometry_render_html is True
    multilingual = internal.multilingual_search
    assert multilingual.enabled is True
    assert multilingual.dataset_root.endswith("datasets/heretic_moe_4lang_v4")
    assert multilingual.split_root.endswith("datasets/heretic_moe_4lang_v4")
    assert multilingual.languages == ["en", "ru", "zh", "ja"]
    assert multilingual.direction_rows_per_cell == 1000
    assert multilingual.trial_rows_per_cell == 400
    assert multilingual.final_rows_per_cell == 200
    assert not hasattr(multilingual, "srg_calibration_source")
    assert multilingual.ordinary_max_new_tokens == 100
    assert multilingual.final_max_new_tokens == 100
    assert multilingual.schedule_capacity == 1000
    assert multilingual.schedule_seed == 20260811
    assert multilingual.schedule_version == 2
    assert multilingual.max_empty_response_rate == 0.0
    assert multilingual.max_truncated_response_rate == 0.0
    assert multilingual.max_safe_d_to_r_rate == 0.0


def test_public_yaml_maps_explicit_generation_and_nll_batches(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["generation"]["batch_size"] = 176
    payload["generation"]["conditional_nll_batch_size"] = 20
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")

    internal = build_internal_settings(
        load_effective_launch_config(source, LaunchOverrides())
    )

    assert internal.batch_size == 176
    assert internal.conditional_nll_batch_size == 20


def test_internal_settings_ignore_legacy_environment_and_toml_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    (tmp_path / "config.toml").write_text(
        "generation_prompt_bucket_multiple = 777\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERETIC_GENERATION_PROMPT_BUCKET_MULTIPLE", "999")
    config = load_effective_launch_config(source, LaunchOverrides())

    internal = build_internal_settings(config)

    assert internal.generation_prompt_bucket_multiple == 32


def test_writes_original_effective_yaml_internal_toml_and_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    output = tmp_path / "run"

    bundle = write_effective_config_bundle(source, config, output)

    assert bundle.original_yaml.name == "config.yaml"
    assert bundle.original_yaml.read_bytes() == source.read_bytes()
    effective = yaml.safe_load(bundle.effective_yaml.read_text(encoding="utf-8"))
    assert effective["model"]["path"] == "example/model"
    assert "settings" not in effective
    internal_toml = bundle.effective_toml.read_text(encoding="utf-8")
    assert 'model = "example/model"' in internal_toml
    assert "save_trial_responses = true" in internal_toml
    assert "max_batch_size = 4096" in internal_toml
    assert "geometry_trajectory_package" in internal_toml
    assert "geometry_render_html" in internal_toml
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["config_sha256"]
    assert manifest["launch_contract_sha256"]
    assert "resume_contract_sha256" not in manifest
    assert manifest["marker"] == "Heretic-MOE-run-root"
    assert manifest["files"]["config.yaml"]["sha256"]


def test_target_extension_keeps_resume_contract_compatible(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    first = load_effective_launch_config(source, LaunchOverrides())
    extended = load_effective_launch_config(
        source,
        LaunchOverrides(target_trials=1000, post_search="recheck"),
    )

    assert effective_resume_contract_sha256(first) == effective_resume_contract_sha256(
        extended
    )


def test_dataset_or_generation_drift_changes_resume_contract(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    first = load_effective_launch_config(source, LaunchOverrides())
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["generation"]["ordinary_max_new_tokens"] = 128
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")
    changed = load_effective_launch_config(source, LaunchOverrides())

    assert effective_resume_contract_sha256(first) != effective_resume_contract_sha256(
        changed
    )


def test_archive_policy_moves_the_entire_incompatible_run_root(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    root = config.run.root
    root.mkdir(parents=True)
    (root / "old-journal.jsonl").write_text("journal", encoding="utf-8")

    resolution = resolve_run_root(root, config, "archive")

    assert resolution.run_root == root
    assert resolution.archived_root is not None
    assert not (root / "old-journal.jsonl").exists()
    assert (resolution.archived_root / "old-journal.jsonl").read_text() == "journal"


def test_new_run_policy_picks_next_free_suffix(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    root = config.run.root
    root.mkdir(parents=True)
    (root / "legacy-journal.jsonl").write_text("legacy", encoding="utf-8")
    root.with_name(root.name + "_v2").mkdir()

    resolution = resolve_run_root(root, config, "new_run")

    assert resolution.run_root == root.with_name(root.name + "_v3")
    assert root.is_dir()


def test_replace_refuses_unrecognized_directory(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    root = config.run.root
    root.mkdir(parents=True)
    (root / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="recognized Heretic-MOE run root"):
        resolve_run_root(root, config, "replace")

    assert (root / "user-file.txt").is_file()


def test_replace_removes_only_a_recognized_incompatible_run(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    original = load_effective_launch_config(source, LaunchOverrides())
    root = original.run.root
    write_effective_config_bundle(source, original, root)
    (root / "journal.jsonl").write_text("old", encoding="utf-8")
    payload = original.model_dump(mode="json")
    payload["run"]["seed"] = 99
    changed = original.__class__.model_validate(payload)

    resolution = resolve_run_root(root, changed, "replace")

    assert resolution.run_root == root
    assert resolution.replaced is True
    assert not root.exists()


def test_fail_policy_preserves_incompatible_root(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    root = config.run.root
    root.mkdir(parents=True)
    marker = root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incompatible"):
        resolve_run_root(root, config, "fail")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_dry_run_resolution_does_not_archive_or_replace_files(tmp_path: Path) -> None:
    source = tmp_path / "config.yaml"
    _write_config(source)
    config = load_effective_launch_config(source, LaunchOverrides())
    root = config.run.root
    root.mkdir(parents=True)
    marker = root / "old-journal.jsonl"
    marker.write_text("journal", encoding="utf-8")

    resolution = resolve_run_root(root, config, "archive", mutate=False)

    assert resolution.run_root == root
    assert resolution.archived_root is not None
    assert marker.read_text(encoding="utf-8") == "journal"
    assert not resolution.archived_root.exists()
