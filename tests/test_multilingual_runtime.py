from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch

from heretic.clean_reference_archive import build_clean_reference_archive
from heretic.config import SelectionPolicy, Settings
from heretic.language_map_data import GeometryRow, text_free_row_index
from heretic.language_map_directions import DirectionMapProfile, write_direction_map_package
from heretic.multilingual_contract import MultilingualDatasetBundle
from heretic.multilingual_runtime import (
    apply_multilingual_search_mode,
    load_multilingual_search_evaluator,
    resolve_srg_runtime_contract,
)
from heretic.multilingual_search_evaluator import MultilingualConstraintContract
from heretic.trial_language_schedule import materialize_trial_language_schedule


class _ReferenceModel:
    def get_response_artifacts_with_prefill_residuals(self, prompts, **kwargs):
        count = len(prompts)
        return (
            [f"clean-{index}" for index in range(count)],
            [[index + 1] for index in range(count)],
            torch.ones((count, 2, 2)),
        )

    def get_conditional_nll(self, prompts, targets):
        return [1.0] * len(prompts)


class _UnusedScorer:
    pass


def _rows(tmp_path: Path) -> list[GeometryRow]:
    return [
        GeometryRow(
            canonical_id="S1",
            row_id="EN-S1",
            language="en",
            direction="safe",
            category_id="C01",
            prompt="private-safe",
            source_path=tmp_path / "private.jsonl",
            source_line=1,
        ),
        GeometryRow(
            canonical_id="U1",
            row_id="EN-U1",
            language="en",
            direction="unsafe",
            category_id="C01",
            prompt="private-unsafe",
            source_path=tmp_path / "private.jsonl",
            source_line=2,
        ),
    ]


def _prepare_runtime(tmp_path: Path) -> tuple[MultilingualDatasetBundle, Path]:
    rows = _rows(tmp_path)
    dataset_sha = "a" * 64
    bundle = MultilingualDatasetBundle(
        direction_rows=(),
        trial_rows=tuple(rows),
        search_rows=(),
        final_rows=(),
        manifest={"status": "PASS", "contract_sha256": dataset_sha},
    )
    runtime_root = tmp_path / "runtime"
    directions = runtime_root / "clean_map" / "directions"
    profile = DirectionMapProfile(
        consensus_refusal_direction=torch.ones((2, 2)),
        per_layer_direction=torch.ones((2, 2)),
        language_subspace=torch.zeros((2, 0, 2)),
        language_ranks=torch.zeros(2, dtype=torch.int64),
        language_explained_variance=torch.zeros((2, 0)),
        category_branch_directions=torch.zeros((2, 1, 2)),
        category_ids=("C01",),
        layer_reliability=torch.ones(2),
        recommended_layer_bounds=(0, 1),
        diagnostics={"schema_version": 1},
    )
    direction_manifest = write_direction_map_package(profile, directions)
    materialize_trial_language_schedule(
        text_free_row_index(rows),
        output_dir=runtime_root / "study" / "schedule",
        languages=("en",),
        seed=7,
        total_trials=1,
        expected_per_direction=1,
    )
    build_clean_reference_archive(
        model=_ReferenceModel(),
        rows=rows,
        refusal_direction=profile.consensus_refusal_direction,
        output_dir=runtime_root / "clean_trial_reference",
        dataset_contract_sha256=dataset_sha,
        direction_sha256=direction_manifest["package_sha256"],
        model_fingerprint="fake-model-v1",
        max_response_length=512,
        batch_size=2,
    )
    calibration_dir = runtime_root / "srg_calibration"
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "calibration_profile.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "PASS",
                "rows": 660,
                "scale": [1.0] * 660,
                "weight": [1.0] * 660,
                "global_scale": 1.0,
                "group_scale": {},
                "group_weight": {},
            }
        ),
        encoding="utf-8",
    )
    return bundle, runtime_root


def test_runtime_loader_verifies_and_wires_all_frozen_packages(tmp_path: Path) -> None:
    bundle, runtime_root = _prepare_runtime(tmp_path)

    evaluator, manifest = load_multilingual_search_evaluator(
        bundle=bundle,
        runtime_root=runtime_root,
        model=object(),
        srg_scorer=_UnusedScorer(),
        constraints=MultilingualConstraintContract(),
        expected_per_direction=1,
        expected_languages=("en",),
    )

    assert evaluator.get_objective_names() == ["Removal", "Preservation loss"]
    assert manifest["status"] == "PASS"
    assert manifest["dataset_contract_sha256"] == "a" * 64
    assert manifest["schedule_trials"] == 1
    serialized = json.dumps(manifest, sort_keys=True)
    assert "private-safe" not in serialized
    assert "private-unsafe" not in serialized


def test_runtime_loader_rejects_clean_archive_from_another_dataset(tmp_path: Path) -> None:
    bundle, runtime_root = _prepare_runtime(tmp_path)
    bundle.manifest["contract_sha256"] = "b" * 64

    try:
        load_multilingual_search_evaluator(
            bundle=bundle,
            runtime_root=runtime_root,
            model=object(),
            srg_scorer=_UnusedScorer(),
            constraints=MultilingualConstraintContract(),
            expected_per_direction=1,
            expected_languages=("en",),
        )
    except ValueError as error:
        assert "dataset contract" in str(error)
    else:
        raise AssertionError("cross-dataset clean references must be rejected")


def test_multilingual_mode_cannot_fall_back_to_legacy_136_objectives(tmp_path: Path) -> None:
    settings = Settings(
        model="example/model",
        max_response_length=48,
        primary_objective="Sparse refusal geometry",
        selection_policy="feasible_cost",
        selection_score_targets={"Sparse refusal geometry": -0.0088},
        selection_score_weights={"Sparse refusal geometry": 344.0},
        multilingual_search={
            "enabled": True,
            "dataset_root": str(tmp_path / "dataset"),
            "runtime_root": str(tmp_path / "runtime"),
        },
    )

    apply_multilingual_search_mode(settings)

    assert settings.max_response_length == 512
    assert settings.primary_objective == "Removal"
    assert settings.selection_policy == SelectionPolicy.FEASIBLE_DIVERSE
    assert settings.selection_score_targets == {}
    assert settings.selection_score_weights == {}


def test_srg_runtime_contract_uses_only_pinned_660_files(tmp_path: Path) -> None:
    root = tmp_path / "srg_calibration"
    private = root / "private"
    private.mkdir(parents=True)
    prototypes = root / "prototypes.jsonl"
    prompts = private / "evaluation_prompts.jsonl"
    prototypes.write_bytes(b"prototype-bank\n")
    prompts.write_bytes(b"calibration-660\n")
    manifest = {
        "status": "PASS",
        "prototype_path": str(prototypes),
        "prototype_sha256": hashlib.sha256(prototypes.read_bytes()).hexdigest(),
        "prompt_path": str(prompts),
        "prompt_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "prompt_rows": 660,
        "top_k": 5,
        "min_df": 2,
        "max_response_length": 128,
        "validate_prompt_alignment": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_srg_runtime_contract(root)

    assert resolved["prompt_rows"] == 660
    assert resolved["prototype_path"] == prototypes.resolve()
    assert resolved["prompt_path"] == prompts.resolve()
    assert resolved["validate_prompt_alignment"] is False
