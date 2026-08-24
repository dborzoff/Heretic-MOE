from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from heretic.language_map_data import GeometryRow
from heretic.language_map_directions import (
    DirectionMapProfile,
    write_direction_map_package,
)
from heretic.multilingual_contract import MultilingualDatasetBundle
from heretic.multilingual_prepare import (
    fingerprint_local_model,
    freeze_direction_package,
    prepare_clean_reference_runtime,
    prepare_static_multilingual_runtime,
)
from heretic.multilingual_runtime import resolve_srg_runtime_contract
from heretic.trial_language_schedule import load_trial_language_schedule


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_source(tmp_path: Path) -> Path:
    profile = DirectionMapProfile(
        consensus_refusal_direction=torch.tensor([[0.0, 1.0]]),
        per_layer_direction=torch.tensor([[0.0, 0.8]]),
        language_subspace=torch.zeros((1, 2, 2)),
        language_ranks=torch.tensor([1]),
        language_explained_variance=torch.tensor([0.9]),
        category_branch_directions=torch.tensor([[[0.0, 1.0]]]),
        category_ids=("C01",),
        layer_reliability=torch.tensor([0.8]),
        recommended_layer_bounds=(0, 0),
        diagnostics={"schema_version": 1, "layers": []},
    )
    source = tmp_path / "direction-source"
    write_direction_map_package(profile, source)
    return source


def _bundle() -> MultilingualDatasetBundle:
    languages = ("en", "ru", "zh", "es", "fr")
    rows = []
    for direction, prefix in (("safe", "S"), ("unsafe", "U")):
        for canonical_number in range(5):
            canonical_id = f"{prefix}{canonical_number + 1:04d}"
            for language in languages:
                rows.append(
                    GeometryRow(
                        canonical_id=canonical_id,
                        row_id=f"{language.upper()}-{canonical_id}",
                        language=language,
                        direction=direction,
                        category_id="C01",
                        prompt="private",
                        source_path=Path("private.jsonl"),
                        source_line=canonical_number + 1,
                    )
                )
    return MultilingualDatasetBundle(
        direction_rows=tuple(rows),
        trial_rows=tuple(rows),
        final_rows=(),
        manifest={
            "schema_version": 1,
            "status": "PASS",
            "contract_sha256": "a" * 64,
            "counts": {"trial": len(rows)},
            "rows_per_cell": {"final_holdout": 132},
        },
    )


def _hard_soft_bundle() -> MultilingualDatasetBundle:
    languages = ("en", "ru", "zh", "ja")
    rows = []
    for direction, prefix, count in (("safe", "S", 8), ("unsafe", "U", 8)):
        for canonical_number in range(count):
            canonical_id = f"{prefix}{canonical_number + 1:04d}"
            behavior = (
                "safe"
                if direction == "safe"
                else "hard" if canonical_number < 4 else "soft"
            )
            for language in languages:
                rows.append(
                    GeometryRow(
                        canonical_id=canonical_id,
                        row_id=f"{language.upper()}-{canonical_id}",
                        language=language,
                        direction=direction,
                        category_id="C01",
                        prompt="private",
                        source_path=Path("private.jsonl"),
                        source_line=canonical_number + 1,
                        trial_behavior_class=behavior,
                    )
                )
    return MultilingualDatasetBundle(
        direction_rows=tuple(rows),
        trial_rows=tuple(rows),
        final_rows=(),
        manifest={
            "schema_version": 1,
            "status": "PASS",
            "contract_sha256": "b" * 64,
            "counts": {"trial": len(rows)},
            "rows_per_cell": {"final_holdout": 132},
        },
    )


def test_direction_package_is_copied_and_verified_portably(tmp_path: Path) -> None:
    source = _direction_source(tmp_path)
    destination = tmp_path / "runtime" / "clean_map" / "directions"

    manifest = freeze_direction_package(source, destination)

    assert manifest["status"] == "PASS"
    assert (
        manifest["package_sha256"]
        == json.loads((source / "manifest.json").read_text(encoding="utf-8"))[
            "package_sha256"
        ]
    )
    assert freeze_direction_package(source, destination) == manifest


def test_static_runtime_freezes_dataset_direction_builtin_srg_and_schedule(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"

    manifest = prepare_static_multilingual_runtime(
        bundle=_bundle(),
        direction_source=_direction_source(tmp_path),
        runtime_root=runtime,
        languages=("en", "ru", "zh", "es", "fr"),
        schedule_seed=17,
        schedule_capacity=10,
        expected_per_direction=5,
    )

    schedule_manifest, records = load_trial_language_schedule(
        runtime / "study" / "schedule"
    )
    assert manifest["status"] == "PASS"
    assert manifest["dataset_contract_sha256"] == "a" * 64
    assert schedule_manifest["trials"] == 10
    assert len(records) == 10
    srg = resolve_srg_runtime_contract(runtime / "srg_profile")
    assert srg["external_only"] is True
    assert not any(
        field in json.dumps(manifest).lower()
        for field in ('"prompt"', '"response"', '"answer"', '"text"')
    )
    assert (
        prepare_static_multilingual_runtime(
            bundle=_bundle(),
            direction_source=tmp_path / "direction-source",
            runtime_root=runtime,
            languages=("en", "ru", "zh", "es", "fr"),
            schedule_seed=17,
            schedule_capacity=10,
            expected_per_direction=5,
        )
        == manifest
    )


def test_static_runtime_preserves_hard_soft_trial_behavior(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime-hard-soft"

    prepare_static_multilingual_runtime(
        bundle=_hard_soft_bundle(),
        direction_source=_direction_source(tmp_path),
        runtime_root=runtime,
        languages=("en", "ru", "zh", "ja"),
        schedule_seed=23,
        schedule_capacity=4,
        expected_per_direction=8,
    )

    schedule_manifest, records = load_trial_language_schedule(
        runtime / "study" / "schedule"
    )
    assert schedule_manifest["schema_version"] == 3
    assert schedule_manifest["coverage_block_trials"] == 4
    assert schedule_manifest["behavior_rows_per_trial"] == {
        "hard": 4,
        "safe": 8,
        "soft": 4,
    }
    assert len(records) == 4


class _ReferenceModel:
    def get_response_artifacts_with_prefill_residuals(
        self, prompts, *, skip_special_tokens: bool
    ):
        assert skip_special_tokens is True
        count = len(prompts)
        return (
            ["private-response"] * count,
            [[11, 12]] * count,
            torch.tensor([[[0.0, 0.5]]] * count),
        )

    def get_conditional_nll(self, prompts, targets):
        assert len(prompts) == len(targets)
        return [0.25] * len(prompts)


def test_clean_reference_runtime_uses_full_trial_archive_once(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    bundle = _bundle()
    prepare_static_multilingual_runtime(
        bundle=bundle,
        direction_source=_direction_source(tmp_path),
        runtime_root=runtime,
        languages=("en", "ru", "zh", "es", "fr"),
        schedule_seed=17,
        schedule_capacity=10,
        expected_per_direction=5,
    )

    manifest = prepare_clean_reference_runtime(
        bundle=bundle,
        runtime_root=runtime,
        model=_ReferenceModel(),
        model_fingerprint="model-fingerprint",
        max_response_length=512,
        batch_size=7,
    )

    assert manifest["status"] == "PASS"
    assert manifest["rows"] == len(bundle.trial_rows)
    assert manifest["safe_rows_with_nll"] == len(bundle.trial_rows) // 2
    assert (
        prepare_clean_reference_runtime(
            bundle=bundle,
            runtime_root=runtime,
            model=_ReferenceModel(),
            model_fingerprint="model-fingerprint",
            max_response_length=512,
            batch_size=7,
        )
        == manifest
    )


def test_local_model_fingerprint_hashes_weights_and_runtime_metadata(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"config-v1")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights-v1")
    (model / "README.md").write_bytes(b"ignored-card")

    first = fingerprint_local_model(model)
    second = fingerprint_local_model(model)

    assert first == second
    assert first["status"] == "PASS"
    assert first["files"] == 2
    assert len(first["model_fingerprint"]) == 64
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights-v2")
    assert (
        fingerprint_local_model(model)["model_fingerprint"]
        != first["model_fingerprint"]
    )
