# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verified assembly of frozen multilingual search worker artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .clean_reference_archive import load_clean_reference_archive
from .config import DatasetSpecification, SelectionPolicy, Settings
from .language_map_data import GeometryRow
from .language_map_directions import DirectionMapProfile, load_direction_map_package
from .multilingual_contract import (
    MultilingualDatasetBundle,
    load_multilingual_dataset_bundle,
)
from .multilingual_search_evaluator import (
    MultilingualConstraintContract,
    MultilingualSearchEvaluator,
)
from .multilingual_final_holdout import load_final_holdout_archive
from .multilingual_finalist_evaluator import MultilingualFinalistEvaluator
from .multilingual_trial_evaluator import FrozenMultilingualTrialEvaluator
from .trial_language_schedule import load_trial_language_schedule


@dataclass(frozen=True)
class MultilingualWorkerRuntime:
    bundle: MultilingualDatasetBundle
    evaluator: MultilingualSearchEvaluator
    direction_profile: DirectionMapProfile
    manifest: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_multilingual_search_mode(settings: Settings) -> None:
    """Pin v3 generation and ranking so legacy 136-row settings cannot leak in."""

    contract = settings.multilingual_search
    if not contract.enabled:
        return
    settings.max_response_length = contract.ordinary_max_new_tokens
    settings.primary_objective = "Removal"
    settings.selection_policy = SelectionPolicy.FEASIBLE_DIVERSE
    settings.selection_score_targets = {}
    settings.selection_score_weights = {}


def resolve_srg_runtime_contract(runtime_dir: str | Path) -> dict[str, Any]:
    """Resolve only the pinned 660-row calibration scorer inputs."""

    root = Path(runtime_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or int(manifest.get("prompt_rows", -1)) != 660:
        raise ValueError("SRG runtime manifest is not a completed 660-row contract")
    prototype_path = Path(str(manifest.get("prototype_path", ""))).resolve()
    prompt_path = Path(str(manifest.get("prompt_path", ""))).resolve()
    for label, path, expected in (
        ("prototype", prototype_path, manifest.get("prototype_sha256")),
        ("prompt", prompt_path, manifest.get("prompt_sha256")),
    ):
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"SRG runtime {label} file hash mismatch")
    if bool(manifest.get("validate_prompt_alignment")):
        raise ValueError("multilingual SRG runtime must disable numeric prompt alignment")
    return {
        "prototype_path": prototype_path,
        "prototype_sha256": str(manifest["prototype_sha256"]),
        "prompt_path": prompt_path,
        "prompt_sha256": str(manifest["prompt_sha256"]),
        "prompt_rows": 660,
        "top_k": int(manifest["top_k"]),
        "min_df": int(manifest["min_df"]),
        "max_response_length": int(manifest["max_response_length"]),
        "validate_prompt_alignment": False,
    }


def build_multilingual_srg_scorer(
    settings: Settings,
    model: Any,
    runtime_root: str | Path,
) -> Any:
    """Initialize the sparse scorer from the completed v3 calibration contract."""

    from .scorer import Context
    from .scorers.sparse_refusal_geometry import (
        Settings as SparseSettings,
        SparseRefusalGeometry,
    )

    contract = resolve_srg_runtime_contract(
        Path(runtime_root).resolve() / "srg_calibration"
    )
    scorer_settings = SparseSettings(
        prototypes=str(contract["prototype_path"]),
        prototypes_sha256=str(contract["prototype_sha256"]),
        prompts=DatasetSpecification(
            dataset=str(contract["prompt_path"]),
            column="prompt",
        ),
        top_k=int(contract["top_k"]),
        min_df=int(contract["min_df"]),
        validate_prompt_alignment=False,
    )
    calibration_settings = settings.model_copy(deep=True)
    calibration_settings.max_response_length = int(
        contract["max_response_length"]
    )
    scorer = SparseRefusalGeometry(
        heretic_settings=calibration_settings,
        settings=scorer_settings,
    )
    scorer.init(Context(settings=calibration_settings, model=model))
    return scorer


def load_multilingual_worker_runtime(
    settings: Settings,
    model: Any,
    *,
    srg_scorer: Any | None = None,
) -> MultilingualWorkerRuntime:
    """Load the dataset and immutable worker packages for the v3 search path."""

    contract = settings.multilingual_search
    if not contract.enabled:
        raise ValueError("multilingual search is not enabled")
    if not contract.runtime_root:
        raise ValueError("multilingual_search.runtime_root is required for workers")
    bundle = load_multilingual_dataset_bundle(
        dataset_root=str(contract.dataset_root),
        split_root=contract.split_root,
        languages=tuple(contract.languages),
        direction_rows_per_cell=contract.direction_rows_per_cell,
        trial_rows_per_cell=contract.trial_rows_per_cell,
        calibration_rows_per_language=contract.calibration_rows_per_language,
    )
    scorer = srg_scorer or build_multilingual_srg_scorer(
        settings,
        model,
        contract.runtime_root,
    )
    constraints = MultilingualConstraintContract(
        max_safe_ppl_drift=float(contract.max_safe_ppl_drift),
        max_safe_geometry_damage=float(contract.max_safe_geometry_damage),
        max_language_instability=float(contract.max_language_instability),
        max_category_instability=float(contract.max_category_instability),
    )
    if contract.evaluation_phase == "finalist":
        evaluator, manifest = load_multilingual_finalist_evaluator(
            bundle=bundle,
            runtime_root=contract.runtime_root,
            model=model,
            srg_scorer=scorer,
            constraints=constraints,
            expected_languages=tuple(contract.languages),
            final_max_new_tokens=contract.final_max_new_tokens,
        )
    else:
        evaluator, manifest = load_multilingual_search_evaluator(
            bundle=bundle,
            runtime_root=contract.runtime_root,
            model=model,
            srg_scorer=scorer,
            constraints=constraints,
            expected_per_direction=contract.trial_rows_per_cell,
            expected_languages=tuple(contract.languages),
        )
    direction_profile, _ = load_direction_map_package(
        Path(contract.runtime_root).resolve() / "clean_map" / "directions"
    )
    return MultilingualWorkerRuntime(
        bundle=bundle,
        evaluator=evaluator,
        direction_profile=direction_profile,
        manifest=manifest,
    )


def recommended_model_layer_bounds(
    profile: DirectionMapProfile,
    model_layer_count: int,
) -> tuple[float, float]:
    """Translate residual indices (embedding first) to model-layer coordinates."""

    if model_layer_count <= 0:
        raise ValueError("model_layer_count must be positive")
    start, end = profile.recommended_layer_bounds
    low = max(0, min(model_layer_count - 1, int(start) - 1))
    high = max(low, min(model_layer_count - 1, int(end) - 1))
    return float(low), float(high)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _trial_index_sha256(rows: Sequence[GeometryRow]) -> str:
    return _canonical_sha256(
        [
            {
                "canonical_id": row.canonical_id,
                "row_id": row.row_id,
                "language": row.language.lower(),
                "direction_class": row.direction.lower(),
                "category_id": row.category_id,
            }
            for row in rows
        ]
    )


def _load_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        profile.get("schema_version") != 2
        or profile.get("status") != "PASS"
        or int(profile.get("rows", -1)) <= 0
        or not isinstance(profile.get("group_scale"), dict)
        or not isinstance(profile.get("group_weight"), dict)
    ):
        raise ValueError("SRG calibration profile is invalid")
    return profile


def load_multilingual_search_evaluator(
    *,
    bundle: MultilingualDatasetBundle,
    runtime_root: str | Path,
    model: Any,
    srg_scorer: Any,
    constraints: MultilingualConstraintContract,
    expected_per_direction: int = 400,
    expected_languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
) -> tuple[MultilingualSearchEvaluator, dict[str, Any]]:
    """Load, cross-check and wire every immutable worker-side artifact."""

    root = Path(runtime_root).resolve()
    profile, direction_manifest = load_direction_map_package(
        root / "clean_map" / "directions"
    )
    schedule_manifest, schedule_records = load_trial_language_schedule(
        root / "study" / "schedule"
    )
    clean_manifest, clean_records = load_clean_reference_archive(
        root / "clean_trial_reference"
    )
    srg_profile_path = root / "srg_calibration" / "calibration_profile.json"
    srg_profile = _load_profile(srg_profile_path)

    dataset_sha = str(bundle.manifest.get("contract_sha256", ""))
    if clean_manifest.get("dataset_contract_sha256") != dataset_sha:
        raise ValueError("clean reference dataset contract mismatch")
    if clean_manifest.get("direction_sha256") != direction_manifest.get(
        "package_sha256"
    ):
        raise ValueError("clean reference direction contract mismatch")
    if schedule_manifest.get("index_sha256") != _trial_index_sha256(
        bundle.trial_rows
    ):
        raise ValueError("trial schedule dataset index mismatch")
    expected_row_ids = [row.row_id for row in bundle.trial_rows]
    if [record.get("row_id") for record in clean_records] != expected_row_ids:
        raise ValueError("clean reference row order differs from trial corpus")
    if int(clean_manifest.get("layers", -1)) != int(
        profile.consensus_refusal_direction.shape[0]
    ) or int(clean_manifest.get("hidden_size", -1)) != int(
        profile.consensus_refusal_direction.shape[1]
    ):
        raise ValueError("clean reference geometry shape mismatch")

    private_output = root / "study" / "response_archive"
    frozen_runtime = FrozenMultilingualTrialEvaluator(
        model=model,
        trial_rows=bundle.trial_rows,
        schedule_records=schedule_records,
        clean_records=clean_records,
        refusal_direction=profile.consensus_refusal_direction,
        layer_reliability=profile.layer_reliability,
        srg_scorer=srg_scorer,
        srg_profile=srg_profile,
        private_output_dir=private_output,
        expected_per_direction=expected_per_direction,
        expected_languages=expected_languages,
    )
    evaluator = MultilingualSearchEvaluator(
        frozen_runtime,
        constraints=constraints,
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_sha256": dataset_sha,
        "direction_package_sha256": direction_manifest["package_sha256"],
        "schedule_contract_sha256": schedule_manifest[
            "schedule_contract_sha256"
        ],
        "clean_reference_contract_sha256": clean_manifest[
            "archive_contract_sha256"
        ],
        "srg_profile_sha256": hashlib.sha256(
            srg_profile_path.read_bytes()
        ).hexdigest(),
        "schedule_trials": int(schedule_manifest["trials"]),
        "trial_rows": len(bundle.trial_rows),
        "objectives": ["Removal", "Preservation loss"],
        "constraints": evaluator.get_constraint_names(),
    }
    manifest["runtime_contract_sha256"] = _canonical_sha256(manifest)
    return evaluator, manifest


def load_multilingual_finalist_evaluator(
    *,
    bundle: MultilingualDatasetBundle,
    runtime_root: str | Path,
    model: Any,
    srg_scorer: Any,
    constraints: MultilingualConstraintContract,
    expected_languages: tuple[str, ...] = ("en", "ru", "zh", "es", "fr"),
    final_max_new_tokens: int = 1024,
) -> tuple[MultilingualSearchEvaluator, dict[str, Any]]:
    """Wire the all-translation trial pool and post-freeze R holdout."""

    root = Path(runtime_root).resolve()
    profile, direction_manifest = load_direction_map_package(
        root / "clean_map" / "directions"
    )
    clean_manifest, clean_records = load_clean_reference_archive(
        root / "clean_trial_reference"
    )
    final_manifest, final_records = load_final_holdout_archive(
        root / "final_holdout_reference"
    )
    srg_profile_path = root / "srg_calibration" / "calibration_profile.json"
    srg_profile = _load_profile(srg_profile_path)
    if clean_manifest.get("dataset_contract_sha256") != bundle.manifest.get(
        "contract_sha256"
    ) or final_manifest.get("dataset_contract_sha256") != bundle.manifest.get(
        "contract_sha256"
    ):
        raise ValueError("finalist reference dataset contract mismatch")
    if clean_manifest.get("direction_sha256") != direction_manifest.get(
        "package_sha256"
    ):
        raise ValueError("finalist direction contract mismatch")
    expected_trial_ids = [row.row_id for row in bundle.trial_rows]
    if [record.get("row_id") for record in clean_records] != expected_trial_ids:
        raise ValueError("full trial reference order differs from trial pool")
    expected_final_ids = [row.row_id for row in bundle.final_rows]
    if [record.get("row_id") for record in final_records] != expected_final_ids:
        raise ValueError("final-holdout reference order differs from R pool")
    languages = tuple(language.lower() for language in expected_languages)
    if not languages or len(bundle.trial_rows) % (2 * len(languages)):
        raise ValueError("full trial pool cannot balance languages and directions")
    expected_per_direction = len(bundle.trial_rows) // 2
    runtime = MultilingualFinalistEvaluator(
        model=model,
        trial_rows=bundle.trial_rows,
        clean_trial_records=clean_records,
        final_rows=bundle.final_rows,
        clean_final_records=final_records,
        refusal_direction=profile.consensus_refusal_direction,
        layer_reliability=profile.layer_reliability,
        srg_scorer=srg_scorer,
        srg_profile=srg_profile,
        private_output_dir=root / "recheck" / "private_responses",
        expected_per_direction=expected_per_direction,
        expected_languages=languages,
        final_max_new_tokens=final_max_new_tokens,
    )
    evaluator = MultilingualSearchEvaluator(runtime, constraints=constraints)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "evaluation_phase": "finalist",
        "dataset_contract_sha256": bundle.manifest["contract_sha256"],
        "direction_package_sha256": direction_manifest["package_sha256"],
        "clean_reference_contract_sha256": clean_manifest[
            "archive_contract_sha256"
        ],
        "final_holdout_contract_sha256": final_manifest[
            "archive_contract_sha256"
        ],
        "srg_profile_sha256": hashlib.sha256(srg_profile_path.read_bytes()).hexdigest(),
        "trial_rows_per_finalist": len(bundle.trial_rows),
        "final_holdout_rows": len(bundle.final_rows),
        "final_max_new_tokens": int(final_max_new_tokens),
        "objectives": ["Removal", "Preservation loss"],
        "constraints": evaluator.get_constraint_names(),
    }
    manifest["runtime_contract_sha256"] = _canonical_sha256(manifest)
    return evaluator, manifest
