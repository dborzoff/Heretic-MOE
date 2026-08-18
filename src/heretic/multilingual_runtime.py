# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verified assembly of frozen multilingual search worker artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clean_reference_archive import load_clean_reference_archive
from .config import SelectionPolicy, Settings, generation_runtime_contract
from .language_map_data import GeometryRow
from .language_map_directions import DirectionMapProfile, load_direction_map_package
from .multilingual_contract import (
    MultilingualDatasetBundle,
    load_multilingual_dataset_bundle,
)
from .multilingual_finalist_evaluator import MultilingualFinalistEvaluator
from .multilingual_search_evaluator import (
    MultilingualConstraintContract,
    MultilingualSearchEvaluator,
)
from .multilingual_trial_evaluator import FrozenMultilingualTrialEvaluator
from .trial_language_schedule import load_trial_language_schedule
from .utils import Prompt


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
    """Pin v4 generation and ranking so legacy 136-row settings cannot leak in."""

    contract = settings.multilingual_search
    if not contract.enabled:
        return
    settings.max_response_length = contract.ordinary_max_new_tokens
    settings.primary_objective = "Removal"
    settings.selection_policy = SelectionPolicy.FEASIBLE_DIVERSE
    settings.selection_score_targets = {}
    settings.selection_score_weights = {}
    # The multilingual corpus is frozen and pretokenized when a resident worker
    # starts. Prefix probing would both waste generation and invalidate that cache.
    if settings.response_prefix is None:
        settings.response_prefix = ""


def multilingual_resident_rows(
    bundle: MultilingualDatasetBundle,
    evaluation_phase: str,
    runtime_root: str | Path,
) -> tuple[Any, ...]:
    if evaluation_phase == "finalist":
        _, schedule_records = load_trial_language_schedule(
            Path(runtime_root).resolve() / "study" / "schedule"
        )
        fixed = next(
            (record for record in schedule_records if int(record["trial_number"]) == 0),
            None,
        )
        if not isinstance(fixed, dict):
            raise ValueError("frozen finalist schedule panel is missing")
        by_id = {row.row_id: row for row in bundle.trial_rows}
        try:
            return tuple(by_id[str(row_id)] for row_id in fixed["row_ids"])
        except KeyError as error:
            raise ValueError("frozen finalist panel is outside the trial corpus") from error
    if evaluation_phase == "search":
        return tuple(bundle.trial_rows)
    raise ValueError(f"unsupported multilingual evaluation phase: {evaluation_phase}")


def resolve_srg_runtime_contract(
    runtime_dir: str | Path,
    *,
    expected_prompt_rows: int | None = None,
) -> dict[str, Any]:
    """Resolve the portable external-only SRG scorer and profile inputs."""

    root = Path(runtime_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("external_only"):
        raise ValueError("SRG runtime manifest is not an external-only contract")
    if expected_prompt_rows is not None:
        raise ValueError("external-only SRG has no fixed prompt row count")
    prototype_path = Path(str(manifest.get("prototype_path", ""))).resolve()
    profile_path = Path(str(manifest.get("profile_path", ""))).resolve()
    for label, path, expected in (
        ("prototype", prototype_path, manifest.get("prototype_sha256")),
        ("profile", profile_path, manifest.get("profile_sha256")),
    ):
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"SRG runtime {label} file hash mismatch")
    if bool(manifest.get("validate_prompt_alignment")):
        raise ValueError(
            "multilingual SRG runtime must disable numeric prompt alignment"
        )
    return {
        "prototype_path": prototype_path,
        "prototype_sha256": str(manifest["prototype_sha256"]),
        "profile_path": profile_path,
        "profile_sha256": str(manifest["profile_sha256"]),
        "external_only": True,
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
    """Initialize the sparse scorer from the built-in cross-model profile."""

    from .scorer import Context
    from .scorers.sparse_refusal_geometry import (
        Settings as SparseSettings,
    )
    from .scorers.sparse_refusal_geometry import (
        SparseRefusalGeometry,
    )

    contract = resolve_srg_runtime_contract(
        Path(runtime_root).resolve() / "srg_profile",
    )
    scorer_settings = SparseSettings(
        prototypes=str(contract["prototype_path"]),
        prototypes_sha256=str(contract["prototype_sha256"]),
        prompts=None,
        top_k=int(contract["top_k"]),
        min_df=int(contract["min_df"]),
        validate_prompt_alignment=False,
    )
    calibration_settings = settings.model_copy(deep=True)
    calibration_settings.max_response_length = int(contract["max_response_length"])
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
    """Load the dataset and immutable worker packages for the v4 search path."""

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
        final_rows_per_cell=contract.final_rows_per_cell,
    )
    prompt_cache_stats: dict[str, Any] | None = None
    if hasattr(model, "prepare_prompt_cache"):
        cache_rows: Sequence[Any] = multilingual_resident_rows(
            bundle, contract.evaluation_phase, contract.runtime_root
        )
        prepared = model.prepare_prompt_cache(
            [Prompt(system="", user=row.prompt) for row in cache_rows]
        )
        prompt_cache_stats = {
            "requested_rows": int(prepared["rows"]),
            "unique_requested_rows": int(prepared["unique"]),
            "new_rows": int(prepared["new"]),
        }
        if hasattr(model, "pin_prompt_cache"):
            packed = model.pin_prompt_cache()
            prompt_cache_stats = {
                **prompt_cache_stats,
                "cached_rows": int(packed["rows"]),
                "tokens": int(packed["tokens"]),
                "pinned": bool(packed["pinned"]),
            }
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
        max_empty_response_rate=float(contract.max_empty_response_rate),
        max_truncated_response_rate=float(contract.max_truncated_response_rate),
        max_safe_d_to_r_rate=float(contract.max_safe_d_to_r_rate),
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
            expected_generation_contract=generation_runtime_contract(settings),
        )
    else:
        evaluator, manifest = load_multilingual_search_evaluator(
            bundle=bundle,
            runtime_root=contract.runtime_root,
            model=model,
            srg_scorer=scorer,
            constraints=constraints,
            expected_per_direction=contract.trial_rows_per_direction,
            expected_languages=tuple(contract.languages),
            expected_generation_contract=generation_runtime_contract(settings),
        )
    if prompt_cache_stats is not None:
        model._last_prompt_cache_stats = prompt_cache_stats
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


def _trial_index_sha256(
    rows: Sequence[GeometryRow],
    *,
    include_behavior: bool = False,
) -> str:
    return _canonical_sha256(
        [
            {
                "canonical_id": row.canonical_id,
                "row_id": row.row_id,
                "language": row.language.lower(),
                "direction_class": row.direction.lower(),
                "category_id": row.category_id,
                **(
                    {"trial_behavior_class": row.trial_behavior_class}
                    if include_behavior
                    else {}
                ),
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
        raise ValueError("built-in SRG profile is invalid")
    return profile


def load_multilingual_search_evaluator(
    *,
    bundle: MultilingualDatasetBundle,
    runtime_root: str | Path,
    model: Any,
    srg_scorer: Any,
    constraints: MultilingualConstraintContract,
    expected_per_direction: int = 400,
    expected_languages: tuple[str, ...] = ("en", "ru", "zh", "ja"),
    expected_generation_contract: Mapping[str, object] | None = None,
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
    srg_profile_path = root / "srg_profile" / "calibration_profile.json"
    srg_profile = _load_profile(srg_profile_path)

    dataset_sha = str(bundle.manifest.get("contract_sha256", ""))
    if clean_manifest.get("dataset_contract_sha256") != dataset_sha:
        raise ValueError("clean reference dataset contract mismatch")
    if expected_generation_contract is not None and dict(
        clean_manifest.get("generation_contract", {})
    ) != dict(expected_generation_contract):
        raise ValueError("clean reference generation backend contract mismatch")
    if clean_manifest.get("direction_sha256") != direction_manifest.get(
        "package_sha256"
    ):
        raise ValueError("clean reference direction contract mismatch")
    if schedule_manifest.get("index_sha256") != _trial_index_sha256(
        bundle.trial_rows,
        include_behavior=int(schedule_manifest.get("schema_version", 1)) >= 3,
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
        max_response_length=int(
            expected_generation_contract.get("max_response_length", 100)
        )
        if expected_generation_contract is not None
        else 100,
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
        "schedule_contract_sha256": schedule_manifest["schedule_contract_sha256"],
        "clean_reference_contract_sha256": clean_manifest["archive_contract_sha256"],
        "srg_profile_sha256": hashlib.sha256(srg_profile_path.read_bytes()).hexdigest(),
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
    expected_languages: tuple[str, ...] = ("en", "ru", "zh", "ja"),
    final_max_new_tokens: int = 1024,
    expected_generation_contract: Mapping[str, object] | None = None,
) -> tuple[MultilingualSearchEvaluator, dict[str, Any]]:
    """Wire one identical frozen 400-row panel for every finalist."""

    root = Path(runtime_root).resolve()
    schedule_manifest, schedule_records = load_trial_language_schedule(
        root / "study" / "schedule"
    )
    fixed_schedule_trial_number = 0
    fixed_record = next(
        (
            record
            for record in schedule_records
            if int(record["trial_number"]) == fixed_schedule_trial_number
        ),
        None,
    )
    if not isinstance(fixed_record, dict):
        raise TypeError("frozen finalist schedule panel is missing")
    fixed_row_ids = tuple(str(value) for value in fixed_record["row_ids"])
    if not fixed_row_ids or len(fixed_row_ids) % 2:
        raise ValueError("frozen finalist schedule panel must balance directions")
    expected_per_direction = len(fixed_row_ids) // 2
    base_evaluator, base_manifest = load_multilingual_search_evaluator(
        bundle=bundle,
        runtime_root=root,
        model=model,
        srg_scorer=srg_scorer,
        constraints=constraints,
        expected_per_direction=expected_per_direction,
        expected_languages=expected_languages,
        expected_generation_contract=expected_generation_contract,
    )
    frozen_runtime = base_evaluator.runtime
    frozen_runtime.private_output_dir = root / "recheck" / "private_responses"
    runtime = MultilingualFinalistEvaluator(
        runtime=frozen_runtime,
        fixed_schedule_trial_number=fixed_schedule_trial_number,
    )
    evaluator = MultilingualSearchEvaluator(runtime, constraints=constraints)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "evaluation_phase": "finalist",
        "dataset_contract_sha256": base_manifest["dataset_contract_sha256"],
        "direction_package_sha256": base_manifest["direction_package_sha256"],
        "schedule_contract_sha256": schedule_manifest["schedule_contract_sha256"],
        "clean_reference_contract_sha256": base_manifest[
            "clean_reference_contract_sha256"
        ],
        "srg_profile_sha256": base_manifest["srg_profile_sha256"],
        "fixed_schedule_trial_number": fixed_schedule_trial_number,
        "fixed_panel_row_ids_sha256": _canonical_sha256(fixed_row_ids),
        "trial_rows_per_finalist": len(fixed_row_ids),
        "final_max_new_tokens": int(final_max_new_tokens),
        "objectives": ["Removal", "Preservation loss"],
        "constraints": evaluator.get_constraint_names(),
    }
    manifest["runtime_contract_sha256"] = _canonical_sha256(manifest)
    return evaluator, manifest
