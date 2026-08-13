# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preparation helpers for one-command multilingual Heretic-MOE runs."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .clean_reference_archive import build_clean_reference_archive
from .config import generation_runtime_contract
from .language_map_directions import load_direction_map_package
from .multilingual_contract import MultilingualDatasetBundle
from .multilingual_runtime import resolve_srg_runtime_contract
from .trial_language_schedule import materialize_trial_language_schedule

_FORBIDDEN_PUBLIC_KEYS = {"prompt", "response", "answer", "text"}
_MODEL_FINGERPRINT_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".onnx",
    ".pt",
    ".pth",
    ".py",
    ".safetensors",
    ".tiktoken",
    ".txt",
}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _assert_text_free(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(
                    f"public runtime manifest contains forbidden field {key}"
                )
            _assert_text_free(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_text_free(nested)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _assert_text_free(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fingerprint_local_model(model_dir: str | Path) -> dict[str, Any]:
    """Hash every runtime-relevant local model artifact for strict resume."""

    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _MODEL_FINGERPRINT_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError("local model contains no runtime artifacts")
    records = [
        {
            "name": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]
    payload = {"schema_version": 1, "artifacts": records}
    return {
        "schema_version": 1,
        "status": "PASS",
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "artifacts": records,
        "model_fingerprint": _canonical_sha256(payload),
    }


def freeze_direction_package(
    source_dir: str | Path,
    destination_dir: str | Path,
) -> dict[str, Any]:
    """Copy only the verified tensor direction package into the run runtime."""

    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    _, source_manifest = load_direction_map_package(source)
    if destination.exists():
        _, destination_manifest = load_direction_map_package(destination)
        if destination_manifest != source_manifest:
            raise ValueError("existing frozen direction package differs")
        return destination_manifest

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        shutil.copy2(source / "directions.safetensors", temporary)
        shutil.copy2(source / "manifest.json", temporary)
        _, copied_manifest = load_direction_map_package(temporary)
        if copied_manifest != source_manifest:
            raise ValueError("copied direction package differs from source")
        os.replace(temporary, destination)
        return copied_manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def freeze_srg_calibration_package(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    expected_prompt_rows: int | None = None,
) -> dict[str, Any]:
    """Copy a completed calibration into a portable frozen package."""

    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    source_contract = resolve_srg_runtime_contract(
        source,
        expected_prompt_rows=expected_prompt_rows,
    )
    source_profile = source / "calibration_profile.json"
    if not source_profile.is_file():
        raise FileNotFoundError(source_profile)
    source_profile_sha = _sha256(source_profile)

    if destination.exists():
        resolved = resolve_srg_runtime_contract(
            destination,
            expected_prompt_rows=expected_prompt_rows,
        )
        destination_profile = destination / "calibration_profile.json"
        if (
            _sha256(destination_profile) != source_profile_sha
            or resolved["prototype_sha256"] != source_contract["prototype_sha256"]
            or resolved["prompt_sha256"] != source_contract["prompt_sha256"]
        ):
            raise ValueError("existing frozen SRG calibration package differs")
        return json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, temporary)
        prototype_target = temporary / "prototypes.jsonl"
        shutil.copy2(source_contract["prototype_path"], prototype_target)
        prompt_target = temporary / "private" / "evaluation_prompts.jsonl"
        prompt_target.parent.mkdir(parents=True, exist_ok=True)
        if Path(source_contract["prompt_path"]).resolve() != prompt_target.resolve():
            shutil.copy2(source_contract["prompt_path"], prompt_target)
        manifest_path = temporary / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "PASS",
                "prototype_path": str(destination / "prototypes.jsonl"),
                "prototype_sha256": _sha256(prototype_target),
                "prompt_path": str(
                    destination / "private" / "evaluation_prompts.jsonl"
                ),
                "prompt_sha256": _sha256(prompt_target),
                "prompt_rows": int(source_contract["prompt_rows"]),
                "validate_prompt_alignment": False,
                "portable_package": True,
                "calibration_profile_sha256": source_profile_sha,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        resolve_srg_runtime_contract(
            destination,
            expected_prompt_rows=expected_prompt_rows,
        )
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_static_multilingual_runtime(
    *,
    bundle: MultilingualDatasetBundle,
    direction_source: str | Path,
    srg_source: str | Path,
    runtime_root: str | Path,
    languages: Sequence[str],
    schedule_seed: int,
    schedule_capacity: int,
    expected_per_direction: int,
) -> dict[str, Any]:
    """Freeze all model-independent v3 artifacts with exact resume checks."""

    if bundle.manifest.get("status") != "PASS":
        raise ValueError("multilingual dataset bundle is not PASS")
    dataset_sha = str(bundle.manifest.get("contract_sha256", ""))
    if len(dataset_sha) != 64:
        raise ValueError("multilingual dataset contract hash is invalid")
    normalized_languages = tuple(str(value).lower() for value in languages)
    if not normalized_languages or len(set(normalized_languages)) != len(
        normalized_languages
    ):
        raise ValueError("languages must be a non-empty unique sequence")

    root = Path(runtime_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    direction_manifest = freeze_direction_package(
        direction_source,
        root / "clean_map" / "directions",
    )
    rows_per_cell = bundle.manifest.get("rows_per_cell")
    if not isinstance(rows_per_cell, dict) or "srg_calibration" not in rows_per_cell:
        raise ValueError(
            "multilingual dataset manifest is missing rows_per_cell.srg_calibration"
        )
    expected_srg_rows = len(normalized_languages) * int(
        rows_per_cell["srg_calibration"]
    )
    srg_manifest = freeze_srg_calibration_package(
        srg_source,
        root / "srg_calibration",
        expected_prompt_rows=expected_srg_rows,
    )
    index = [
        {
            "canonical_id": row.canonical_id,
            "row_id": row.row_id,
            "language": row.language,
            "direction_class": row.direction,
            "category_id": row.category_id,
        }
        for row in bundle.trial_rows
    ]
    schedule_manifest = materialize_trial_language_schedule(
        index,
        output_dir=root / "study" / "schedule",
        languages=normalized_languages,
        seed=schedule_seed,
        total_trials=schedule_capacity,
        expected_per_direction=expected_per_direction,
    )
    dataset_manifest_path = root / "dataset" / "manifest.json"
    if dataset_manifest_path.is_file():
        existing_dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if existing_dataset != bundle.manifest:
            raise ValueError("existing frozen dataset manifest differs")
    else:
        _write_json_atomic(dataset_manifest_path, dict(bundle.manifest))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_sha256": dataset_sha,
        "direction_package_sha256": str(direction_manifest["package_sha256"]),
        "srg_profile_sha256": str(srg_manifest["calibration_profile_sha256"]),
        "srg_prompt_rows": int(srg_manifest["prompt_rows"]),
        "schedule_contract_sha256": str(schedule_manifest["schedule_contract_sha256"]),
        "schedule_trials": int(schedule_manifest["trials"]),
        "rows_per_trial": int(schedule_manifest["rows_per_trial"]),
        "languages": list(normalized_languages),
    }
    manifest["static_runtime_sha256"] = _canonical_sha256(manifest)
    manifest_path = root / "static_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing static multilingual runtime differs")
        return existing
    _write_json_atomic(manifest_path, manifest)
    return manifest


def prepare_clean_reference_runtime(
    *,
    bundle: MultilingualDatasetBundle,
    runtime_root: str | Path,
    model: Any,
    model_fingerprint: str,
    max_response_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Generate the full 4,000-row clean trial reference exactly once."""

    root = Path(runtime_root).resolve()
    static_manifest_path = root / "static_manifest.json"
    if not static_manifest_path.is_file():
        raise FileNotFoundError(static_manifest_path)
    static_manifest = json.loads(static_manifest_path.read_text(encoding="utf-8"))
    if static_manifest.get("status") != "PASS" or static_manifest.get(
        "dataset_contract_sha256"
    ) != bundle.manifest.get("contract_sha256"):
        raise ValueError("static runtime and dataset bundle differ")
    profile, direction_manifest = load_direction_map_package(
        root / "clean_map" / "directions"
    )
    return build_clean_reference_archive(
        model=model,
        rows=bundle.trial_rows,
        refusal_direction=profile.consensus_refusal_direction,
        output_dir=root / "clean_trial_reference",
        dataset_contract_sha256=str(bundle.manifest["contract_sha256"]),
        direction_sha256=str(direction_manifest["package_sha256"]),
        model_fingerprint=model_fingerprint,
        max_response_length=max_response_length,
        batch_size=batch_size,
        generation_contract=generation_runtime_contract(
            getattr(model, "settings", object())
        ),
    )
